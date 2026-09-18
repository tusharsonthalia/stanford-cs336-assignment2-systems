#!/usr/bin/env bash
# =============================================================================
# uv environment setup for this repo inside the ppi-bench pod.
#
#   source ./setup-uv.sh                 # set up this repo (the usual case)
#   source ./setup-uv.sh /jfs/other/repo # also set up other /jfs projects
#   source ./setup-uv.sh --env-only      # vars + wrapper only (used by ~/.bashrc)
#
# Run it once after the pod is recreated. Source it rather than executing it,
# so the `uv` wrapper lands in your current shell.
#
# Paths are derived from this file's own location, so moving the repo is fine.
#
# -----------------------------------------------------------------------------
# Why it is shaped this way
# -----------------------------------------------------------------------------
# Code lives on /jfs so git tracks it. Virtualenvs must not. Measured on this
# pod (torch 2.11 + cu130, 78 packages, ~4.8G):
#
#   cache + venv both on /jfs  : 24m04s to build cold
#   venv on /jfs               : import torch 4.6s warm, ~91s cold (fresh pod)
#                                full project import 11-14s warm
#   venv on local disk         : import torch 1.5s, full project import 2.8s
#   rebuild local venv from the warm /jfs cache: ~17-35s
#
# So the ~4.9G wheel cache persists here on /jfs in .uv-cache (the expensive
# thing to re-fetch), and each project's venv is built on LOCAL disk, under
# $HOME/.venvs/<repo>-<hash of its path>.
#
# This repo is shared over /jfs by two machines with different homes:
#   ppi-bench pod          HOME=/home/ray      (GPU, uv 0.11.7)
#   tushar-model-01-0 host HOME=/home/jovyan   (uv 0.7.13)
# so the venv path is derived from $HOME at call time. Run this script on each
# machine you use; the venvs are independent and neither can see the other's.
#
# Three gotchas that shaped this:
#
#   1. uv will NOT create a venv through a dangling symlink -- it fails with
#      "File exists (os error 17)". A pod recreate wipes $HOME, and the other
#      machine's $HOME never existed here, so <repo>/.venv is dangling as often
#      as not. Hence uv is always handed an explicit UV_PROJECT_ENVIRONMENT and
#      never left to discover .venv on its own.
#
#   2. Reaching a venv *through* a /jfs symlink makes sys.prefix a /jfs path, so
#      every site-packages lookup pays a FUSE round trip: 2.2s vs 1.6s for
#      `import torch`, 4.3s vs 3.0s for a full import. The wrapper passes the
#      real local path instead.
#
#   3. <repo>/.venv is still maintained, but as a pure convenience for
#      `source .venv/bin/activate` and the VS Code interpreter picker. It points
#      at whichever machine last ran setup and is dangling on the other one.
#      Nothing depends on it.
#
# Note: .uv-cache and .venv are dot-prefixed, so `make_submission.sh` already
# excludes them via its `-x '.*'` rule. They stay out of the submission zip.
# =============================================================================

# --- locate ourselves --------------------------------------------------------
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _UV_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
else
    _UV_SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
fi
_UV_REPO="$(dirname "$_UV_SELF")"
_UV_VENV_ROOT="$HOME/.venvs"   # local disk on whichever machine you are on

# =============================================================================
# 1. Environment + `uv` wrapper  (always applied; cheap, no side effects)
# =============================================================================

# Where the wheel cache goes depends on whether THIS machine's $HOME survives.
#
# The /jfs cache exists only because the pod's $HOME is wiped on every recreate.
# The notebook host's $HOME is a separate persistent disk, so there it is pure
# downside: writing a multi-GB cache through FUSE is slow, and uv versions do
# not share cache layouts anyway (uv 0.7 uses wheels-v5/simple-v21, uv 0.11 uses
# wheels-v6), so the host would duplicate the whole thing rather than reuse it.
#
# Test: if $HOME is on the same filesystem as /, it is the container's ephemeral
# overlay and needs the /jfs cache. If it is a separate device, it persists and
# uv's own default (~/.cache/uv, local and fast) is correct.
_uv_fs() { df -P "$1" 2>/dev/null | tail -1 | awk '{print $1}'; }
if [ "$(_uv_fs "$HOME")" = "$(_uv_fs /)" ]; then
    export UV_CACHE_DIR="$_UV_REPO/.uv-cache"   # ephemeral $HOME (the pod)
else
    unset UV_CACHE_DIR                          # persistent $HOME (notebook host)
fi
unset -f _uv_fs

# NOT exporting UV_PROJECT_ENVIRONMENT globally: it takes precedence over each
# project's .venv symlink, so one global value would silently force every /jfs
# project into a single shared virtualenv. Set per-invocation here instead.
#
# This is a pure optimisation -- without it everything still works, just at the
# slower "through the symlink" timings above.
# Where a given project's venv lives ON THIS MACHINE. The repo is shared over
# /jfs between the ppi-bench pod (HOME=/home/ray) and the notebook host
# (HOME=/home/jovyan), so this must resolve per-machine. The path is derived,
# never read from the .venv symlink -- that symlink points at whichever machine
# last ran setup, and is dangling on the other one.
_uv_env_path() {
    local root="$1"
    printf '%s/.venvs/%s-%s' "$HOME" "$(basename "$root")" \
        "$(printf '%s' "$root" | md5sum | cut -c1-8)"
}

_uv_project_root() {
    local root=$PWD
    while [ "$root" != "/" ] && [ ! -f "$root/pyproject.toml" ]; do
        root=$(dirname "$root")
    done
    [ -f "$root/pyproject.toml" ] && printf '%s' "$root"
}

uv() {
    local root
    root=$(_uv_project_root)
    if [ -n "$root" ]; then
        UV_PROJECT_ENVIRONMENT="$(_uv_env_path "$root")" command uv "$@"
        return $?
    fi
    command uv "$@"
}

if [ "${1:-}" = "--env-only" ]; then
    unset _UV_SELF _UV_REPO _UV_VENV_ROOT
    return 0 2>/dev/null || exit 0
fi

# =============================================================================
# 2. Build the environments
# =============================================================================

if [ "${BASH_SOURCE[0]:-$0}" = "${0}" ]; then
    echo "setup-uv: NOTE - executed, not sourced. Virtualenvs will be built"
    echo "          correctly, but the 'uv' wrapper won't reach your shell."
    echo "          Prefer: source $_UV_SELF"
fi

# Make future shells in this pod pick up the wrapper ($HOME is wiped on recreate).
#
# Both files, and .bashrc gets it PREPENDED, for a reason: `bash -l` (login,
# e.g. `kubectl exec ... -- bash -lc`) reads .profile and never .bashrc, while
# .bashrc itself starts with `case $- in *i*) ;; *) return;; esac` and bails out
# early for non-interactive shells. Appending to .bashrc alone silently leaves
# the wrapper undefined in those shells, which costs ~45% on every import.
_UV_LINE="source $_UV_SELF --env-only"
_UV_NOTE="# uv: cache on /jfs beside the code, venvs on local disk"

if ! grep -qF "setup-uv.sh" "$HOME/.bashrc" 2>/dev/null; then
    printf '%s\n%s\n\n' "$_UV_NOTE" "$_UV_LINE" > "$HOME/.bashrc.uvtmp"
    [ -f "$HOME/.bashrc" ] && cat "$HOME/.bashrc" >> "$HOME/.bashrc.uvtmp"
    mv "$HOME/.bashrc.uvtmp" "$HOME/.bashrc"
    echo "setup-uv: added setup-uv.sh to ~/.bashrc (prepended, above the non-interactive guard)"
fi

if ! grep -qF "setup-uv.sh" "$HOME/.profile" 2>/dev/null; then
    printf '\n%s\n%s\n' "$_UV_NOTE" "$_UV_LINE" >> "$HOME/.profile"
    echo "setup-uv: added setup-uv.sh to ~/.profile (login shells)"
fi

# Keep the wheel cache out of `git status` without touching the tracked
# .gitignore (which would show up in your diff and your submission).
_UV_EXCLUDE="$_UV_REPO/.git/info/exclude"
if [ -f "$_UV_EXCLUDE" ] && ! grep -q "^\.uv-cache/$" "$_UV_EXCLUDE" 2>/dev/null; then
    printf '\n# local uv wheel cache + backups (see setup-uv.sh)\n.uv-cache/\n*.bak\n' >> "$_UV_EXCLUDE"
    echo "setup-uv: added .uv-cache/ to .git/info/exclude"
fi

_uv_projects=("$@")
[ ${#_uv_projects[@]} -eq 0 ] && _uv_projects=("$_UV_REPO")

mkdir -p "$_UV_VENV_ROOT"

for _repo in "${_uv_projects[@]}"; do
    _repo="$(cd "$_repo" 2>/dev/null && pwd)" || { echo "setup-uv: no such dir" >&2; continue; }
    if [ ! -f "$_repo/pyproject.toml" ]; then
        echo "setup-uv: skipping $_repo (no pyproject.toml)" >&2
        continue
    fi

    # Disambiguate same-named repos living at different paths.
    _target="$(_uv_env_path "$_repo")"

    if [ -e "$_repo/.venv" ] && [ ! -L "$_repo/.venv" ]; then
        echo "setup-uv: $_repo/.venv is a real directory, not a symlink." >&2
        echo "          Refusing to touch it -- remove it yourself first." >&2
        continue
    fi

    mkdir -p "$_target" || { echo "setup-uv: cannot create $_target" >&2; continue; }

    # Best-effort convenience only: makes `source .venv/bin/activate` and the
    # VS Code interpreter picker work on THIS machine. It flips to whichever
    # machine last ran setup, and nothing above depends on it -- uv is always
    # given an explicit UV_PROJECT_ENVIRONMENT, so a dangling .venv is harmless.
    # NOTE: this used to be `ln -sfn "$_target" "$_repo/.venv"`, which repointed
    # <repo>/.venv at whichever machine ran setup last. That broke the VS Code
    # Python extension on the *other* machine every single time, because Pylance
    # discovers the interpreter via ${workspaceFolder}/.venv and silently falls
    # back to a system Python when it dangles. .venv is now left alone: it is
    # pinned to the notebook host (where the editor runs) and never rewritten.
    # Nothing here depends on it -- uv is always handed an explicit
    # UV_PROJECT_ENVIRONMENT (see gotcha 1 in the header).

    [ -z "${_UV_PRIMARY:-}" ] && _UV_PRIMARY="$_target"
    echo "setup-uv: syncing $_repo"
    echo "          .venv -> $_target"
    _start=$(date +%s)
    if (cd "$_repo" && UV_PROJECT_ENVIRONMENT="$_target" command uv sync); then
        echo "setup-uv: ready in $(( $(date +%s) - _start ))s"
    else
        echo "setup-uv: uv sync FAILED for $_repo" >&2
    fi
done

# Activate the first project's env in the caller's shell, so plain `python`,
# `pytest` and `ruff` work without the `uv run` prefix. Only possible when this
# file is sourced -- an executed script cannot change its parent's environment.
#
# `uv run <cmd>` works either way and does not need this; activation is purely
# for the convenience of a shell where the venv is already on PATH.
if [ "${BASH_SOURCE[0]:-$0}" != "${0}" ] && [ -n "${_UV_PRIMARY:-}" ]; then
    if [ -f "$_UV_PRIMARY/bin/activate" ]; then
        # Step out of a previously activated venv, if any. Gate on VIRTUAL_ENV:
        # under conda, `deactivate` is conda's own function and calling it here
        # just prints a DeprecationWarning and messes with their base env.
        if [ -n "${VIRTUAL_ENV:-}" ] && command -v deactivate >/dev/null 2>&1; then
            deactivate
        fi
        source "$_UV_PRIMARY/bin/activate"
        echo "setup-uv: activated $_UV_PRIMARY"
        echo "          python -> $(command -v python)"
    else
        echo "setup-uv: no activate script at $_UV_PRIMARY/bin/activate" >&2
    fi
fi

unset _uv_projects _repo _target _start _UV_EXCLUDE _UV_LINE _UV_NOTE
unset _UV_SELF _UV_REPO _UV_VENV_ROOT _UV_PRIMARY

import torch
from torch import Tensor
from jaxtyping import Float
import triton
import triton.language as tl
from einops import einsum


@triton.jit
def flash_attention_fwd_kern(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_blk_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE,D),
        order=(1,0),
    )

    k_blk_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE,D),
        order=(1,0),
    )

    v_blk_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE,D),
        order=(1,0),
    )

    o_blk_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE,0),
        block_shape=(Q_TILE_SIZE,D),
        order=(1,0),
    )

    l_blk_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Output tile, running rowsum (l), and running rowmax (m) stored in registers
    output = tl.zeros((Q_TILE_SIZE,D), tl.float32) # [B_q, D]
    l = tl.zeros((Q_TILE_SIZE,), tl.float32) # [B_q]
    m = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32) # [B_q]

    # Loading Q tile in registers for the thread block
    q_tile = tl.load(q_blk_ptr, boundary_check=(0,1), padding_option="zero") # [B_q, D]

    # Looping through K and V tiles to compute attention scores for Q tile in SRAM
    col_offsets = tl.arange(0, K_TILE_SIZE)[None, :]
    row_offsets = tl.arange(0, Q_TILE_SIZE)[:, None] + query_tile_index * Q_TILE_SIZE

    # Causal: query i may attend only to keys j <= i, so a query tile can never
    # reach past its own last query.
    key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        # Clamped: with N_QUERIES > N_KEYS (cross-attention) the causal bound
        # can exceed the number of K tiles that exist
        iterations = tl.minimum(
            tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE), key_tiles
        )
    else:
        iterations = key_tiles

    for j in range(iterations):
        k_tile = tl.load(k_blk_ptr, boundary_check=(0,), padding_option="zero") # [B_k, D]
        v_tile = tl.load(v_blk_ptr, boundary_check=(0,), padding_option="zero") # [B_k, D]

        # S = Q @ K^T / sqrt(D)
        s_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale # [B_q, B_k]

        # Mask invalid columns
        key_idx = col_offsets + j * K_TILE_SIZE
        mask = key_idx < N_KEYS
        if is_causal:
            mask = mask & (key_idx <= row_offsets)
        s_tile = tl.where(mask, s_tile, -1e6)

        m_new = tl.maximum(m, tl.max(s_tile, axis=1)) # [B_q]

        # P = e ^ (S - M)
        p_tile = tl.exp(s_tile - m_new[:, None]).to(dtype=v_tile.dtype) # [B_q, B_k]

        alpha = tl.exp(m - m_new) # [B_q]
        l = alpha * l + tl.sum(p_tile, axis=1) # [B_q]

        # Rescaling existing output block with the new max
        output = alpha[:, None] * output # [B_q, D]

        # O = P @ V
        output = tl.dot(p_tile, v_tile, acc=output) # [B_q, D]

        # updating the running rowmax
        m = m_new

        # moving to the next tile idx for K and V matrix
        k_blk_ptr = k_blk_ptr.advance((K_TILE_SIZE,0))
        v_blk_ptr = v_blk_ptr.advance((K_TILE_SIZE,0))

    # Dividing the output by rowsum
    output = output / l[:, None] # [B_q, D]

    # storing the logsumexp because exp(s - m) / l = exp(s - m - log(l))
    l = m + tl.log(l) # [B_q]

    # writing the Output tile and and logsumexp to global memory
    tl.store(o_blk_ptr, output.to(dtype=o_blk_ptr.type.element_ty), boundary_check=(0,))
    tl.store(l_blk_ptr, l, boundary_check=(0,))

@triton.jit
def flash_attention_bwd_kern_dK_dV(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr,
    dK_ptr, dV_ptr, dO_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_dob, stride_doq, stride_dod,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    if is_causal:
        first_q_tile = (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE
    else:
        first_q_tile = 0

    q_blk_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(first_q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    k_blk_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    v_blk_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    l_blk_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(first_q_tile * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    d_blk_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(first_q_tile * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dK_blk_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    dV_blk_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    dO_blk_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(first_q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # K and V are loaded once.
    k_tile = tl.load(k_blk_ptr, boundary_check=(0,), padding_option="zero")
    v_tile = tl.load(v_blk_ptr, boundary_check=(0,), padding_option="zero")

    # Private accumulators in registers, fp32 dtype.
    dK_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    col_offsets = tl.arange(0, K_TILE_SIZE)[None, :] + K_TILE_SIZE * key_tile_index
    row_offsets = tl.arange(0, Q_TILE_SIZE)[:, None]

    for i in range(first_q_tile, q_tiles):
        q_tile = tl.load(q_blk_ptr, boundary_check=(0,), padding_option="zero")
        dO_tile = tl.load(dO_blk_ptr, boundary_check=(0,), padding_option="zero")
        l_tile = tl.load(l_blk_ptr, boundary_check=(0,), padding_option="zero")
        d_tile = tl.load(d_blk_ptr, boundary_check=(0,), padding_option="zero")

        # Recompute S from Q and K
        s_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale

        # Mask BEFORE the exp.
        mask = col_offsets < N_KEYS
        if is_causal:
            mask = mask & (col_offsets <= (row_offsets + i * Q_TILE_SIZE))
        s_tile = tl.where(mask, s_tile, -1e6)

        # P = exp(S - L)
        p_tile = tl.exp(s_tile - l_tile[:, None])

        dV_tile = tl.dot(tl.trans(p_tile).to(dO_tile.dtype), dO_tile, acc=dV_tile)

        dP_tile = tl.dot(dO_tile, tl.trans(v_tile))
        dS_tile = p_tile * (dP_tile - d_tile[:, None])

        dK_tile = tl.dot(tl.trans(dS_tile).to(q_tile.dtype), q_tile, acc=dK_tile)

        # Everything indexed by the streaming axis must advance
        q_blk_ptr = q_blk_ptr.advance((Q_TILE_SIZE, 0))
        dO_blk_ptr = dO_blk_ptr.advance((Q_TILE_SIZE, 0))
        l_blk_ptr = l_blk_ptr.advance((Q_TILE_SIZE,))
        d_blk_ptr = d_blk_ptr.advance((Q_TILE_SIZE,))

    dK_tile = dK_tile * scale

    # Written once, after the full reduction over queries.
    tl.store(dK_blk_ptr, dK_tile.to(dK_blk_ptr.type.element_ty), boundary_check=(0,))
    tl.store(dV_blk_ptr, dV_tile.to(dV_blk_ptr.type.element_ty), boundary_check=(0,))

@triton.jit
def flash_attention_bwd_kern_dQ(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr,
    dQ_ptr, dO_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dob, stride_doq, stride_dod,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_blk_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    k_blk_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    v_blk_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    l_blk_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    d_blk_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dO_blk_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_blk_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Q, dO, L and D all belong to this program's query tile, so they are
    # loaded once here and never advanced -- queries are the fixed axis.
    q_tile = tl.load(q_blk_ptr, boundary_check=(0,), padding_option="zero")
    dO_tile = tl.load(dO_blk_ptr, boundary_check=(0,), padding_option="zero")

    l_tile = tl.load(l_blk_ptr, boundary_check=(0,), padding_option="zero")
    d_tile = tl.load(d_blk_ptr, boundary_check=(0,), padding_option="zero")

    # Private accumulator, fp32. Reduced over keys, stored once at the end.
    dQ_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    row_offsets = tl.arange(0, Q_TILE_SIZE)[:, None] + query_tile_index * Q_TILE_SIZE
    col_offsets = tl.arange(0, K_TILE_SIZE)[None, :]

    key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # Causal skip. A query tile can never attend past its own last query, so
    # every K tile beyond that is entirely masked
    # Clamped against key_tiles for N_QUERIES > N_KEYS (cross-attention).
    if is_causal:
        iterations = tl.minimum(
            tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE), key_tiles
        )
    else:
        iterations = key_tiles

    for j in range(iterations):
        k_tile = tl.load(k_blk_ptr, boundary_check=(0,), padding_option="zero")
        v_tile = tl.load(v_blk_ptr, boundary_check=(0,), padding_option="zero")

        s_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale

        key_idx = col_offsets + j * K_TILE_SIZE
        mask = key_idx < N_KEYS
        if is_causal:
            mask = mask & (key_idx <= row_offsets)
        s_tile = tl.where(mask, s_tile, -1e6)

        # P = exp(S - L), exact, no online trick needed.
        p_tile = tl.exp(s_tile - l_tile[:, None])

        dP_tile = tl.dot(dO_tile, tl.trans(v_tile))
        dS_tile = p_tile * (dP_tile - d_tile[:, None])

        dQ_tile = tl.dot(dS_tile.to(k_tile.dtype), k_tile, acc=dQ_tile)

        # Only K and V advance: they are the streaming axis here.
        k_blk_ptr = k_blk_ptr.advance((K_TILE_SIZE, 0))
        v_blk_ptr = v_blk_ptr.advance((K_TILE_SIZE, 0))

    dQ_tile = dQ_tile * scale

    tl.store(dQ_blk_ptr, dQ_tile.to(dQ_blk_ptr.type.element_ty), boundary_check=(0,))


@torch.compile
def flash_attention_fwd_kern_pytorch(Q, K, V):
    D = Q.shape[-1]

    S = einsum(Q, K, "... queries d, ... keys d -> ... queries keys") * (1 / (D ** 0.5))
    L = torch.logsumexp(S, dim=-1)
    P = torch.exp(S - L[..., None])
    O = einsum(P, V, "... queries keys, ... keys d -> ... queries d")

    return O, L

@torch.compile
def flash_attention_bwd_kern_pytorch(Q, K, V, O, dO, L):
    d = Q.shape[-1]

    D = torch.sum(O * dO, dim=-1, keepdim=True)

    S = einsum(Q, K, "... queries d, ... keys d -> ... queries keys") * (1 / (d ** 0.5))
    P = torch.exp(S - L[..., None])

    dV = einsum(P, dO, "... queries keys, ... queries d -> ... keys d")
    dP = einsum(dO, V, "... queries d, ... keys d -> ... queries keys")
    dS = P * (dP - D)
    dQ = einsum(dS, K, "... queries keys, ... keys d -> ... queries d") * (1 / (d ** 0.5))
    dK = einsum(dS, Q, "... queries keys, ... queries d -> ... keys d") * (1 / (d ** 0.5))

    return dQ, dK, dV


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "batch Q D"],
        K: Float[Tensor, "batch K D"],
        V: Float[Tensor, "batch K D"],
        is_causal: bool = False,
    ):
        # assert Q.is_cuda and K.is_cuda and V.is_cuda, "inputs must be on CUDA"
        assert K.shape == V.shape, f"K {tuple(K.shape)} != V {tuple(V.shape)}"
        assert Q.shape[0] == K.shape[0], "batch mismatch"
        assert Q.shape[-1] == K.shape[-1], "head dim mismatch"

        O, L = flash_attention_fwd_kern_pytorch(Q, K, V)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_outputs): # type: ignore
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        dQ, dK, dV = flash_attention_bwd_kern_pytorch(Q, K, V, O, grad_outputs, L)

        return dQ, dK, dV, None


class FlashAttentionTriton(torch.autograd.Function):
    # Tile sizes
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16

    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "batch Q D"],
        K: Float[Tensor, "batch K D"],
        V: Float[Tensor, "batch K D"],
        is_causal: bool = False,
    ):
        batch_size, N_QUERIES, d_model = Q.shape
        N_KEYS = K.shape[1]

        assert Q.is_cuda and K.is_cuda and V.is_cuda, "inputs must be on CUDA"
        assert K.shape == V.shape, f"K {tuple(K.shape)} != V {tuple(V.shape)}"
        assert Q.shape[0] == K.shape[0], "batch mismatch"
        assert Q.shape[-1] == K.shape[-1], "head dim mismatch"

        O = torch.empty_like(Q)
        L = torch.empty((batch_size, N_QUERIES), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = FlashAttentionTriton.Q_TILE_SIZE
        K_TILE_SIZE = FlashAttentionTriton.K_TILE_SIZE

        flash_attention_fwd_kern[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            1 / (d_model ** 0.5),
            d_model, # type: ignore
            Q_TILE_SIZE, # type: ignore
            K_TILE_SIZE, # type: ignore
            is_causal, # type: ignore
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_outputs): # type: ignore

        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        # D is a reduction over the head dim and feeds the softmax-Jacobian
        # subtraction, so it is accumulated in fp32 even when O and dO are
        # bf16 -- summing d terms in bf16 loses far too much here. Matches L,
        # which the forward kernel also stores as fp32.
        D = torch.sum(O.float() * grad_outputs.float(), dim=-1)

        dQ, dK, dV = torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)
        batch_size, N_QUERIES, d_model = Q.shape
        N_KEYS = K.shape[1]

        Q_TILE_SIZE = FlashAttentionTriton.Q_TILE_SIZE
        K_TILE_SIZE = FlashAttentionTriton.K_TILE_SIZE

        flash_attention_bwd_kern_dK_dV[(triton.cdiv(N_KEYS, K_TILE_SIZE), batch_size)](
            Q, K , V,
            L, D,
            dK, dV, grad_outputs,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            grad_outputs.stride(0), grad_outputs.stride(1), grad_outputs.stride(2),
            N_QUERIES, N_KEYS,
            1 / (d_model ** 0.5),
            d_model,
            Q_TILE_SIZE, # type: ignore
            K_TILE_SIZE, # type: ignore
            is_causal,
        )

        flash_attention_bwd_kern_dQ[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)](
            Q, K , V,
            L, D,
            dQ, grad_outputs,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            grad_outputs.stride(0), grad_outputs.stride(1), grad_outputs.stride(2),
            N_QUERIES, N_KEYS,
            1 / (d_model ** 0.5),
            d_model,
            Q_TILE_SIZE, # type: ignore
            K_TILE_SIZE, # type: ignore
            is_causal,
        )

        return dQ, dK, dV, None



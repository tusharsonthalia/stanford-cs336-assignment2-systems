import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.cuda import nvtx

class Linear(nn.Module):
    """A bias-free linear transformation, y = x @ W.T.

    Follows the interface of torch.nn.Linear, without the bias term.

    Args:
        in_features (int): Size of the last dimension of the input.
        out_features (int): Size of the last dimension of the output.
        device (torch.device | None): Device to store the parameter on.
        dtype (torch.dtype | None): Data type of the parameter.

    Attributes:
        weight (nn.Parameter): Shape (out_features, in_features) -- i.e. W, not
            W-transpose. Each row holds the weights of a single output unit, so a
            row is contiguous in row-major memory.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # (d_out, d_in): output units first, so W[j] is one contiguous row
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # truncated normal, sigma^2 = 2 / (d_in + d_out), cut at +/- 3 sigma
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(
        self,
        x: Float[Tensor, "... d_in"]
    ) -> Float[Tensor, "... d_out"]:
        """Apply the transformation. Accepts any number of leading batch dims."""
        # contract over d_in; d_out survives -> equivalent to x @ W.T
        return einsum(x, self.weight,"... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    """
    Token embedding table; maps integer token IDs to dense vectors.

    Follows the interface of torch.nn.Embedding. The forward pass is a row
    gather.

    Args:
        num_embeddings (int): Vocabulary size.
        embedding_dim (int): Dimension of each embedding vector (d_model).
        device (torch.device | None): Device to store the parameter on.
        dtype (torch.dtype | None): Data type of the parameter.

    Attributes:
        weight (nn.Parameter): Shape (num_embeddings, embedding_dim). Vocab first
            so that each embedding vector is a contiguous row.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # (vocab, d_model): rows are gathered, so d_model must be the last axis
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # unit-variance truncated normal, cut at +/- 3
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(
        self,
        token_ids: Int[Tensor, "..."]
    ) -> Float[Tensor, "... d_model"]:
        """Look up embeddings for a batch of token IDs.

        Args:
            token_ids: Integer tensor of any shape, values in [0, num_embeddings).

        Returns:
            Tensor of shape token_ids.shape + (embedding_dim,).
        """
        # advanced indexing: gathers along dim 0 and substitutes token_ids' shape
        # for the vocab axis. Backward is a scatter-add, so repeated IDs accumulate.
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """Root Mean Square layer normalization:  RMSNorm(a_i) = a_i / RMS(a) * g_i,
    where RMS(a) = sqrt(mean(a^2) + eps).

    Normalizes over the LAST axis only; every leading dimension is treated as a
    batch dimension. Unlike LayerNorm there is no mean subtraction and no bias.

    Args:
        d_model (int): Size of the normalized (last) dimension.
        eps (float): Added inside the radicand for numerical stability.
        device (torch.device | None): Device to store the parameter on.
        dtype (torch.dtype | None): Data type of the parameter.

    Attributes:
        weight (nn.Parameter): Learned per-feature gain of shape (d_model,),
            initialized to ones so the layer starts as a pure normalizer.
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(
        self,
        x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        """Normalize x over its last dimension, returning the original dtype."""
        # squaring in fp16/bf16 overflows, so normalize in fp32 and cast back
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_sq = reduce(x ** 2,"... d_model -> ... 1", reduction="mean")
        inv_rms = torch.rsqrt(mean_sq + self.eps)
        result = x * inv_rms * self.weight

        return result.to(in_dtype)

def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """SiLU / Swish activation: x * sigmoid(x).

    Uses torch.sigmoid rather than the algebraically equivalent x / (1 + exp(-x)),
    which overflows to inf for x < -88 in fp32 (x < -11 in fp16) and produces a
    NaN gradient there.
    """
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    """SwiGLU position-wise feed-forward network:

        FFN(x) = W2 @ (SiLU(W1 @ x) * W3 @ x)

    A Gated Linear Unit: w1 and w3 are two independent projections of the same
    input, multiplied elementwise, so the layer computes a product of linear
    readouts rather than a pointwise function of one. w3 acts as a data-dependent
    gate on the w1 branch.

    Operates position-wise -- any number of leading batch dims pass through
    untouched. No bias terms.

    Args:
        d_model (int): Input and output dimension.
        d_ff (int): Inner dimension. Supplied by the caller. The canonical
            choice is d_ff ~= (8/3) * d_model rounded to a multiple of 64, which
            keeps the three matrices at the same parameter count as the original
            two-matrix FFN's 8 * d_model^2.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        w1 (Linear): (d_ff, d_model) up-projection; its output receives the SiLU.
        w2 (Linear): (d_model, d_ff) down-projection.
        w3 (Linear): (d_ff, d_model) up-projection; the gate. Shape-identical to
            w1.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        """Apply the gated feed-forward transform. Shape is preserved."""
        # w1 branch is activated, w3 branch is the gate
        gated = silu(self.w1(x)) * self.w3(x)
        activated = self.w2(gated)

        return activated


class RoPE(nn.Module):
    """Rotary Position Embedding.

    Splits the feature axis into d_k/2 ADJACENT pairs and rotates each pair as a
    2D vector by an angle proportional to the token's position:

        theta(i, k) = i / Theta^(2k / d_k)      k = 0 .. d_k/2 - 1

    Because a rotation's transpose is its inverse and rotations compose additively,
    (R_i q) . (R_j k) = q^T R_(j-i) k -- so attention LOGITS depend only on the
    relative offset j-i, never on absolute positions.

    Pairing is interleaved -- (x0,x1), (x2,x3), ...
    Has no learnable parameters.

    Args:
        theta (float): Base Theta of the frequency ladder.
        d_k (int): PER-HEAD dimension (d_model // num_heads), not d_model. Must be
            even.
        max_seq_len (int): Number of position rows to precompute.
        device (torch.device | None): Device to build the tables on.
        dtype (torch.dtype | None): Stored dtype of the tables.

    Attributes:
        sin_table, cos_table (Tensor): Shape (max_seq_len, d_k/2), registered with
            persistent=False -- they are fully derived from the constructor args.
    """
    sin_table: Float[Tensor, "max_seq_len d_k/2"]
    cos_table: Float[Tensor, "max_seq_len d_k/2"]

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert d_k % 2 == 0, f"d_k should be even. It is currently {d_k}"
        if dtype is None:
            dtype = torch.get_default_dtype()

        # angles reach thousands of radians at long context, where a float32 ulp
        # is ~1e-4; compute in float64 and cast only the bounded sin/cos results
        seq_positions = torch.arange(0, max_seq_len, dtype=torch.float64)

        # pair_idx = 0, 2, 4, ...
        pair_idx = torch.arange(0, d_k, 2, dtype=torch.float64)
        inv_freq = theta ** (-pair_idx / d_k)

        # outer product -> (max_seq_len, d_k/2)
        angles = seq_positions.reshape(-1, 1) * inv_freq.reshape(1, -1)
        sin_angles = torch.sin(angles).to(dtype=dtype, device=device)
        cos_angles = torch.cos(angles).to(dtype=dtype, device=device)

        self.register_buffer("sin_table", sin_angles, persistent=False)
        self.register_buffer("cos_table", cos_angles, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        """Rotate the pairs of x according to each token's position.

        Args:
            x: Shape (..., seq_len, d_k). Any number of leading batch/head dims.
            token_positions: Shape (..., seq_len), integer. The ABSOLUTE position
                of each token.

        Returns:
            Tensor of the same shape and dtype as x.
        """
        # gather one table row per token: (..., seq_len, d_k/2)
        sin = self.sin_table[token_positions]
        cos = self.cos_table[token_positions]

        # (..., seq_len, d_k/2)
        x_even = x[..., ::2]    # first element of each pair
        x_odd  = x[..., 1::2]   # second element of each pair

        # [[cos, -sin], [sin, cos]] rotation matrix applied to the pair
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        # interleave back to [e0, o0, e1, o1, ...].
        # Shape: (..., seq, d_k)
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

def softmax(x: Float[Tensor, "..."], dimension: int) -> Float[Tensor, "..."]:
    """Numerically stable softmax along `dimension`.

    Subtracts the max before exponentiating. softmax is invariant to adding a
    constant to all inputs, so this changes nothing mathematically while keeping
    the largest exponent at exp(0) = 1 -- without it, exp overflows to inf for
    inputs above ~88 in fp32 and inf/inf gives NaN.

    Args:
        x: Tensor of any shape.
        dimension (int): Axis to normalize over.

    Returns:
        Tensor of the same shape; `dimension` sums to 1.
    """
    z = x - torch.amax(x, dim=dimension, keepdim=True)

    exp_z = torch.exp(z)
    sum_exp = torch.sum(exp_z, dim=dimension, keepdim=True)

    return exp_z / sum_exp

def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    *,
    mask: Bool[Tensor, "queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:
    """Scaled dot-product attention:

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Maps n queries to n outputs by pooling over m keys.

    Args:
        Q: Shape (..., queries, d_k).
        K: Shape (..., keys, d_k).
        V: Shape (..., keys, d_v).
        mask: Optional boolean mask; False positions are set to -inf so
            they receive zero probability after the softmax.

    Returns:
        Tensor of shape (..., queries, d_v).
    """
    d_k = Q.shape[-1]
    scale = 1 / (d_k ** 0.5)
    qk_proj = einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
    ) * scale

    if mask is not None:
        qk_proj = torch.where(mask, qk_proj, -torch.inf)

    # shape: (..., num_heads, queries, keys)
    probabilities = softmax(qk_proj, dimension=-1)

    attention = einsum(
        probabilities, V,
        "... queries keys, ... keys d_v -> ... queries d_v"
    )

    return attention

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    *,
    mask: Bool[Tensor, "queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:
    """Scaled dot-product attention:

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Maps n queries to n outputs by pooling over m keys.

    Args:
        Q: Shape (..., queries, d_k).
        K: Shape (..., keys, d_k).
        V: Shape (..., keys, d_v).
        mask: Optional boolean mask; False positions are set to -inf so
            they receive zero probability after the softmax.

    Returns:
        Tensor of shape (..., queries, d_v).
    """
    d_k = Q.shape[-1]
    scale = 1 / (d_k ** 0.5)

    with nvtx.range("computing attention scores"):
        qk_proj = einsum(
            Q, K, "... queries d_k, ... keys d_k -> ... queries keys"
        ) * scale

        if mask is not None:
            qk_proj = torch.where(mask, qk_proj, -torch.inf)

    with nvtx.range("computing softmax"):
        # shape: (..., num_heads, queries, keys)
        probabilities = softmax(qk_proj, dimension=-1)

    with nvtx.range("computing output result"):
        attention = einsum(
            probabilities, V,
            "... queries keys, ... keys d_v -> ... queries d_v"
        )

    return attention

class CausalMaskedMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention without rotary position embeddings.

    MultiHeadSelfAttention(x) = W_O @ MultiHead(W_Q @ x, W_K @ x, W_V @ x)

    Multi-head attention is FLOP and parameter-neutral versus single-head: each
    head does 2 n^2 d_k work and there are h of them, so the total is 2 n^2 d_model
    and W_Q is (h * d_k, d_model) = (d_model, d_model). It allows
    h independent attention distributions per query instead of one, so the layer can
    attend to several things at once rather than averaging them into a single
    blurred retrieval.

    Args:
        d_model (int): Model dimension. Must be divisible by num_heads.
        num_heads (int): Number of attention heads. Each gets d_k = d_v = d_model/h.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        q_proj, k_proj, v_proj (Linear): (d_model, d_model). One matmul each covers
            ALL heads -- the output's last axis is the heads' d_k blocks concatenated.
        output_proj (Linear): (d_model, d_model). Mixes the concatenated head outputs
            back into the residual stream. Named output_proj, not o_proj, to match the
            reference state dict key layers.{i}.attn.output_proj.weight.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = self.d_model // self.num_heads

        self.q_proj = Linear(self.d_model, self.head_dim * self.num_heads, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.head_dim * self.num_heads, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.head_dim * self.num_heads, device=device, dtype=dtype)
        self.output_proj = Linear(self.head_dim * self.num_heads, self.d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply causal multi-head self-attention.

        Args:
            x: Shape (..., seq_len, d_model).

        Returns:
            Tensor of the same shape as x.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshaping the QKV projections into (..., num_heads, seq_len, head_dim) shape for independent Attention
        q = rearrange(q, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)

        # slicing a causal mask for the appropriate seq len
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool))

        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # merging head back into (..., seq_len, d_model)
        out = rearrange(out, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")

        return self.output_proj(out)

class CausalMaskedMultiHeadSelfAttentionWithRoPE(nn.Module):
    """Causal multi-head self-attention with rotary position embeddings.

    MultiHeadSelfAttention(x) = W_O @ MultiHead(W_Q @ x, W_K @ x, W_V @ x)

    Multi-head attention is FLOP and parameter-neutral versus single-head: each
    head does 2 n^2 d_k work and there are h of them, so the total is 2 n^2 d_model
    and W_Q is (h * d_k, d_model) = (d_model, d_model). It allows
    h independent attention distributions per query instead of one, so the layer can
    attend to several things at once rather than averaging them into a single
    blurred retrieval.

    Args:
        d_model (int): Model dimension. Must be divisible by num_heads.
        num_heads (int): Number of attention heads. Each gets d_k = d_v = d_model/h.
        theta (float): RoPE base Theta.
        max_seq_len (int): Longest sequence RoPE must support.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.

    Attributes:
        causal_mask (Tensor): (max_seq_len, max_seq_len) lower-triangular bool.
            True means "attend". Built once and sliced per call, since it depends
            only on max_seq_len.
        q_proj, k_proj, v_proj (Linear): (d_model, d_model). One matmul each covers
            ALL heads -- the output's last axis is the heads' d_k blocks concatenated.
        output_proj (Linear): (d_model, d_model). Mixes the concatenated head outputs
            back into the residual stream. Named output_proj, not o_proj, to match the
            reference state dict key layers.{i}.attn.output_proj.weight.
        rope (RoPE): Shared by q and k. Has no parameters, so one instance suffices.
    """
    causal_mask: Bool[Tensor, "max_seq_len max_seq_len"]

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = self.d_model // self.num_heads

        # causal mask for the attention operation where True means allowed
        causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device=device, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        
        self.q_proj = Linear(self.d_model, self.head_dim * self.num_heads, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.head_dim * self.num_heads, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.head_dim * self.num_heads, device=device, dtype=dtype)
        self.output_proj = Linear(self.head_dim * self.num_heads, self.d_model, device=device, dtype=dtype)

        # RoPE operates on per head dimensions
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Apply causal multi-head self-attention.

        Args:
            x: Shape (..., seq_len, d_model). seq_len must be <= max_seq_len.
            token_positions: Optional, Default: arange(seq_len).
                Shape (..., seq_len), integer absolute positions for RoPE.

        Returns:
            Tensor of the same shape as x.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshaping the QKV projections into (..., num_heads, seq_len, d_k) shape for independent RoPE and Attention
        q = rearrange(q, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)

        # slicing a causal mask for the appropriate seq len
        seq_len = x.shape[-2]
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        # synthesizing token positions if missing
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(-2)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # merging head back into (..., seq_len, d_model)
        out = rearrange(out, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")

        return self.output_proj(out)

class TransformerBlock(nn.Module):
    """A pre-norm Transformer block.

    Two sub-layers:

        x = x + MultiHeadSelfAttention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    PRE-norm normalizes the sub-layer's INPUT and adds the result to the residual. 
    The original post-norm form, RMSNorm(x + sublayer(x)), puts a normalization 
    directly on the residual path, so a gradient travelling from the last layer 
    to the first crosses num_layers norm Jacobians instead of num_layers clean 
    additions.

    Shape is preserved: (..., seq_len, d_model) in and out.

    Args:
        d_model (int): Residual stream width.
        num_heads (int): Attention heads. d_model must be divisible by it.
        d_ff (int): Inner width of the feed-forward network.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.
        eps (float): RMSNorm epsilon.
        theta (float): RoPE base Theta.
        max_seq_len (int): Longest sequence RoPE and the causal mask must support.

    Attributes:
        ln1, ln2 (RMSNorm): Norms for the attention and feed-forward sub-layers.
        attn (CausalMaskedMultiHeadSelfAttentionWithRoPE): Multi Head Self Attention
            block with causal mask over non attending tokens.
        ffn (SwiGLU): Position-wise feed-forward network.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        eps: float = 1e-5,
        theta: float,
        max_seq_len: int,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.attn = CausalMaskedMultiHeadSelfAttentionWithRoPE(
            d_model, num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype
        )

        self.ln2 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
    ) -> Float[Tensor, "... seq_len d_model"]:
        """Run both pre-norm sub-layers. Shape is preserved."""
        # normalize -> transform -> ADD to the residual.
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

        token_embeddings -> num_layers x TransformerBlock -> ln_final -> lm_head

    Takes integer token IDs of shape (batch, seq_len) and returns next-token
    LOGITS of shape (batch, seq_len, vocab_size).

    Args:
        d_model (int): Residual stream width.
        num_heads (int): Attention heads per block.
        d_ff (int): Inner width of each feed-forward network.
        theta (float): RoPE base Theta.
        vocab_size (int): Number of tokens; sizes both the embedding and the LM head.
        context_length (int): Maximum sequence length. Sizes the RoPE tables and
            causal masks inside every block.
        num_layers (int): Number of Transformer blocks.
        device (torch.device | None): Device to store the parameters on.
        dtype (torch.dtype | None): Data type of the parameters.
        eps (float): RMSNorm epsilon, shared by every norm in the model.

    Attributes:
        token_embeddings (Embedding): (vocab_size, d_model).
        layers (nn.ModuleList): The Transformer blocks.
        ln_final (RMSNorm): Final normalization before the LM head.
        lm_head (Linear): (vocab_size, d_model). Same shape as token_embeddings.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embeddings = Embedding(self.vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, device=device, dtype=dtype,
                    eps=eps, theta=theta, max_seq_len=self.context_length,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, self.vocab_size, device=device, dtype=dtype)

    def forward(
            self,
            in_indices: Int[Tensor, "batch_size sequence_length"]
        ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        """Map token IDs to next-token logits.

        Args:
            in_indices: Integer token IDs, values in [0, vocab_size).
                sequence_length must be <= context_length.

        Returns:
            Unnormalized logits of shape (batch, sequence_length, vocab_size).
        """
        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)

        # pre-norm blocks do not normalize their own output, so the accumulated
        # residual stream needs one final normalization before the LM head
        x = self.ln_final(x)

        # logits
        return self.lm_head(x)
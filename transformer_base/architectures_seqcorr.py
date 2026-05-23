# SOFTMAX ATTENTION, UNROLLED setting
# finite shared isotropic tasks

import numpy as np
import argparse
import csv
import json
import fcntl
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.nn import softmax, gelu
import optax

print("imports done", flush=True)

def _psd_factor(M, jitter=1e-10):
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {M.shape}")

    M = 0.5 * (M + M.T)
    eye = np.eye(M.shape[0], dtype=M.dtype)
    try:
        return np.linalg.cholesky(M + jitter * eye)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(M)
        w = np.clip(w, 0.0, None)
        return V @ (np.sqrt(w)[:, None] * V.T)

def sample_matrix_normal(L, d, Sigma=None, K=None, rng=None, jitter=1e-10, sqrtSigma=None, sqrt_K=None):
    """
    Sample X with shape (L, d) and covariance
    E[X_{ai} X_{bj}] = K_{ab} * Sigma_{ij}.
    """
    if sqrt_K is None:
        if K is None:
            raise ValueError("sample_matrix_normal requires K or sqrt_K")
        sqrt_K = _psd_factor(K, jitter=jitter)
    if sqrtSigma is None:
        if Sigma is None:
            raise ValueError("sample_matrix_normal requires Sigma or sqrtSigma")
        sqrtSigma = _psd_factor(Sigma, jitter=jitter)

    if rng is None:
        Z = np.random.randn(L, d)
    else:
        Z = rng.standard_normal(size=(L, d))
    return sqrt_K @ Z @ sqrtSigma.T

def sample_matrix_normal_batch(n, L, d, Sigma=None, K=None, rng=None, jitter=1e-10, sqrtSigma=None, sqrt_K=None):
    """
    Sample X with shape (n, L, d) where each X[i] is matrix normal with
    E[X_{ai} X_{bj}] = K_{ab} * Sigma_{ij}.
    """
    if sqrt_K is None:
        if K is None:
            raise ValueError("sample_matrix_normal_batch requires K or sqrt_K")
        sqrt_K = _psd_factor(K, jitter=jitter)
    if sqrtSigma is None:
        if Sigma is None:
            raise ValueError("sample_matrix_normal_batch requires Sigma or sqrtSigma")
        sqrtSigma = _psd_factor(Sigma, jitter=jitter)

    if rng is None:
        Z = np.random.randn(n, L, d)
    else:
        Z = rng.standard_normal(size=(n, L, d))
    return np.einsum("ab,nbd,de->nae", sqrt_K, Z, sqrtSigma.T)

def kernel_exp(P_tr, corr_len):
    if corr_len < 0.0:
        raise ValueError("corr_len must be nonnegative")
    idx = jnp.arange(P_tr, dtype=jnp.float32)
    K = jnp.where(
        corr_len == 0.0,
        jnp.eye(P_tr, dtype=jnp.float32),
        jnp.exp(-jnp.abs(idx[:, None] - idx[None, :]) / corr_len).astype(jnp.float32),
    )
    return K

def sample_ar1_batch(n, length, d, corr_len, rng=None):
    """
    Sample a batch of AR(1) sequences with stationary marginal N(0, I_d / d).
    For the exponential kernel K_ij = exp(-|i-j| / corr_len), the exact AR(1)
    coefficient is phi = exp(-1 / corr_len).
    """
    if corr_len < 0.0:
        raise ValueError("corr_len must be nonnegative")
    if length < 0:
        raise ValueError("length must be nonnegative")
    if length == 0:
        return np.empty((n, 0, d))

    randn = np.random.randn if rng is None else rng.standard_normal
    X = np.empty((n, length, d))
    X[:, 0, :] = randn(n, d)

    if length > 1:
        if corr_len == 0.0:
            X[:, 1:, :] = randn(n, length - 1, d)
        else:
            phi = float(np.exp(-1.0 / corr_len))
            noise_scale = np.sqrt(1.0 - phi ** 2)
            eps = randn(n, length - 1, d)
            for t in range(1, length):
                X[:, t, :] = phi * X[:, t - 1, :] + noise_scale * eps[:, t - 1, :]

    return X / np.sqrt(d)

def init_params_transformer(
    d,
    N,
    L,
    sigma=0.4,
    mlp_ratio=1,
    key=None,
    use_attention=True,
    use_mlp=True,
):

    if N <= d:
        raise ValueError("init_params_transformer expects N > d so there is a dedicated label channel")

    if key is None:
        key = jax.random.PRNGKey(0)
    noise_scale = 1.0e-3

    W_x_embed = jnp.sqrt(2.0) * jnp.sqrt(N) * sigma * jnp.eye(N, d)  # (N,d)
    # Reserve the first extra channel as a dedicated label axis.
    label_idx = d if N > d else 0
    W_y_embed = jnp.zeros(N)                                          # (N,)
    W_y_embed = W_y_embed.at[label_idx].set(1.0)
    W_out = jnp.zeros(N)                                              # (N,)
    W_out = W_out.at[label_idx].set(1.0)

    x_mask = jnp.eye(N).at[label_idx, label_idx].set(0.0)
    label_vec = jnp.zeros(N).at[label_idx].set(1.0)
    label_proj = jnp.outer(label_vec, label_vec)

    key, q_key, k_key, v_key, o_key, mlp1_key, mlp2_key = jax.random.split(key, 7)

    # Make attention scores depend on x channels only, and move information
    # primarily through the dedicated label channel.
    Wq_base = sigma * jnp.sqrt(N) * x_mask
    Wq_base = Wq_base + noise_scale * jax.random.normal(q_key, Wq_base.shape)
    Wk_base = sigma * jnp.sqrt(N) * x_mask
    Wk_base = Wk_base + noise_scale * jax.random.normal(k_key, Wk_base.shape)
    Wv_base = sigma * jnp.sqrt(N) * label_proj
    Wv_base = Wv_base + noise_scale * jax.random.normal(v_key, Wv_base.shape)
    Wo_base = 1.0 * label_proj
    Wo_base = Wo_base + noise_scale * jax.random.normal(o_key, Wo_base.shape)

    # Keep the MLP initially close to off so it does not immediately become
    # a memorization shortcut, while still allowing gradients to shape it.
    mlp_scale = 1.0e-3
    if use_mlp:
        W_mlp1_base = mlp_scale * jnp.concatenate(
            [jnp.eye(N)] + [jnp.zeros((N, N)) for _ in range(mlp_ratio - 1)],
            axis=0,
        )
        W_mlp1_base = W_mlp1_base + noise_scale * jax.random.normal(mlp1_key, W_mlp1_base.shape)
        W_mlp2_base = mlp_scale * jnp.concatenate(
            [jnp.eye(N)] + [jnp.zeros((N, N)) for _ in range(mlp_ratio - 1)],
            axis=1,
        )
        W_mlp2_base = W_mlp2_base + noise_scale * jax.random.normal(mlp2_key, W_mlp2_base.shape)
    if use_attention and use_mlp:
        params_tr_layers = [
            (Wq_base, Wk_base, Wv_base, Wo_base, W_mlp1_base, W_mlp2_base)
            for _ in range(L)
        ]
    elif use_attention:
        params_tr_layers = [
            (Wq_base, Wk_base, Wv_base, Wo_base)
            for _ in range(L)
        ]
    elif use_mlp:
        params_tr_layers = [
            (W_mlp1_base, W_mlp2_base)
            for _ in range(L)
        ]
    else:
        raise ValueError("init_params_transformer requires at least one of use_attention or use_mlp to be True")
    return params_tr_layers, W_x_embed, W_y_embed, W_out

def _zero_linear_attention_value_row(Wv):
    return Wv.at[-1, :-1].set(0.0)

def constrain_linear_attention_layers(
    params_tr_layers,
    *,
    use_mlp=False,
    tied_attention=False,
    zero_value_row=False,
):
    if not zero_value_row:
        return params_tr_layers

    constrained_layers = []
    for layer in params_tr_layers:
        if tied_attention:
            if use_mlp:
                Wqk, Wv, W_mlp1, W_mlp2 = layer
                constrained_layers.append((Wqk, _zero_linear_attention_value_row(Wv), W_mlp1, W_mlp2))
            else:
                Wqk, Wv = layer
                constrained_layers.append((Wqk, _zero_linear_attention_value_row(Wv)))
        else:
            if use_mlp:
                Wq, Wk, Wv, W_mlp1, W_mlp2 = layer
                constrained_layers.append((Wq, Wk, _zero_linear_attention_value_row(Wv), W_mlp1, W_mlp2))
            else:
                Wq, Wk, Wv = layer
                constrained_layers.append((Wq, Wk, _zero_linear_attention_value_row(Wv)))
    return constrained_layers

def init_params_linear_attention_full(
    d,
    L,
    key=None,
    mlp_ratio=1,
    use_mlp=False,
    tied_attention=False,
    zero_value_row=False,
):
    D = d + 1

    if key is None:
        key = jax.random.PRNGKey(0)

    noise_scale = 1.0e-2
    eye = jnp.eye(D)
    if tied_attention:
        num_keys_per_layer = 4 if use_mlp else 2
    else:
        num_keys_per_layer = 5 if use_mlp else 3
    layer_keys = jax.random.split(key, num_keys_per_layer * L)
    mlp_scale = 1.0e-3

    params_tr_layers = []
    for l in range(L):
        offset = num_keys_per_layer * l
        if tied_attention:
            qk_key, v_key = layer_keys[offset: offset + 2]
            Wqk = eye + noise_scale * jax.random.normal(qk_key, (D, D))
            Wv = eye + noise_scale * jax.random.normal(v_key, (D, D))
        else:
            q_key, k_key, v_key = layer_keys[offset: offset + 3]
            Wq = eye + noise_scale * jax.random.normal(q_key, (D, D))
            Wk = eye + noise_scale * jax.random.normal(k_key, (D, D))
            Wv = eye + noise_scale * jax.random.normal(v_key, (D, D))
        if zero_value_row:
            Wv = _zero_linear_attention_value_row(Wv)
        if use_mlp:
            mlp_start = offset + 2 if tied_attention else offset + 3
            mlp1_key, mlp2_key = layer_keys[mlp_start: mlp_start + 2]
            W_mlp1 = mlp_scale * jnp.concatenate(
                [jnp.eye(D)] + [jnp.zeros((D, D)) for _ in range(mlp_ratio - 1)],
                axis=0,
            )
            W_mlp1 = W_mlp1 + noise_scale * jax.random.normal(mlp1_key, W_mlp1.shape)
            W_mlp2 = mlp_scale * jnp.concatenate(
                [jnp.eye(D)] + [jnp.zeros((D, D)) for _ in range(mlp_ratio - 1)],
                axis=1,
            )
            W_mlp2 = W_mlp2 + noise_scale * jax.random.normal(mlp2_key, W_mlp2.shape)
            if tied_attention:
                params_tr_layers.append((Wqk, Wv, W_mlp1, W_mlp2))
            else:
                params_tr_layers.append((Wq, Wk, Wv, W_mlp1, W_mlp2))
        else:
            if tied_attention:
                params_tr_layers.append((Wqk, Wv))
            else:
                params_tr_layers.append((Wq, Wk, Wv))

    return params_tr_layers

def init_params_hybrid_attention_full(d, order, key=None):
    D = d + 1

    if key is None:
        key = jax.random.PRNGKey(0)

    noise_scale = 1.0e-2
    eye = jnp.eye(D)
    keys = jax.random.split(key, sum(4 if kind == "softmax" else 3 for kind in order))

    params_tr_layers = []
    offset = 0
    for kind in order:
        if kind == "linear":
            q_key, k_key, v_key = keys[offset: offset + 3]
            offset += 3
            Wq = eye + noise_scale * jax.random.normal(q_key, (D, D))
            Wk = eye + noise_scale * jax.random.normal(k_key, (D, D))
            Wv = eye + noise_scale * jax.random.normal(v_key, (D, D))
            params_tr_layers.append((Wq, Wk, Wv))
        elif kind == "softmax":
            q_key, k_key, v_key, o_key = keys[offset: offset + 4]
            offset += 4
            Wq = eye + noise_scale * jax.random.normal(q_key, (D, D))
            Wk = eye + noise_scale * jax.random.normal(k_key, (D, D))
            Wv = eye + noise_scale * jax.random.normal(v_key, (D, D))
            Wo = eye + noise_scale * jax.random.normal(o_key, (D, D))
            params_tr_layers.append((Wq, Wk, Wv, Wo))
        else:
            raise ValueError(f"Unknown hybrid layer kind: {kind}")

    return params_tr_layers

def layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)

def softmax_transformer(
    params_tr_layers,  # list of attn tuples, mlp tuples, or both
    W_x_embed,         # (N,d)
    W_y_embed,         # (N,)
    W_out,             # (N,)
    X,                 # (B,P,d)
    y,                 # (B,P)
    *,
    n_heads=1,
    P_test=1,
    beta=1.0,
    attn_temperature=1.0,
    use_attention=True,
    use_layernorm=True,
    use_mlp=True,
):
    L = len(params_tr_layers)
    N, d = W_x_embed.shape
    if N % n_heads != 0:
        raise ValueError("N must be divisible by n_heads")
    Dh = N // n_heads

    B = y.shape[0]
    P = X.shape[1]
    P_tr = P - P_test
    if attn_temperature <= 0.0:
        raise ValueError("attn_temperature must be positive")

    mask_y = jnp.ones_like(y)
    mask_y = mask_y.at[:, P_tr:].set(0.0)
    y_embed = jnp.einsum('bp,n->bpn', y * mask_y, W_y_embed)          # (B,P,N)
    x_embed = jnp.einsum('bpd,nd->bpn', X, W_x_embed)                 # (B,P,N)
    h = x_embed + y_embed

    if not (use_attention or use_mlp):
        raise ValueError("softmax_transformer requires at least one of attention or MLP")

    causal_mask = jnp.tril(jnp.ones((P, P), dtype=h.dtype))
    neg_inf = jnp.array(-1e9, dtype=h.dtype)
    attn_mask = (1.0 - causal_mask)[None, None, :, :] * neg_inf      # (1,1,P,P)

    def split_heads(t):
        return t.reshape(B, P, n_heads, Dh).transpose(0, 2, 1, 3)    # (B,H,P,Dh)

    def merge_heads(t):
        return t.transpose(0, 2, 1, 3).reshape(B, P, N)              # (B,P,N)

    for l in range(L):
        if use_attention and use_mlp:
            Wq, Wk, Wv, Wo, W_mlp1, W_mlp2 = params_tr_layers[l]
        elif use_attention:
            Wq, Wk, Wv, Wo = params_tr_layers[l]
        else:
            W_mlp1, W_mlp2 = params_tr_layers[l]

        if use_attention:
            h_attn_in = layer_norm(h) if use_layernorm else h
            q = jnp.einsum('bpn,ln->bpl', h_attn_in, Wq) / jnp.sqrt(N)
            k = jnp.einsum('bpn,ln->bpl', h_attn_in, Wk) / jnp.sqrt(N)
            v = jnp.einsum('bpn,ln->bpl', h_attn_in, Wv) / jnp.sqrt(N)

            qh, kh, vh = map(split_heads, (q, k, v))
            att_logits = jnp.einsum('bhid,bhjd->bhij', qh, kh) / jnp.sqrt(Dh)
            att_logits = att_logits + attn_mask
            A_attn = softmax(att_logits / attn_temperature, axis=-1)

            ctx_h = jnp.einsum('bhij,bhjd->bhid', A_attn, vh)
            ctx = merge_heads(ctx_h)
            attn_out = jnp.einsum('bpn,ln->bpl', ctx, Wo)
            h = h + (beta / L) * attn_out

        if use_mlp:
            h_mlp_in = layer_norm(h) if use_layernorm else h
            h_hidden = gelu(jnp.einsum('bpn,hn->bph', h_mlp_in, W_mlp1))
            mlp_out = jnp.einsum('bph,nh->bpn', h_hidden, W_mlp2) / jnp.sqrt(W_mlp1.shape[0])
            h = h + (beta / L) * mlp_out

    h_out = layer_norm(h) if use_layernorm else h
    out = jnp.einsum('bpn,n->bp', h_out, W_out) / jnp.sqrt(N)
    return out, [], []

def pure_linear_attention(
    params_tr_layers,  # list of (Wq, Wk, Wv), (Wqk, Wv), or MLP-augmented variants
    X,                 # (B,P,d)
    y,                 # (B,P)
    *,
    P_test=1,
    beta=1.0,
    use_mlp=False,
    tied_attention=False,
):
    L = len(params_tr_layers)
    d = X.shape[-1]
    D = d + 1

    B = y.shape[0]
    P = X.shape[1]
    P_tr = P - P_test

    mask_y = jnp.ones_like(y)
    mask_y = mask_y.at[:, P_tr:].set(0.0)
    y_channel = (y * mask_y)[..., None]
    h = jnp.concatenate([X, y_channel], axis=-1)  # (B,P,d+1)

    key_mask = jnp.ones((P, P), dtype=h.dtype)
    key_mask = key_mask.at[:, P_tr:].set(0.0)

    for l in range(L):
        if tied_attention:
            if use_mlp:
                Wqk, Wv, W_mlp1, W_mlp2 = params_tr_layers[l]
            else:
                Wqk, Wv = params_tr_layers[l]
        else:
            if use_mlp:
                Wq, Wk, Wv, W_mlp1, W_mlp2 = params_tr_layers[l]
            else:
                Wq, Wk, Wv = params_tr_layers[l]
        v = jnp.einsum('bpd,ed->bpe', h, Wv)

        if tied_attention:
            att = jnp.einsum('bpd,de,bqe->bpq', h, Wqk, h) / P_tr
        else:
            q = jnp.einsum('bpd,ed->bpe', h, Wq) / jnp.sqrt(P_tr)
            k = jnp.einsum('bpd,ed->bpe', h, Wk) / jnp.sqrt(P_tr)
            att = jnp.einsum('bid,bjd->bij', q, k)
        att = att * key_mask[None, :, :]
        ctx = jnp.einsum('bij,bjd->bid', att, v)
        h = h + (beta / L) * ctx

        if use_mlp:
            h_hidden = gelu(jnp.einsum('bpd,hd->bph', h, W_mlp1))
            mlp_out = jnp.einsum('bph,dh->bpd', h_hidden, W_mlp2) / jnp.sqrt(W_mlp1.shape[0])
            h = h + (beta / L) * mlp_out

    out = h[:, :, -1]
    return out, [], []

def hybrid_attention_stack(
    params_tr_layers,  # list of linear tuples or softmax tuples
    layer_order,       # e.g. ["linear", "softmax"]
    X,                 # (B,P,d)
    y,                 # (B,P)
    *,
    P_test=1,
    beta=1.0,
    attn_temperature=1.0,
):
    d = X.shape[-1]
    D = d + 1
    L = len(params_tr_layers)

    B = y.shape[0]
    P = X.shape[1]
    P_tr = P - P_test
    if attn_temperature <= 0.0:
        raise ValueError("attn_temperature must be positive")

    mask_y = jnp.ones_like(y)
    mask_y = mask_y.at[:, P_tr:].set(0.0)
    y_channel = (y * mask_y)[..., None]
    h = jnp.concatenate([X, y_channel], axis=-1)  # (B,P,d+1)

    linear_key_mask = jnp.ones((P, P), dtype=h.dtype)
    linear_key_mask = linear_key_mask.at[:, P_tr:].set(0.0)

    causal_mask = jnp.tril(jnp.ones((P, P), dtype=h.dtype))
    neg_inf = jnp.array(-1e9, dtype=h.dtype)
    softmax_mask = (1.0 - causal_mask)[None, :, :] * neg_inf

    for kind, layer in zip(layer_order, params_tr_layers):
        if kind == "linear":
            Wq, Wk, Wv = layer
            q = jnp.einsum('bpd,ed->bpe', h, Wq) / jnp.sqrt(D)
            k = jnp.einsum('bpd,ed->bpe', h, Wk) / jnp.sqrt(D)
            v = jnp.einsum('bpd,ed->bpe', h, Wv)

            att = jnp.einsum('bid,bjd->bij', q, k) * linear_key_mask[None, :, :]
            ctx = jnp.einsum('bij,bjd->bid', att, v)
            h = h + (beta / L) * ctx
        elif kind == "softmax":
            Wq, Wk, Wv, Wo = layer
            q = jnp.einsum('bpd,ed->bpe', h, Wq) / jnp.sqrt(D)
            k = jnp.einsum('bpd,ed->bpe', h, Wk) / jnp.sqrt(D)
            v = jnp.einsum('bpd,ed->bpe', h, Wv) / jnp.sqrt(D)

            att_logits = jnp.einsum('bid,bjd->bij', q, k) + softmax_mask
            att = softmax(att_logits / attn_temperature, axis=-1)
            ctx = jnp.einsum('bij,bjd->bid', att, v)
            h = h + (beta / L) * jnp.einsum('bpd,ed->bpe', ctx, Wo)
        else:
            raise ValueError(f"Unknown hybrid layer kind: {kind}")

    out = h[:, :, -1]
    return out, [], []

# ---------------------------------
# Training function (Optax AdamW)
# ---------------------------------

def train_model(
    X, y, w_train,
    data_params,            # (d, P_tr, P_test, B, batch_size)
    model_params,           # (N, L, beta, gamma, attn_temperature)
    opt_params,             # (T, lr, lamb)
    model_type,
    n_heads=0,
    warmup_steps=None,
    min_lr_ratio=0.1,
    rho=0.01,
    corr_len=0.0,
    correlated_query=False,
    test_every_epoch=0,
    ema_decay=None,
    scatter_eval_size=200,
    zero_value_row=False,
    tied_attention=False,
    init_key=None,
):
    d, P_tr, P_test, B, batch_size = data_params
    N, L, beta_model, gamma, attn_temperature = model_params
    T, lr, weight_decay = opt_params

    X = np.asarray(X)
    y = np.asarray(y)
    w_train = np.asarray(w_train)
    if X.shape[0] != B or y.shape[0] != B:
        raise ValueError("Offline dataset size does not match B")
    if w_train.shape != (B, d):
        raise ValueError(f"w_train must have shape {(B, d)}, got {w_train.shape}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch_size = min(batch_size, B)
    if (zero_value_row or tied_attention) and model_type not in (4, 7):
        raise ValueError("--zeroval and --tiedattention are only supported for pure linear attention models (model_type 4 or 7)")

    linear_use_mlp = model_type == 7

    def constrain_linear_attention_params(params):
        if model_type not in (4, 7):
            return params
        return {
            **params,
            "layers": constrain_linear_attention_layers(
                params["layers"],
                use_mlp=linear_use_mlp,
                tied_attention=tied_attention,
                zero_value_row=zero_value_row,
            ),
        }

    if model_type == 3:
        layers, W_x_embed, W_y_embed, W_out = init_params_transformer(
            d, N, L, sigma=0.4, key=init_key, use_attention=True, use_mlp=True
        )
        params = {
            "layers": layers,
            "W_x_embed": W_x_embed,
            "W_y_embed": W_y_embed,
            "W_out": W_out,
        }
    elif model_type == 4:
        if N != d + 1:
            raise ValueError(f"model_type=4 expects N=d+1={d + 1}, got N={N}")
        params = {
            "layers": init_params_linear_attention_full(
                d,
                L,
                key=init_key,
                use_mlp=False,
                tied_attention=tied_attention,
                zero_value_row=zero_value_row,
            ),
        }
    elif model_type == 5:
        layers, W_x_embed, W_y_embed, W_out = init_params_transformer(
            d, N, L, sigma=0.4, key=init_key, use_attention=True, use_mlp=False
        )
        params = {
            "layers": layers,
            "W_x_embed": W_x_embed,
            "W_y_embed": W_y_embed,
            "W_out": W_out,
        }
    elif model_type == 6:
        layers, W_x_embed, W_y_embed, W_out = init_params_transformer(
            d, N, L, sigma=0.4, key=init_key, use_attention=True, use_mlp=False
        )
        params = {
            "layers": layers,
            "W_x_embed": W_x_embed,
            "W_y_embed": W_y_embed,
            "W_out": W_out,
        }
    elif model_type == 7:
        if N != d + 1:
            raise ValueError(f"model_type=7 expects N=d+1={d + 1}, got N={N}")
        params = {
            "layers": init_params_linear_attention_full(
                d,
                L,
                key=init_key,
                use_mlp=True,
                tied_attention=tied_attention,
                zero_value_row=zero_value_row,
            ),
        }
    elif model_type == 8:
        layers, W_x_embed, W_y_embed, W_out = init_params_transformer(
            d, N, L, sigma=0.4, key=init_key, use_attention=False, use_mlp=True
        )
        params = {
            "layers": layers,
            "W_x_embed": W_x_embed,
            "W_y_embed": W_y_embed,
            "W_out": W_out,
        }
    elif model_type == 9:
        layers, W_x_embed, W_y_embed, W_out = init_params_transformer(
            d, N, L, sigma=0.4, key=init_key, use_attention=True, use_mlp=True
        )
        params = {
            "layers": layers,
            "W_x_embed": W_x_embed,
            "W_y_embed": W_y_embed,
            "W_out": W_out,
        }
    elif model_type == 10:
        if N != d + 1:
            raise ValueError(f"model_type=10 expects N=d+1={d + 1}, got N={N}")
        if L != 2:
            raise ValueError("model_type=10 is a fixed two-layer hybrid and expects L=2")
        params = {
            "layers": init_params_hybrid_attention_full(d, ["linear", "softmax"], key=init_key),
        }
    elif model_type == 11:
        if N != d + 1:
            raise ValueError(f"model_type=11 expects N=d+1={d + 1}, got N={N}")
        if L != 2:
            raise ValueError("model_type=11 is a fixed two-layer hybrid and expects L=2")
        params = {
            "layers": init_params_hybrid_attention_full(d, ["softmax", "linear"], key=init_key),
        }
    else:
        raise ValueError("model_type must be one of {3, 4, 5, 6, 7, 8, 9, 10, 11}")

    # ---------- loss ----------
    if model_type == 3:
        def predict_test_fn(params, X, y):
            out, _, _ = softmax_transformer(
                params["layers"],
                params["W_x_embed"],
                params["W_y_embed"],
                params["W_out"],
                X,
                y,
                n_heads=n_heads,
                P_test=P_test,
                beta=beta_model,
                attn_temperature=attn_temperature,
                use_layernorm=True,
                use_mlp=True,
            )
            return out[:, P_tr:] / gamma
        def loss_fn(params, X, y):
            pred_test = predict_test_fn(params, X, y)
            target_test = y[:, P_tr:]
            return jnp.mean((pred_test - target_test) ** 2)
    elif model_type == 4:
        def predict_test_fn(params, X, y):
            out, _, _ = pure_linear_attention(
                params["layers"],
                X,
                y,
                P_test=P_test,
                beta=beta_model,
                use_mlp=False,
                tied_attention=tied_attention,
            )
            return out[:, P_tr:] / gamma
        def loss_fn(params, X, y):
            pred_test = predict_test_fn(params, X, y)
            target_test = y[:, P_tr:]
            return jnp.mean((pred_test - target_test) ** 2)
    elif model_type == 5:
        def predict_test_fn(params, X, y):
            out, _, _ = softmax_transformer(
                params["layers"],
                params["W_x_embed"],
                params["W_y_embed"],
                params["W_out"],
                X,
                y,
                n_heads=n_heads,
                P_test=P_test,
                beta=beta_model,
                attn_temperature=attn_temperature,
                use_layernorm=False,
                use_mlp=False,
            )
            return out[:, P_tr:] / gamma
        def loss_fn(params, X, y):
            pred_test = predict_test_fn(params, X, y)
            target_test = y[:, P_tr:]
            return jnp.mean((pred_test - target_test) ** 2)
    elif model_type == 6:
        def predict_test_fn(params, X, y):
            out, _, _ = softmax_transformer(
                params["layers"],
                params["W_x_embed"],
                params["W_y_embed"],
                params["W_out"],
                X,
                y,
                n_heads=n_heads,
                P_test=P_test,
                beta=beta_model,
                attn_temperature=attn_temperature,
                use_attention=True,
                use_layernorm=True,
                use_mlp=False,
            )
            return out[:, P_tr:] / gamma
        def loss_fn(params, X, y):
            pred_test = predict_test_fn(params, X, y)
            target_test = y[:, P_tr:]
            return jnp.mean((pred_test - target_test) ** 2)
    elif model_type == 7:
        def predict_test_fn(params, X, y):
            out, _, _ = pure_linear_attention(
                params["layers"],
                X,
                y,
                P_test=P_test,
                beta=beta_model,
                use_mlp=True,
                tied_attention=tied_attention,
            )
            return out[:, P_tr:] / gamma
        def loss_fn(params, X, y):
            pred_test = predict_test_fn(params, X, y)
            target_test = y[:, P_tr:]
            return jnp.mean((pred_test - target_test) ** 2)
    elif model_type == 8:
        def predict_test_fn(params, X, y):
            out, _, _ = softmax_transformer(
                params["layers"],
                params["W_x_embed"],
                params["W_y_embed"],
                params["W_out"],
                X,
                y,
                n_heads=n_heads,
                P_test=P_test,
                beta=beta_model,
                attn_temperature=attn_temperature,
                use_attention=False,
                use_layernorm=False,
                use_mlp=True,
            )
            return out[:, P_tr:] / gamma
        def loss_fn(params, X, y):
            pred_test = predict_test_fn(params, X, y)
            target_test = y[:, P_tr:]
            return jnp.mean((pred_test - target_test) ** 2)
    elif model_type == 9:
        def predict_test_fn(params, X, y):
            out, _, _ = softmax_transformer(
                params["layers"],
                params["W_x_embed"],
                params["W_y_embed"],
                params["W_out"],
                X,
                y,
                n_heads=n_heads,
                P_test=P_test,
                beta=beta_model,
                attn_temperature=attn_temperature,
                use_attention=True,
                use_layernorm=False,
                use_mlp=True,
            )
            return out[:, P_tr:] / gamma
        def loss_fn(params, X, y):
            pred_test = predict_test_fn(params, X, y)
            target_test = y[:, P_tr:]
            return jnp.mean((pred_test - target_test) ** 2)
    elif model_type == 10:
        def predict_test_fn(params, X, y):
            out, _, _ = hybrid_attention_stack(
                params["layers"],
                ["linear", "softmax"],
                X,
                y,
                P_test=P_test,
                beta=beta_model,
                attn_temperature=attn_temperature,
            )
            return out[:, P_tr:] / gamma
        def loss_fn(params, X, y):
            pred_test = predict_test_fn(params, X, y)
            target_test = y[:, P_tr:]
            return jnp.mean((pred_test - target_test) ** 2)
    elif model_type == 11:
        def predict_test_fn(params, X, y):
            out, _, _ = hybrid_attention_stack(
                params["layers"],
                ["softmax", "linear"],
                X,
                y,
                P_test=P_test,
                beta=beta_model,
                attn_temperature=attn_temperature,
            )
            return out[:, P_tr:] / gamma
        def loss_fn(params, X, y):
            pred_test = predict_test_fn(params, X, y)
            target_test = y[:, P_tr:]
            return jnp.mean((pred_test - target_test) ** 2)

    if model_type in (3, 4, 5, 6, 7, 8, 9, 10, 11):
        transforms = None
        labels = None

    if model_type in (3, 4, 5, 6, 7, 8, 9, 10, 11):
        if warmup_steps is None:
            warmup_steps = max(1, int(0.1 * T))
        warmup_steps = min(max(1, warmup_steps), T)
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=lr,
                    transition_steps=warmup_steps,
                ),
                optax.constant_schedule(lr),
            ],
            boundaries=[warmup_steps],
        )
        tx = optax.chain(
            optax.scale_by_adam(),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1.0),
        )
    else:
        tx = optax.multi_transform(transforms, labels)
    opt_state = tx.init(params)

    ema_enabled = ema_decay is not None
    if ema_enabled and not (0.0 < ema_decay < 1.0):
        raise ValueError("ema_decay must lie in (0, 1)")
    ema_params = params if ema_enabled else None

    if ema_enabled:
        @jit
        def train_step(params, ema_params, opt_state, X, y):
            loss_val, grads = value_and_grad(loss_fn)(params, X, y)
            grads = constrain_linear_attention_params(grads)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = constrain_linear_attention_params(params)
            ema_params = optax.incremental_update(params, ema_params, 1.0 - ema_decay)
            ema_params = constrain_linear_attention_params(ema_params)
            return params, ema_params, opt_state, loss_val
    else:
        @jit
        def train_step(params, opt_state, X, y):
            loss_val, grads = value_and_grad(loss_fn)(params, X, y)
            grads = constrain_linear_attention_params(grads)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = constrain_linear_attention_params(params)
            return params, opt_state, loss_val

    @jit
    def eval_step(params, X, y):
        return loss_fn(params, X, y)

    @jit
    def predict_step(params, X, y):
        return predict_test_fn(params, X, y)

    loss_history = []
    train_eval_history = []
    train_eval_steps = []
    same_task_history = []
    same_task_steps = []
    fresh_test_history = []
    fresh_test_steps = []
    best_fresh_test_loss = None
    best_fresh_test_step = None
    best_eval_params = None
    perm = np.arange(B)
    batch_start = 0

    eval_interval = None
    if test_every_epoch and test_every_epoch > 0:
        eval_interval = max(1, T // test_every_epoch)

    for step in range(T):
        if batch_size == B:
            X_batch = X
            y_batch = y
        else:
            if batch_start == 0:
                perm = np.random.permutation(B)
            batch_end = batch_start + batch_size
            if batch_end <= B:
                batch_idx = perm[batch_start:batch_end]
            else:
                head = perm[batch_start:]
                perm = np.random.permutation(B)
                tail = perm[:batch_end - B]
                batch_idx = np.concatenate((head, tail))
            batch_start = batch_end % B
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

        if ema_enabled:
            params, ema_params, opt_state, loss_val = train_step(params, ema_params, opt_state, X_batch, y_batch)
        else:
            params, opt_state, loss_val = train_step(params, opt_state, X_batch, y_batch)
        loss_history.append(float(loss_val))
        if eval_interval is not None and (step + 1) % eval_interval == 0:
            eval_params = ema_params if ema_enabled else params
            train_eval_loss = eval_step(eval_params, X, y)
            train_eval_steps.append(step + 1)
            train_eval_history.append(float(train_eval_loss))

            X_same, y_same = draw_pretraining_data_fixedtasks(
                B, d, P_tr, rho, corr_len, w_train, correlated_query=correlated_query
            )
            same_task_loss = eval_step(eval_params, X_same, y_same)
            same_task_steps.append(step + 1)
            same_task_history.append(float(same_task_loss))

            X_fresh, y_fresh, _ = draw_pretraining_data(
                5 * B, d, P_tr, -1, rho, corr_len, correlated_query=correlated_query
            )
            fresh_loss = eval_step(eval_params, X_fresh, y_fresh)
            fresh_loss_float = float(fresh_loss)
            fresh_test_steps.append(step + 1)
            fresh_test_history.append(fresh_loss_float)
            if best_fresh_test_loss is None or fresh_loss_float < best_fresh_test_loss:
                best_fresh_test_loss = fresh_loss_float
                best_fresh_test_step = step + 1
                best_eval_params = eval_params

    if eval_interval is not None and (not fresh_test_steps or fresh_test_steps[-1] != T):
        eval_params = ema_params if ema_enabled else params
        train_eval_loss = eval_step(eval_params, X, y)
        train_eval_steps.append(T)
        train_eval_history.append(float(train_eval_loss))

        X_same, y_same = draw_pretraining_data_fixedtasks(
            B, d, P_tr, rho, corr_len, w_train, correlated_query=correlated_query
        )
        same_task_loss = eval_step(eval_params, X_same, y_same)
        same_task_steps.append(T)
        same_task_history.append(float(same_task_loss))

        X_fresh, y_fresh, _ = draw_pretraining_data(
            5 * B, d, P_tr, -1, rho, corr_len, correlated_query=correlated_query
        )
        fresh_loss = eval_step(eval_params, X_fresh, y_fresh)
        fresh_loss_float = float(fresh_loss)
        fresh_test_steps.append(T)
        fresh_test_history.append(fresh_loss_float)
        if best_fresh_test_loss is None or fresh_loss_float < best_fresh_test_loss:
            best_fresh_test_loss = fresh_loss_float
            best_fresh_test_step = T
            best_eval_params = eval_params

    if best_eval_params is None:
        best_eval_params = ema_params if ema_enabled else params
        best_fresh_test_step = T
        best_fresh_test_loss = float(fresh_test_history[-1]) if fresh_test_history else None

    X_scatter, y_scatter, _ = draw_pretraining_data(
        scatter_eval_size, d, P_tr, -1, rho, corr_len, correlated_query=correlated_query
    )
    scatter_preds = np.asarray(predict_step(best_eval_params, X_scatter, y_scatter))
    scatter_y_true = np.asarray(y_scatter[:, P_tr:]).reshape(-1)
    scatter_y_pred = scatter_preds.reshape(-1)

    return (
        params,
        best_eval_params,
        loss_history,
        train_eval_steps,
        train_eval_history,
        same_task_steps,
        same_task_history,
        fresh_test_steps,
        fresh_test_history,
        best_fresh_test_step,
        best_fresh_test_loss,
        scatter_y_true,
        scatter_y_pred,
    )

def draw_pretraining_data(n, d, l, k, rho, corr_len, use_these_ws=None, correlated_query=False):
    if correlated_query:
        X = sample_ar1_batch(n, l + 1, d, corr_len)
    else:
        x_context = sample_ar1_batch(n, l, d, corr_len)
        x_query = np.random.randn(n, 1, d) / np.sqrt(d)
        X = np.concatenate([x_context, x_query], axis=1)

    if use_these_ws is not None:
        w = np.asarray(use_these_ws)
        if w.shape != (n, d):
            raise ValueError(
                f"use_these_ws must have shape {(n, d)}, got {w.shape}"
            )
    elif k == -1:
        w = np.random.randn(n, d)
    else:
        w_bank = np.random.randn(k, d)
        task_ids = np.random.randint(0, k, size=n)
        w = w_bank[task_ids]

    epsilon = np.random.randn(n, l + 1) * np.sqrt(rho)
    y = np.einsum('npd,nd->np', X, w) + epsilon
    return X, y, w

def draw_pretraining_data_fixedtasks(n, d, l, rho, corr_len, w_set, correlated_query=False):
    X, y, _ = draw_pretraining_data(
        n,
        d,
        l,
        -1,
        rho,
        corr_len,
        use_these_ws=w_set,
        correlated_query=correlated_query,
    )
    return X, y

def append_experiment_row(csv_path, row):
    fieldnames = [
        "d",
        "L",
        "P_tr",
        "P_test",
        "N",
        "K",
        "corr_len",
        "correlated_query",
        "B",
        "batch_size",
        "ema_decay",
        "lr",
        "T",
        "model_type",
        "number_of_heads",
        "beta_model",
        "attn_temperature",
        "gamma",
        "rho",
        "lamb",
        "test_every_epoch",
        "losses_json",
        "train_eval_steps_json",
        "train_eval_losses_json",
        "same_task_steps_json",
        "same_task_losses_json",
        "fresh_test_steps_json",
        "fresh_test_losses_json",
        "best_fresh_test_step",
        "best_fresh_test_loss",
        "scatter_eval_size",
        "scatter_y_true_json",
        "scatter_y_pred_json",
    ]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the unrolled models with sequentially correlated contexts and a single query token."
    )
    parser.add_argument(
        "--experiments_csv",
        type=str,
        default=None,
        help="Path to the global experiment log CSV. Defaults to experiments.csv next to this script.",
    )
    parser.add_argument("--d", type=int, default=64, help="Token dim")
    parser.add_argument("--B", type=int, default=512, help="Offline training set size")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optimizer minibatch size. Defaults to full-batch training on all B examples.",
    )
    parser.add_argument("--P_tr", type=int, default=128, help="Train context length")
    parser.add_argument("--corr_len", type=float, default=0.0, help="Correlation length in exponential kernel")
    parser.add_argument(
        "--correlated_query",
        action="store_true",
        help="If set, the single query token continues the same AR(1) process as the context.",
    )
    parser.add_argument(
        "--P_test",
        type=int,
        default=1,
        help="Held-out query count. This sequential-correlation variant only supports P_test=1.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=32,
        help="Number of unique training tasks. Use -1 for fresh tasks with no shared bank.",
    )
    parser.add_argument("--rho", type=float, default=0.01, help='Label noise')

    parser.add_argument("--N", type=int, default=128, help="Model width")
    parser.add_argument("--L", type=int, default=1, help="Model depth")
    parser.add_argument("--beta_model", type=float, default=1.0)
    parser.add_argument(
        "--attn_temperature",
        type=float,
        default=1.0,
        help="Temperature applied to softmax attention logits for softmax-based architectures.",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=1e-4)
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=None,
        help="If set, evaluate and export predictions using an EMA of the weights with this decay.",
    )
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--T", type=int, required=True, help='Total training steps')
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Warmup steps for the transformer scheduler. Default is 10%% of T for model_type=3.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="Final learning rate as a fraction of peak lr for model_type=3.",
    )
    
    parser.add_argument(
        "--model_type",
        type=int,
        required=True,
        help=(
            "3=softmax+LN+MLP, 4=pure linear attention, 5=softmax only, "
            "6=softmax+LN, 7=linear attention+MLP, 8=MLP only, 9=softmax+MLP, "
            "10=linear->softmax, 11=softmax->linear"
        ),
    )
    parser.add_argument("--number_of_heads", type=int, default=1)
    parser.add_argument(
        "--test_every_epoch",
        type=int,
        default=0,
        help="If > 0, evaluate on freshly sampled tasks every T // test_every_epoch steps.",
    )
    parser.add_argument(
        "--scatter_eval_size",
        type=int,
        default=0,
        help="Number of fresh sequences sampled after training for exporting y_true vs y_model arrays.",
    )
    parser.add_argument(
        "--zeroval",
        action="store_true",
        help="For pure linear attention, keep V[-1, :-1] fixed at zero throughout training.",
    )
    parser.add_argument(
        "--tiedattention",
        action="store_true",
        help="For pure linear attention, train a single bilinear attention matrix instead of separate Q and K matrices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, use this seed for NumPy sampling/shuffling and JAX initialization.",
    )
 
    return parser.parse_args()

def main():
    args = parse_args()
    experiments_csv = args.experiments_csv
    if experiments_csv is None:
        experiments_csv = Path(__file__).resolve().with_name("experiments.csv")

    d = args.d
    P_tr = args.P_tr
    corr_len = args.corr_len
    correlated_query = args.correlated_query
    P_test = args.P_test
    if P_test != 1:
        raise ValueError("architectures_seqcorr.py only supports P_test=1")
    B = args.B
    batch_size = args.batch_size if args.batch_size is not None else B
    N = args.N
    L = args.L
    num_tasks = args.K
    rho = args.rho

    beta_model = args.beta_model
    attn_temperature = args.attn_temperature
    gamma = args.gamma
    lamb = args.lamb
    ema_decay = args.ema_decay
    T = args.T
    lr = args.lr
    warmup_steps = args.warmup_steps
    min_lr_ratio = args.min_lr_ratio
    model_type = args.model_type
    number_of_heads = args.number_of_heads
    test_every_epoch = args.test_every_epoch
    scatter_eval_size = args.scatter_eval_size
    zero_value_row = args.zeroval
    tied_attention = args.tiedattention
    seed = args.seed

    if seed is not None:
        np.random.seed(seed)
        init_key = jax.random.PRNGKey(seed)
        print(f"using seed={seed}", flush=True)
    else:
        init_key = None

    data_params = [d, P_tr, P_test, B, batch_size]
    opt_params = [T, lr, lamb]
    model_params = [N, L, beta_model, gamma, attn_temperature]

    print("parameters done", flush=True)

    X, y, w_train = draw_pretraining_data(
        B, d, P_tr, num_tasks, rho, corr_len, correlated_query=correlated_query
    )
    print("sampling done", flush=True)

    (
        _,
        _,
        losses,
        train_eval_steps,
        train_eval_losses,
        same_task_steps,
        same_task_losses,
        fresh_test_steps,
        fresh_test_losses,
        best_fresh_test_step,
        best_fresh_test_loss,
        scatter_y_true,
        scatter_y_pred,
    ) = train_model(
        X,
        y,
        w_train,
        data_params,
        model_params,
        opt_params,
        model_type,
        number_of_heads,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        rho=rho,
        corr_len=corr_len,
        correlated_query=correlated_query,
        test_every_epoch=test_every_epoch,
        ema_decay=ema_decay,
        scatter_eval_size=scatter_eval_size,
        zero_value_row=zero_value_row,
        tied_attention=tied_attention,
        init_key=init_key,
    )
    print("training done", flush=True)
    append_experiment_row(
        experiments_csv,
        {
            "d": d,
            "L": L,
            "P_tr": P_tr,
            "P_test": P_test,
            "N": N,
            "K": num_tasks,
            "corr_len": corr_len,
            "correlated_query": correlated_query,
            "B": B,
            "batch_size": batch_size,
            "ema_decay": ema_decay,
            "lr": lr,
            "T": T,
            "model_type": model_type,
            "number_of_heads": number_of_heads,
            "beta_model": beta_model,
            "attn_temperature": attn_temperature,
            "gamma": gamma,
            "rho": rho,
            "lamb": lamb,
            "test_every_epoch": test_every_epoch,
            "losses_json": json.dumps(losses),
            "train_eval_steps_json": json.dumps(train_eval_steps),
            "train_eval_losses_json": json.dumps(train_eval_losses),
            "same_task_steps_json": json.dumps(same_task_steps),
            "same_task_losses_json": json.dumps(same_task_losses),
            "fresh_test_steps_json": json.dumps(fresh_test_steps),
            "fresh_test_losses_json": json.dumps(fresh_test_losses),
            "best_fresh_test_step": best_fresh_test_step,
            "best_fresh_test_loss": best_fresh_test_loss,
            "scatter_eval_size": scatter_eval_size,
            "scatter_y_true_json": json.dumps(scatter_y_true.tolist()),
            "scatter_y_pred_json": json.dumps(scatter_y_pred.tolist()),
        },
    )
    print(f"logged results to {experiments_csv}", flush=True)

if __name__ == "__main__":
    main()

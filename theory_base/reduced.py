
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, lax
from jax.scipy.sparse.linalg import cg
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Kernel construction + stats
# -----------------------
def kernel_exp(l, corr_len):
    idx = jnp.arange(l + 1, dtype=jnp.float32)
    K = jnp.where(
        corr_len == 0.0,
        jnp.eye(l + 1, dtype=jnp.float32),
        jnp.exp(-jnp.abs(idx[:, None] - idx[None, :]) / corr_len).astype(jnp.float32),
    )
    return K

def icl_kernel_stats(K):
    k = K[:-1, -1]            # (l,)
    Ksub = K[:-1, :-1]        # (l,l)
    k0 = k @ k
    k1 = k @ (Ksub @ k)
    l_ = Ksub.shape[0]
    tk  = (1.0 / l_) * jnp.trace(Ksub)
    tk2 = (1.0 / l_) * jnp.trace(Ksub @ Ksub)
    return k0, k1, tk, tk2

# -----------------------
# AR(1) sampling for exponential kernel
# -----------------------
def simulate_sufficient_stats_finite_k(key, n, d, l, rho, C, corr_len, k_tasks, old_w_set=None):
    key_wset, key_widx, key_scan = random.split(key, 3)
    L = l + 1

    dtype = C.dtype
    rho = jnp.asarray(rho, dtype=dtype)
    corr_len = jnp.asarray(corr_len, dtype=dtype)

    # Task bank
    sqrtC = jnp.linalg.cholesky(C + jnp.asarray(1e-12, dtype) * jnp.eye(d, dtype=dtype))
    if old_w_set==None:
        w_set = random.normal(key_wset, (k_tasks, d), dtype=dtype) @ sqrtC.T
    else:
        w_set = old_w_set
    w_idx = random.randint(key_widx, (n,), 0, k_tasks)
    w = w_set[w_idx]  # (n,d)

    # AR(1) params for exp kernel
    phi = jnp.where(
        corr_len == jnp.asarray(0, dtype),
        jnp.asarray(0, dtype),
        jnp.exp(-jnp.asarray(1, dtype) / corr_len),
    )
    sigma = jnp.sqrt(jnp.maximum(jnp.asarray(1, dtype) - phi * phi, jnp.asarray(0, dtype)))
    inv_sqrt_d = jnp.asarray(1, dtype) / jnp.sqrt(jnp.asarray(d, dtype))

    # t = 0 draw
    key_scan, kx0, ke0 = random.split(key_scan, 3)
    z0 = random.normal(kx0, (n, d), dtype=dtype)
    e0 = random.normal(ke0, (n,), dtype=dtype) * jnp.sqrt(rho)

    x0 = z0 * inv_sqrt_d
    y0 = jnp.sum(x0 * w, axis=1) + e0

    # initialize sums over t < l (includes t=0 if l>0)
    y_sum_x0 = y0[:, None] * x0
    y_sum_y0 = y0 * y0

    # If l==0, we should NOT include t=0 in training sums.
    y_sum_x = lax.cond(l > 0,
                       lambda _: y_sum_x0,
                       lambda _: jnp.zeros((n, d), dtype=dtype),
                       operand=None)
    y_sum_y = lax.cond(l > 0,
                       lambda _: y_sum_y0,
                       lambda _: jnp.zeros((n,), dtype=dtype),
                       operand=None)

    def step(carry, t):
        key_t, x_prev, y_sum_x, y_sum_y, y_last = carry

        key_t, kx, ke = random.split(key_t, 3)
        z_t = random.normal(kx, (n, d), dtype=dtype)
        e_t = random.normal(ke, (n,), dtype=dtype) * jnp.sqrt(rho)

        # ✅ Correct AR(1) scaling: only innovation has 1/sqrt(d)
        x_t = phi * x_prev + sigma * z_t * inv_sqrt_d
        y_t = jnp.sum(x_t * w, axis=1) + e_t

        def accum_fn(args):
            y_sum_x, y_sum_y = args
            y_sum_x = y_sum_x + y_t[:, None] * x_t
            y_sum_y = y_sum_y + y_t * y_t
            return (y_sum_x, y_sum_y)

        # accumulate for training times t < l
        y_sum_x, y_sum_y = lax.cond(t < l, accum_fn, lambda a: a, (y_sum_x, y_sum_y))

        return (key_t, x_t, y_sum_x, y_sum_y, y_t), None  # keep only last y in carry

    # scan t = 1..L-1, last step yields x_l and y_l in carry
    init = (key_scan, x0, y_sum_x, y_sum_y, y0)
    (key_end, x_last, y_sum_x, y_sum_y, y_last), _ = lax.scan(step, init, jnp.arange(1, L))

    return x_last, y_sum_x, y_sum_y, y_last, w_set

def simulate_sufficient_stats_INDEPENDENT_QUERY(key, n, d, l, rho, C, corr_len, k_tasks, old_w_set=None):
    key_wset, key_widx, key_scan, key_query = random.split(key, 4)
    L = l + 1

    dtype = C.dtype
    rho = jnp.asarray(rho, dtype=dtype)
    corr_len = jnp.asarray(corr_len, dtype=dtype)

    # Task bank
    sqrtC = jnp.linalg.cholesky(C + jnp.asarray(1e-12, dtype) * jnp.eye(d, dtype=dtype))
    if old_w_set == None:
        w_set = random.normal(key_wset, (k_tasks, d), dtype=dtype) @ sqrtC.T
    else:
        w_set = old_w_set
    w_idx = random.randint(key_widx, (n,), 0, k_tasks)
    w = w_set[w_idx]  # (n,d)

    # AR(1) params for exp kernel
    phi = jnp.where(
        corr_len == jnp.asarray(0, dtype),
        jnp.asarray(0, dtype),
        jnp.exp(-jnp.asarray(1, dtype) / corr_len),
    )
    sigma = jnp.sqrt(jnp.maximum(jnp.asarray(1, dtype) - phi * phi, jnp.asarray(0, dtype)))
    inv_sqrt_d = jnp.asarray(1, dtype) / jnp.sqrt(jnp.asarray(d, dtype))

    # t = 0 draw
    key_scan, kx0, ke0 = random.split(key_scan, 3)
    z0 = random.normal(kx0, (n, d), dtype=dtype)
    e0 = random.normal(ke0, (n,), dtype=dtype) * jnp.sqrt(rho)

    x0 = z0 * inv_sqrt_d
    y0 = jnp.sum(x0 * w, axis=1) + e0

    # initialize sums over t < l (includes t=0 if l>0)
    y_sum_x0 = y0[:, None] * x0
    y_sum_y0 = y0 * y0

    # If l==0, do NOT include t=0 in training sums
    y_sum_x = lax.cond(
        l > 0,
        lambda _: y_sum_x0,
        lambda _: jnp.zeros((n, d), dtype=dtype),
        operand=None,
    )
    y_sum_y = lax.cond(
        l > 0,
        lambda _: y_sum_y0,
        lambda _: jnp.zeros((n,), dtype=dtype),
        operand=None,
    )

    def step(carry, t):
        key_t, x_prev, y_sum_x, y_sum_y = carry

        key_t, kx, ke = random.split(key_t, 3)
        z_t = random.normal(kx, (n, d), dtype=dtype)
        e_t = random.normal(ke, (n,), dtype=dtype) * jnp.sqrt(rho)

        # Correlated AR(1) training sample
        x_t = phi * x_prev + sigma * z_t * inv_sqrt_d
        y_t = jnp.sum(x_t * w, axis=1) + e_t

        def accum_fn(args):
            y_sum_x, y_sum_y = args
            y_sum_x = y_sum_x + y_t[:, None] * x_t
            y_sum_y = y_sum_y + y_t * y_t
            return y_sum_x, y_sum_y

        # accumulate only for training times t < l
        y_sum_x, y_sum_y = lax.cond(t < l, accum_fn, lambda a: a, (y_sum_x, y_sum_y))

        return (key_t, x_t, y_sum_x, y_sum_y), None

    # Run correlated process only up to time l-1
    # For l=0 or l=1 this scan is empty, which is fine.
    init = (key_scan, x0, y_sum_x, y_sum_y)
    (_, _, y_sum_x, y_sum_y), _ = lax.scan(step, init, jnp.arange(1, l))

    # Independent query sample x_l with same marginal covariance I/d
    key_query, kx_last, ke_last = random.split(key_query, 3)
    z_last = random.normal(kx_last, (n, d), dtype=dtype)
    e_last = random.normal(ke_last, (n,), dtype=dtype) * jnp.sqrt(rho)

    x_last = z_last * inv_sqrt_d
    y_last = jnp.sum(x_last * w, axis=1) + e_last

    return x_last, y_sum_x, y_sum_y, y_last, w_set
    
# -----------------------
# Matrix-free H and H^T operators
# -----------------------
def Hv(v_mat, x_l, y_sum_x, y_sum_y, d, l):
    vA = v_mat[:, :d]     # (d,d)
    v_last = v_mat[:, d]  # (d,)

    tmp = y_sum_x @ vA.T
    term1 = (d / l) * jnp.sum(x_l * tmp, axis=1)
    term2 = (1.0 / l) * y_sum_y * (x_l @ v_last)
    return term1 + term2

def HTu(u, x_l, y_sum_x, y_sum_y, d, l):
    Xu = x_l * u[:, None]
    A = (d / l) * (Xu.T @ y_sum_x)
    b = (1.0 / l) * (x_l.T @ (u * y_sum_y))
    return jnp.concatenate([A, b[:, None]], axis=1)

# -----------------------
# Per-sample predictor from Gamma
# -----------------------
def predict_from_Gamma(Gamma, x_l, y_sum_x, y_sum_y, d, l):
    Gamma_A = Gamma[:, :d]   # (d,d)
    Gamma_b = Gamma[:, d]    # (d,)

    # First block contribution:
    term1 = (d / l) * jnp.sum((x_l @ Gamma_A) * y_sum_x, axis=1)
    # Last column contribution:
    term2 = (1.0 / l) * y_sum_y * (x_l @ Gamma_b)

    return term1 + term2

# -----------------------
# Direct Monte Carlo MSE from fresh sampled data
# -----------------------
def mse_from_data(key, Gamma, n_eval, d, l, rho, C, corr_len, query):
    if query: 
        x_l, y_sum_x, y_sum_y, y_last, _ = simulate_sufficient_stats_finite_k(
            key, n_eval, d, l, rho, np.eye(d), corr_len, k_tasks=n_eval)
    else:
        x_l, y_sum_x, y_sum_y, y_last, _ = simulate_sufficient_stats_INDEPENDENT_QUERY(
            key, n=n_eval, d=d, l=l, rho=rho, C=C, corr_len=corr_len, k_tasks=n_eval)

    y_hat = predict_from_Gamma(Gamma, x_l, y_sum_x, y_sum_y, d, l)
    mse = jnp.mean((y_hat - y_last) ** 2)
    return mse, y_hat, y_last

# -----------------------
# Conjugate gradient solution for Gamma
# -----------------------
def gamma_cg_operator(x_l, y_sum_x, y_sum_y, y_last, n, d, l, lambdareg,
                      cg_tol=1e-10, cg_maxiter=5000):

    p = d * (d + 1)
    reg = (n / d) * lambdareg

    def matvec(v_flat):
        v_mat = v_flat.reshape(d, d + 1)
        hv = Hv(v_mat, x_l, y_sum_x, y_sum_y, d, l)
        ht_h_v = HTu(hv, x_l, y_sum_x, y_sum_y, d, l)
        return (reg * v_mat + ht_h_v).reshape(p)

    b_mat = HTu(y_last, x_l, y_sum_x, y_sum_y, d, l)
    b = b_mat.reshape(p)

    gamma, _ = cg(matvec, b, tol=cg_tol, maxiter=cg_maxiter)

    # report residual
    r = b - matvec(gamma)
    rel_res = jnp.linalg.norm(r) / (jnp.linalg.norm(b) + 1e-30)

    return gamma.reshape(d, d + 1), rel_res

@partial(
    jax.jit,
    static_argnames=("n", "d", "l", "k_tasks", "cg_maxiter", "query"),
)
def _one_replicate_from_key(
    key,
    n,
    d,
    l,
    k_tasks,
    rho,
    corr_len,
    lambdareg=1e-8,
    cg_tol=1e-4,
    cg_maxiter=2000,
    query=False,
):
    K = kernel_exp(l, corr_len)
    _, _, tk, tk2 = icl_kernel_stats(K)

    if query:
        x_last, y_sum_x, y_sum_y, y_last, _ = simulate_sufficient_stats_finite_k(
            key, n, d, l, rho, jnp.eye(d), corr_len, k_tasks)
    else:
        x_last, y_sum_x, y_sum_y, y_last, _ = simulate_sufficient_stats_INDEPENDENT_QUERY(
            key, n=n, d=d, l=l, rho=rho, C=jnp.eye(d), corr_len=corr_len, k_tasks=k_tasks)

    Gamma, rel_res = gamma_cg_operator(x_last, y_sum_x, y_sum_y, y_last, n, d, l, lambdareg, cg_tol, cg_maxiter)
    error_data, _, _ = mse_from_data(5 * key, Gamma, n_eval=5 * n, d=d, l=l, rho=rho, C=jnp.eye(d), corr_len=corr_len, query=query)

    return error_data


def one_replicate(n, d, l, k_tasks, rho, corr_len, lambdareg=1e-8, cg_tol=1e-4, cg_maxiter=2000, seed=0, query=False):
    key = random.PRNGKey(seed)
    return _one_replicate_from_key(
        key,
        n,
        d,
        l,
        k_tasks,
        rho,
        corr_len,
        lambdareg=lambdareg,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        query=query,
    )


@partial(
    jax.jit,
    static_argnames=("num_average", "n", "d", "l", "k_tasks", "cg_maxiter", "query"),
)
def average_over_replicate(
    n,
    d,
    l,
    k_tasks,
    rho,
    corr_len,
    num_average,
    lambdareg=1e-8,
    cg_tol=1e-4,
    cg_maxiter=2000,
    seed=0,
    query=False,
):
    keys = random.split(random.PRNGKey(seed), num_average)
    errors = jax.vmap(
        lambda key: _one_replicate_from_key(
            key,
            n,
            d,
            l,
            k_tasks,
            rho,
            corr_len,
            lambdareg=lambdareg,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
            query=query,
        )
    )(keys)
    return jnp.mean(errors), jnp.std(errors)

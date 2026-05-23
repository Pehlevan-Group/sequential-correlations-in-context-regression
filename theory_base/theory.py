import numpy as np

# -----------------------
# Stieltjes transforms
# -----------------------
def kernel_exp(l, corr_len):
    idx = np.arange(l + 1, dtype=np.float32)
    if corr_len == 0:
        return np.eye(l+1)
    else:
        return np.exp(-np.abs(idx[:, None] - idx[None, :]) / corr_len)

def icl_kernel_stats(K):
    k = K[:-1, -1]            # (l,)
    Ksub = K[:-1, :-1]        # (l,l)
    k0 = k @ k
    k1 = k @ (Ksub @ k)
    l_ = Ksub.shape[0]
    tk  = (1.0 / l_) * np.trace(Ksub)
    tk2 = (1.0 / l_) * np.trace(Ksub @ Ksub)
    return k0, k1, tk, tk2

# -----------------------
# Stieltjes transforms
# -----------------------
def M_kappa(nu, kappa, c):
    # Ctr = c*I
    return 2 / ( (nu + c - c/kappa) + np.sqrt((nu + c - c/kappa)**2 + 4*c*nu/kappa) )
def M_kappa_prime(nu, kappa, c):
    M = M_kappa(nu, kappa, c)
    return (-1/2) * M**2 * (1 + (nu + c + c/kappa)/np.sqrt((nu + c - c/kappa)**2 + 4*c*nu/kappa))

# -----------------------
# Current error formula
# -----------------------
def icl_correlated_REARRANGED(tau, alpha, kappa, rho, K):
    # --- necessary constants ---
    k = K[:-1, -1]; K_sub = K[:-1,:-1]
    k0 = k@k
    k1 = k@K_sub@k
    ell = K_sub.shape[0]
    rho1 = 1.0 + rho
    rho2 = (1/ell)*np.trace(K_sub@K_sub) + rho
    phi1 = (k1 + rho * k0) / (alpha**2)
    phi2 = k0 / alpha
    psi1, psi2 = phi1, phi2

    # --- effective ridges ---
    if tau >= 1:
        tilde_lambda = 0
    else:
        tilde_lambda = ((1-tau)/tau)/M_kappa(rho2/alpha, kappa/tau, 1)
    sigma = tilde_lambda + rho2/alpha
    tilde_sigma = sigma - k0 / alpha

    # --- stieltjes transforms ---
    calM = M_kappa(sigma, kappa, 1)
    calM_prime = M_kappa_prime(sigma, kappa, 1)

    # --- vectors m1, m2, m3 (shape (2,)) ---
    m1 = np.array([
        calM,
        phi2 * (1.0 - sigma * calM),
    ], dtype=float)

    m2 = np.array([
        1.0 - tilde_sigma * calM,
        phi2 * (1.0 - tilde_sigma + sigma * tilde_sigma * calM),
    ], dtype=float)

    m3 = np.array([
        calM + tilde_sigma * calM_prime,
        phi2 * (1.0 - sigma * calM - tilde_sigma * calM - sigma * tilde_sigma * calM_prime),
    ], dtype=float)

    # --- 2x2 matrices ---
    MatM = np.array([
        [-calM_prime,                 phi2 * (calM + sigma * calM_prime)],
        [ phi2 * (calM + sigma * calM_prime),  (phi2**2) * (1.0 - 2.0 * sigma * calM - (sigma**2) * calM_prime)],
    ], dtype=float)
    S_inv = np.array([
        [calM,                         1.0 + phi2 * (1.0 - sigma * calM)],
        [1.0 + phi2 * (1.0 - sigma * calM),   -phi1 + (phi2**2) * (1.0 - sigma + (sigma**2) * calM)],
    ], dtype=float)
    S = np.linalg.inv(S_inv)

    # helper quadratic forms
    def qform(v, A, w):
        return float(v.T @ A @ w)

    # --- q ---
    denom_q = tau - (1.0 - 2.0 * tilde_lambda * calM - (tilde_lambda**2) * calM_prime)
    old_q = (rho + sigma - (sigma**2) * calM - tilde_lambda * (1 - 2*sigma*calM - (sigma**2)*calM_prime)) / denom_q

    new_q = (qform(m2, S, m2) - tilde_lambda * (- 2.0 * qform(m2, S, m3) + float(m2.T @ S.T @ MatM @ S @ m2))
            - 2*phi2*(tilde_lambda*(calM + sigma*calM_prime) + 1 - sigma*calM)
            - (phi2**2)*(calM - tilde_lambda*calM_prime))
    new_q = new_q/denom_q

    # --- independent part ---
    e_ICL_independent_query = (rho
        + (rho2 / alpha)
        * (
            1
            + (old_q - 2 * sigma) * calM
            + (old_q * tilde_lambda - sigma**2) * calM_prime
        )
        + old_q * calM + (old_q * tilde_lambda - sigma**2) * calM_prime
    )

    tail = (
        - 2 * psi2
        + 2 * psi2 * (tilde_sigma + rho2 / alpha) * calM
        + (1 + rho2 / alpha) * (
            new_q * (calM + tilde_lambda * calM_prime)
            + psi2 * (2 * sigma - psi2) * calM_prime
            + float(m2.T @ S.T @ MatM @ S @ m2)
            - 2 * qform(m2, S, m3)
        )
        + 2 * (1 + psi2) * qform(m1, S, m2)
        + (psi1 + 2 * psi2) * (1 - tilde_sigma * calM - qform(m1, S, m2)) ** 2
    )

    return e_ICL_independent_query, tail

# -----------------------
# Original error formula
# -----------------------
def icl_uncorrelated(tau, alpha_tr, alpha_test, kappa, rho, ctr, ctest):
    if tau == 1:
        return None
    if tau > 1:
        xi = 0
        nu = (rho + ctr)/alpha_tr
    if tau < 1:
        xi = ((1-tau)/tau)/M_kappa((rho + ctr)/alpha_tr, kappa/tau, ctr)
        nu = (rho + ctr)/alpha_tr + xi

    M = M_kappa(nu, kappa, ctr)
    Mprime = M_kappa_prime(nu,kappa,ctr)
    zero = ctest + rho
    linear = ctest*(1 - nu*M)
    quadratic_eq = ((ctest+rho)/alpha_test + ctest)*(1 - 2*nu*M - (nu**2)*Mprime)
    c_e = ((rho + ctr) - (ctr-nu+(nu**2)*M) - xi*(1 - 2*nu*M - (nu**2)*Mprime))/(1 - 2*xi*M - (xi**2)*Mprime - tau)
    quadratic_extra = -c_e * ((ctest+rho)/alpha_test + ctest) * (M + xi*Mprime)
    return zero -2*linear + quadratic_eq + quadratic_extra
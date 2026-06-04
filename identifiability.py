"""
Identifiability Analysis for Latent World Models.

Implements metrics and tools to evaluate whether a learned JEPA encoder
recovers the true latent factors of an environment up to a well-defined
equivalence class (affine, permutation, element-wise nonlinearity).

Theoretical grounding:
    - Nonlinear ICA theory (Hyvarinen & Morioka 2016, Khemakhem et al. 2020)
      shows that temporal structure + auxiliary variables (here: actions) can
      make latent representations identifiable up to element-wise transforms.
    - SIGReg's isotropic Gaussian constraint further restricts the equivalence
      class toward orthogonal/affine maps.
    - Action diversity is the key variable: more diverse actions → stronger
      identifiability guarantees.

Key result we test empirically:
    With sufficiently diverse actions, the JEPA encoder should be identifiable
    up to an affine map from the true factors. Without action diversity (or
    without action conditioning), identifiability degrades to arbitrary
    diffeomorphisms — i.e., the representation is not meaningfully structured.
"""

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# ===========================================================================
#                       2.  Affine Identifiability (R²)
# ===========================================================================

def affine_identifiability(
    z: np.ndarray,
    s: np.ndarray,
    train_frac: float = 1.0,
    seed: int = 42,
) -> dict:
    """Measure identifiability up to affine transformations.

    Fits the best affine map A such that s ≈ A @ z + b, and reports R².
    If identifiability holds up to affine transforms, R² should be ~1.0.

    Args:
        z: (N, D_z) learned latent representations
        s: (N, D_s) ground-truth state factors
        train_frac: fraction of data used for training the probe.
            When < 1.0, metrics are reported on the held-out test set
            (paper-style evaluation). Default 1.0 = in-sample.
        seed: random seed for train/test split

    Returns:
        dict with 'r2_per_factor' (D_s,), 'r2_mean', 'coefficients'
    """
    z = _to_numpy(z)
    s = _to_numpy(s)

    if train_frac < 1.0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(z))
        n_train = int(len(z) * train_frac)
        z_train, z_eval = z[idx[:n_train]], z[idx[n_train:]]
        s_train, s_eval = s[idx[:n_train]], s[idx[n_train:]]
    else:
        z_train, z_eval = z, z
        s_train, s_eval = s, s

    reg = Ridge(alpha=1e-4)
    reg.fit(z_train, s_train)
    s_pred = reg.predict(z_eval)

    r2_per_factor = np.array([
        r2_score(s_eval[:, j], s_pred[:, j]) for j in range(s.shape[1])
    ])
    mse_per_factor = np.array([
        float(np.mean((s_eval[:, j] - s_pred[:, j]) ** 2)) for j in range(s.shape[1])
    ])
    pearson_r_per_factor = np.array([
        float(scipy_stats.pearsonr(s_eval[:, j], s_pred[:, j])[0]) for j in range(s.shape[1])
    ])

    return {
        "r2_per_factor": r2_per_factor,
        "r2_mean": float(r2_per_factor.mean()),
        "mse_per_factor": mse_per_factor,
        "mse_mean": float(mse_per_factor.mean()),
        "pearson_r_per_factor": pearson_r_per_factor,
        "pearson_r_mean": float(pearson_r_per_factor.mean()),
        "coefficients": reg.coef_,
        "intercept": reg.intercept_,
    }


# ===========================================================================
#                  3.  Nonlinear Identifiability (MLP R²)
# ===========================================================================

def nonlinear_identifiability(
    z: np.ndarray,
    s: np.ndarray,
    hidden_sizes: tuple = (256, 256),
    max_iter: int = 2000,
    train_frac: float = 1.0,
    seed: int = 42,
) -> dict:
    """Measure identifiability up to element-wise nonlinearities.

    Fits an MLP from z → s and reports R². If the representation is
    identifiable up to element-wise nonlinearities (the weakest guarantee
    from nonlinear ICA), this should be ~1.0 even when affine R² is lower.

    The gap between affine R² and nonlinear R² tells us *how nonlinear*
    the residual equivalence class is.

    Args:
        z: (N, D_z) learned latent representations
        s: (N, D_s) ground-truth state factors
        train_frac: fraction of data for training the probe. When < 1.0,
            metrics are reported on the held-out test set (paper-style).
        seed: random seed for train/test split

    Returns:
        dict with 'r2_per_factor', 'r2_mean'
    """
    z = _to_numpy(z)
    s = _to_numpy(s)

    if train_frac < 1.0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(z))
        n_train = int(len(z) * train_frac)
        z_train, z_eval = z[idx[:n_train]], z[idx[n_train:]]
        s_train, s_eval = s[idx[:n_train]], s[idx[n_train:]]
    else:
        z_train, z_eval = z, z
        s_train, s_eval = s, s

    scaler_z = StandardScaler().fit(z_train)
    scaler_s = StandardScaler().fit(s_train)
    z_train_sc = scaler_z.transform(z_train)
    z_eval_sc = scaler_z.transform(z_eval)
    s_train_sc = scaler_s.transform(s_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    mlp.fit(z_train_sc, s_train_sc)
    s_pred_sc = mlp.predict(z_eval_sc)
    s_pred = scaler_s.inverse_transform(s_pred_sc)

    r2_per_factor = np.array([
        r2_score(s_eval[:, j], s_pred[:, j]) for j in range(s.shape[1])
    ])
    mse_per_factor = np.array([
        float(np.mean((s_eval[:, j] - s_pred[:, j]) ** 2)) for j in range(s.shape[1])
    ])
    pearson_r_per_factor = np.array([
        float(scipy_stats.pearsonr(s_eval[:, j], s_pred[:, j])[0]) for j in range(s.shape[1])
    ])

    return {
        "r2_per_factor": r2_per_factor,
        "r2_mean": float(r2_per_factor.mean()),
        "mse_per_factor": mse_per_factor,
        "mse_mean": float(mse_per_factor.mean()),
        "pearson_r_per_factor": pearson_r_per_factor,
        "pearson_r_mean": float(pearson_r_per_factor.mean()),
    }


# ===========================================================================
#                    5.  Action Diversity Metrics
# ===========================================================================

def action_diversity(
    actions: np.ndarray,
    n_bins: int = 50,
) -> dict:
    """Quantify the diversity of actions in a dataset.

    Key theoretical variable: identifiability guarantees require that the
    action distribution has sufficient support / diversity. We measure:

    1. Effective rank of action covariance (nuclear norm / spectral norm):
       how many independent directions actions span.
    2. Entropy of discretized action distribution:
       how uniformly actions cover the action space.
    3. Volume coverage: fraction of the action space that is visited
       (estimated via uniform grid binning).

    Args:
        actions: (N, T, A) or (N, A) action sequences

    Returns:
        dict with diversity metrics
    """
    actions = _to_numpy(actions)

    # Flatten to (N_total, A)
    if actions.ndim == 3:
        actions = actions.reshape(-1, actions.shape[-1])
    elif actions.ndim == 1:
        actions = actions.reshape(-1, 1)

    A = actions.shape[1]

    # --- Effective rank of covariance ---
    cov = np.cov(actions, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    singular_values = np.linalg.svd(cov, compute_uv=False)
    singular_values = singular_values[singular_values > 1e-10]
    nuclear_norm = singular_values.sum()
    spectral_norm = singular_values[0] if len(singular_values) > 0 else 1e-10
    effective_rank = nuclear_norm / spectral_norm if spectral_norm > 0 else 0.0

    # --- Entropy of marginal distributions ---
    marginal_entropies = []
    for d in range(A):
        col = actions[:, d]
        hist, _ = np.histogram(col, bins=n_bins, density=True)
        hist = hist / (hist.sum() + 1e-10)
        h = -np.sum(hist * np.log(hist + 1e-10))
        marginal_entropies.append(h)
    mean_entropy = float(np.mean(marginal_entropies))
    max_entropy = float(np.log(n_bins))
    normalized_entropy = mean_entropy / max_entropy if max_entropy > 0 else 0.0

    # --- Per-step diversity (for sequential data) ---
    action_std = float(np.std(actions, axis=0).mean())

    return {
        "effective_rank": float(effective_rank),
        "normalized_entropy": normalized_entropy,
        "mean_entropy": mean_entropy,
        "action_std": action_std,
        "action_dim": A,
        "num_samples": actions.shape[0],
    }


# ===========================================================================
#        6.  Temporal Contrastivity Score (Action-Conditional)
# ===========================================================================

def temporal_contrastivity(
    z: np.ndarray,
    actions: np.ndarray,
    n_samples: int = 5000,
) -> dict:
    """Measure whether action conditioning creates non-stationary structure.

    This is the empirical test of the key identifiability assumption:
    p(z_{t+1} | z_t, a_t) must vary with a_t. If the transition is
    independent of actions, identifiability via action conditioning fails.

    We estimate this by:
    1. Binning actions into clusters
    2. For each cluster, computing the conditional transition distribution
    3. Measuring the KL divergence between conditional distributions

    Returns:
        dict with 'mean_transition_divergence', 'action_dependence_score'
    """
    z = _to_numpy(z)
    actions = _to_numpy(actions)

    if z.ndim == 3:
        # (B, T, D) → transitions
        z_t = z[:, :-1].reshape(-1, z.shape[-1])
        z_tp1 = z[:, 1:].reshape(-1, z.shape[-1])
        acts = actions[:, :-1].reshape(-1, actions.shape[-1])
    elif z.ndim == 2:
        z_t = z[:-1]
        z_tp1 = z[1:]
        acts = actions[:-1] if actions.ndim == 2 else actions[:-1].reshape(-1, 1)
    else:
        raise ValueError(f"Expected z with 2 or 3 dims, got {z.ndim}")

    # Subsample for efficiency
    n = min(n_samples, len(z_t))
    idx = np.random.choice(len(z_t), n, replace=False)
    z_t, z_tp1, acts = z_t[idx], z_tp1[idx], acts[idx]

    # Compute residuals: delta_z = z_{t+1} - z_t
    delta_z = z_tp1 - z_t

    # Bin actions using k-means-style quantization (simple: median split per dim)
    n_action_bins = min(8, max(2, int(np.sqrt(n / 50))))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_action_bins, random_state=42, n_init=3)
    action_labels = kmeans.fit_predict(acts)

    # Compute mean and variance of delta_z per action cluster
    cluster_means = []
    cluster_vars = []
    for c in range(n_action_bins):
        mask = action_labels == c
        if mask.sum() < 5:
            continue
        cluster_means.append(delta_z[mask].mean(axis=0))
        cluster_vars.append(delta_z[mask].var(axis=0).mean())

    if len(cluster_means) < 2:
        return {
            "mean_transition_divergence": 0.0,
            "action_dependence_score": 0.0,
        }

    cluster_means = np.array(cluster_means)

    # Action dependence = variance of cluster means / overall variance
    overall_var = delta_z.var(axis=0).mean()
    between_var = cluster_means.var(axis=0).mean()
    action_dependence = between_var / (overall_var + 1e-10)

    # Pairwise L2 between cluster means (normalized)
    n_clusters = len(cluster_means)
    pairwise_dists = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            pairwise_dists.append(np.linalg.norm(cluster_means[i] - cluster_means[j]))
    mean_divergence = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0

    return {
        "mean_transition_divergence": mean_divergence,
        "action_dependence_score": float(action_dependence),
    }


# ===========================================================================
#              7.  Equivariance Error (Encoder-Dynamics Consistency)
# ===========================================================================

def equivariance_error(
    z_t: np.ndarray,
    z_tp1: np.ndarray,
    z_tp1_pred: np.ndarray,
) -> dict:
    """Measure the equivariance error of the encoder-predictor pair.

    For identifiability, we need the diagram to commute:
        o_t  →(f)→  o_{t+1}        (true dynamics in observation space)
        ↓ enc        ↓ enc
        z_t  →(pred)→ z_{t+1}       (predicted dynamics in latent space)

    The equivariance error is ||enc(o_{t+1}) - pred(enc(o_t), a_t)||².
    Low error is necessary (but not sufficient) for identifiability.

    We also decompose the error into systematic bias and variance components.

    Args:
        z_t: (N, D) encoded current states
        z_tp1: (N, D) encoded next states (ground truth in latent space)
        z_tp1_pred: (N, D) predicted next states

    Returns:
        dict with 'equivariance_mse', 'equivariance_bias', 'equivariance_var'
    """
    z_tp1 = _to_numpy(z_tp1)
    z_tp1_pred = _to_numpy(z_tp1_pred)

    residual = z_tp1 - z_tp1_pred
    mse = float(np.mean(residual ** 2))
    bias = float(np.mean(residual, axis=0).__pow__(2).mean())
    var = float(np.var(residual, axis=0).mean())

    return {
        "equivariance_mse": mse,
        "equivariance_bias": bias,
        "equivariance_variance": var,
    }


# ===========================================================================
#   8.  Action Invertibility  — can we recover a_t from (z_t, z_{t+1})?
# ===========================================================================

def action_invertibility(
    z: np.ndarray,
    actions: np.ndarray,
) -> dict:
    """Test whether actions are linearly recoverable from latent transitions.

    This is the *reverse* of action conditioning: if knowing the action
    improves prediction (iVAE non-stationarity condition), then the action
    should be decodable from the latent transition Δz = z_{t+1} - z_t.

    Fits a Ridge regression  Δz → a_t  and reports R² per action dimension.

    Args:
        z:       (N, T, D)  or  (N*T, D)  latent representations
        actions: (N, T, A)  or  (N*T, A)  corresponding actions

    Returns:
        dict with 'r2_per_dim', 'r2_mean', 'r2_mean_raw_dims'
        (raw_dims = the 2 original action dims before frameskip concat)
    """
    z = _to_numpy(z)
    actions = _to_numpy(actions)

    if z.ndim == 3:
        # Transitions: consecutive frames within each trajectory
        delta_z = (z[:, 1:] - z[:, :-1]).reshape(-1, z.shape[-1])   # (N*(T-1), D)
        act_flat = actions[:, :-1].reshape(-1, actions.shape[-1])    # (N*(T-1), A)
    else:
        delta_z = np.diff(z, axis=0)
        act_flat = actions[:-1]

    scaler_dz = StandardScaler().fit(delta_z)
    scaler_a  = StandardScaler().fit(act_flat)
    dz_sc = scaler_dz.transform(delta_z)
    a_sc  = scaler_a.transform(act_flat)

    reg = Ridge(alpha=1e-3)
    reg.fit(dz_sc, a_sc)
    a_pred = reg.predict(dz_sc)

    r2_per_dim = np.array([
        r2_score(a_sc[:, j], a_pred[:, j]) for j in range(a_sc.shape[1])
    ])

    # Raw 2D action dims: for EFFECTIVE_ACT_DIM = frameskip × 2, reshape
    A = act_flat.shape[1]
    if A > 2 and A % 2 == 0:
        # Each pair of dims corresponds to one raw timestep
        r2_raw = r2_per_dim.reshape(-1, 2).mean(axis=1)  # mean over x/y per step
    else:
        r2_raw = r2_per_dim

    return {
        "r2_per_dim": r2_per_dim,
        "r2_mean": float(r2_per_dim.mean()),
        "r2_mean_raw_dims": r2_raw,
    }


# ===========================================================================
#   9.  Probe Generalization  — train/test episode split
# ===========================================================================

def probe_generalization(
    z: np.ndarray,
    s: np.ndarray,
    train_frac: float = 0.8,
    seed: int = 42,
) -> dict:
    """Measure linear probe R² on held-out episodes.

    The existing affine_identifiability() fits and evaluates on the same data,
    which can inflate R² for a locally-linear but globally inconsistent code.
    This function trains the probe on the first `train_frac` episodes and
    evaluates on the remaining episodes.

    Args:
        z: (N_eps, T, D)  — N_eps episodes, T timesteps, D latent dims
        s: (N_eps, T, 7)  — ground-truth state factors
        train_frac: fraction of episodes used for training the probe

    Returns:
        dict with 'train_r2_per_factor', 'test_r2_per_factor',
                  'train_r2_mean', 'test_r2_mean', 'generalization_gap'
    """
    z = _to_numpy(z)
    s = _to_numpy(s)

    N = z.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    z_train = z[train_idx].reshape(-1, z.shape[-1])
    s_train = s[train_idx].reshape(-1, s.shape[-1])
    z_test  = z[test_idx].reshape(-1, z.shape[-1])
    s_test  = s[test_idx].reshape(-1, s.shape[-1])

    reg = Ridge(alpha=1e-4)
    reg.fit(z_train, s_train)

    s_pred_train = reg.predict(z_train)
    s_pred_test  = reg.predict(z_test)

    train_r2 = np.array([
        r2_score(s_train[:, j], s_pred_train[:, j]) for j in range(s.shape[-1])
    ])
    test_r2 = np.array([
        r2_score(s_test[:, j], s_pred_test[:, j]) for j in range(s.shape[-1])
    ])

    return {
        "train_r2_per_factor": train_r2,
        "test_r2_per_factor":  test_r2,
        "train_r2_mean":       float(train_r2.mean()),
        "test_r2_mean":        float(test_r2.mean()),
        "generalization_gap":  float(train_r2.mean() - test_r2.mean()),
    }


# ===========================================================================
#   10.  DCI Metrics  (Eastwood & Williams 2018)
# ===========================================================================

def dci_metrics(
    z: np.ndarray,
    s: np.ndarray,
    lasso_alpha: float = 0.02,
) -> dict:
    """Compute Disentanglement, Completeness, Informativeness (DCI).

    References:
        Eastwood & Williams (2018), "A Framework for the Quantitative
        Evaluation of Disentangled Representations", ICLR 2018.

    Procedure:
        1. For each ground-truth factor s_j, fit Lasso(z → s_j).
           Importance matrix R_{ij} = |coeff from z_i predicting s_j|.
        2. Informativeness I_j: R² of the Lasso fit for factor j.
        3. Disentanglement D_i: For each latent z_i, how concentrated are
           its contributions across factors?
               P_{ij} = R_{ij} / sum_j R_{ij}
               D_i = 1 - H(P_{i,*}) / log(n_factors)   (normalized entropy)
           Weighted mean over z_i by sum_j R_{ij} (how active z_i is).
        4. Completeness C_j: For each factor s_j, how concentrated are
           the latent contributions?
               Q_{ij} = R_{ij} / sum_i R_{ij}
               C_j = 1 - H(Q_{*,j}) / log(n_latent)   (normalized entropy)

    Args:
        z: (N, D_z)
        s: (N, D_s)
        lasso_alpha: Lasso regularization. Higher → sparser importances.

    Returns:
        dict with 'disentanglement', 'completeness', 'informativeness_per_factor',
                  'importance_matrix', 'dci_score' (harmonic mean of D and C means)
    """
    z = _to_numpy(z)
    s = _to_numpy(s)

    scaler_z = StandardScaler().fit(z)
    z_sc = scaler_z.transform(z)

    D_s = s.shape[1]
    D_z = z.shape[1]

    # Importance matrix R: (D_z, D_s)
    R = np.zeros((D_z, D_s))
    informativeness = np.zeros(D_s)

    for j in range(D_s):
        scaler_sj = StandardScaler().fit(s[:, j: j + 1])
        s_sc = scaler_sj.transform(s[:, j: j + 1]).ravel()

        lasso = Lasso(alpha=lasso_alpha, max_iter=5000, random_state=42)
        lasso.fit(z_sc, s_sc)
        s_pred = lasso.predict(z_sc)

        R[:, j] = np.abs(lasso.coef_)
        informativeness[j] = r2_score(s_sc, s_pred)

    # ----- Disentanglement -----
    row_sums = R.sum(axis=1, keepdims=True) + 1e-10   # (D_z, 1)
    P = R / row_sums                                   # (D_z, D_s) — prob dist over factors per z_i

    def norm_entropy(p_row):
        """Normalized entropy of a probability row (eps-clipped)."""
        p = np.clip(p_row, 1e-10, 1.0)
        p = p / p.sum()
        h = -np.sum(p * np.log(p))
        return h / np.log(D_s) if D_s > 1 else 0.0

    D_i = np.array([1.0 - norm_entropy(P[i]) for i in range(D_z)])
    # Weight by total relevance of each z_i
    weights = R.sum(axis=1)
    if weights.sum() > 1e-10:
        disentanglement = float(np.average(D_i, weights=weights))
    else:
        disentanglement = 0.0

    # ----- Completeness -----
    col_sums = R.sum(axis=0, keepdims=True) + 1e-10   # (1, D_s)
    Q = R / col_sums                                   # (D_z, D_s) — prob dist over z_i per factor j

    def norm_entropy_col(p_col):
        p = np.clip(p_col, 1e-10, 1.0)
        p = p / p.sum()
        h = -np.sum(p * np.log(p))
        return h / np.log(D_z) if D_z > 1 else 0.0

    C_j = np.array([1.0 - norm_entropy_col(Q[:, j]) for j in range(D_s)])
    completeness = float(C_j.mean())

    # DCI score: harmonic mean of mean D and mean C
    mean_D = float(D_i.mean())
    mean_C = completeness
    if mean_D + mean_C > 1e-10:
        dci_score = 2 * mean_D * mean_C / (mean_D + mean_C)
    else:
        dci_score = 0.0

    return {
        "disentanglement_per_latent": D_i,
        "completeness_per_factor":    C_j,
        "informativeness_per_factor": informativeness,
        "importance_matrix":          R,
        "disentanglement_mean":       mean_D,
        "completeness_mean":          mean_C,
        "informativeness_mean":       float(informativeness.mean()),
        "dci_score":                  dci_score,
    }


# ===========================================================================
#                         Utility functions
# ===========================================================================

def _to_numpy(x):
    """Convert tensor to numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)

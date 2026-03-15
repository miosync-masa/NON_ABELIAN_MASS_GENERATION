import numpy as np
import matplotlib.pyplot as plt

print("=" * 88)
print("FAST FINITE-SIZE FALSIFICATION SUITE")
print("Naive continuum-scaling diagnostics for a controlled SU(2) two-lump family")
print("=" * 88)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

BETA = 2.0
ROTATION_AXIS = "y"
THETA = np.pi / 4.0

# 重いときは [4,5,6,7,8] まで落とす
L_VALUES = np.array([4, 5, 6, 7, 8, 10, 12], dtype=np.int32)

# =============================================================================
# SU(2) BASICS
# =============================================================================

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
I2 = np.eye(2, dtype=np.complex64)


def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c]
    ], dtype=np.float32)


def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float32)


def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


def get_rotation_matrix(theta, axis="z"):
    if axis == "x":
        return rotation_matrix_x(theta)
    if axis == "y":
        return rotation_matrix_y(theta)
    return rotation_matrix_z(theta)


def rotate_color_field(omega, R):
    # omega: (..., 3)
    return np.einsum("...a,ab->...b", omega, R).astype(np.float32)


def abelianize_projection(omega, axis=2):
    out = np.zeros_like(omega)
    out[..., axis] = omega[..., axis]
    return out


# =============================================================================
# VECTORIAL FIELD CONSTRUCTION
# =============================================================================

def periodic_displacement_array(coords, center, L):
    return ((coords - center + L / 2.0) % L) - L / 2.0


def create_instanton_config(L, center, rho, charge=1):
    """
    BPST-like controlled lattice ansatz, vectorized.
    Output shape: (L,L,L,L,4,3)
    """
    L = int(L)
    cx, cy, cz, ct = center

    grid = np.arange(L, dtype=np.float32)
    X, Y, Z, T = np.meshgrid(grid, grid, grid, grid, indexing="ij")

    dx = periodic_displacement_array(X, cx, L)
    dy = periodic_displacement_array(Y, cy, L)
    dz = periodic_displacement_array(Z, cz, L)
    dt = periodic_displacement_array(T, ct, L)

    r = np.stack([dx, dy, dz, dt], axis=-1).astype(np.float32)  # (...,4)
    r2 = np.sum(r**2, axis=-1)

    eta = np.zeros((3, 4, 4), dtype=np.float32)
    eta[0, 0, 1] = 1; eta[0, 1, 0] = -1
    eta[0, 2, 3] = 1; eta[0, 3, 2] = -1
    eta[1, 0, 2] = 1; eta[1, 2, 0] = -1
    eta[1, 3, 1] = 1; eta[1, 1, 3] = -1
    eta[2, 0, 3] = 1; eta[2, 3, 0] = -1
    eta[2, 1, 2] = 1; eta[2, 2, 1] = -1

    if charge == -1:
        eta = -eta

    base = np.einsum("amn,...n->...ma", eta, r).astype(np.float32)  # (...,4,3)
    factor = (2.0 * rho**2 / (r2 + rho**2 + 1e-6) / (rho**2 + 0.1)).astype(np.float32)
    omega = base * factor[..., None, None]
    return omega.astype(np.float32)


def create_two_lump_components(L, sep, rho=0.8, theta=0.0, rotation_axis="z"):
    """
    Topologically trivial two-lump family with continuous centers.
    """
    L = int(L)
    center = L / 2.0
    center1 = (center - sep / 2.0, center, center, center)
    center2 = (center + sep / 2.0, center, center, center)

    omega1 = create_instanton_config(L, center1, rho, charge=+1)
    omega2 = create_instanton_config(L, center2, rho, charge=-1)

    R = get_rotation_matrix(theta, rotation_axis)
    omega2_rot = rotate_color_field(omega2, R)
    return omega1, omega2_rot


def create_two_lump_family(L, sep, rho=0.8, theta=0.0, rotation_axis="z", abelianized=False):
    omega1, omega2 = create_two_lump_components(L, sep, rho, theta, rotation_axis)
    if abelianized:
        omega1 = abelianize_projection(omega1, axis=2)
        omega2 = abelianize_projection(omega2, axis=2)
    return omega1 + omega2, omega1, omega2


# =============================================================================
# FAST SU(2) MATRIX FIELDS
# =============================================================================

def build_link_field(omega):
    """
    Vectorized SU(2) exponential.
    omega shape: (...,3)
    U shape: (...,2,2)
    """
    w0 = omega[..., 0]
    w1 = omega[..., 1]
    w2 = omega[..., 2]

    norm = np.sqrt(w0*w0 + w1*w1 + w2*w2 + 1e-12).astype(np.float32)
    n0 = w0 / norm
    n1 = w1 / norm
    n2 = w2 / norm

    cos_half = np.cos(norm / 2.0).astype(np.complex64)
    sin_half = np.sin(norm / 2.0).astype(np.complex64)

    U = np.empty(omega.shape[:-1] + (2, 2), dtype=np.complex64)

    # n·sigma = [[n2, n0 - i n1], [n0 + i n1, -n2]]
    U[..., 0, 0] = cos_half + 1j * sin_half * n2
    U[..., 0, 1] = 1j * sin_half * (n0 - 1j * n1)
    U[..., 1, 0] = 1j * sin_half * (n0 + 1j * n1)
    U[..., 1, 1] = cos_half - 1j * sin_half * n2
    return U


def build_alg_field(omega):
    """
    Vectorized su(2) algebra matrix:
      A = i/2 * (omega · sigma)
    """
    w0 = omega[..., 0].astype(np.complex64)
    w1 = omega[..., 1].astype(np.complex64)
    w2 = omega[..., 2].astype(np.complex64)

    A = np.empty(omega.shape[:-1] + (2, 2), dtype=np.complex64)
    A[..., 0, 0] = 0.5j * w2
    A[..., 0, 1] = 0.5j * (w0 - 1j * w1)
    A[..., 1, 0] = 0.5j * (w0 + 1j * w1)
    A[..., 1, 1] = -0.5j * w2
    return A


# =============================================================================
# FAST LATTICE OBSERVABLES
# =============================================================================

def compute_wilson_action_and_vorticity_from_links(U, beta=2.0):
    """
    Fully vectorized over lattice sites.
    U shape: (L,L,L,L,4,2,2)
    """
    L = U.shape[0]
    S_total = 0.0
    V_total = 0.0

    for mu in range(4):
        for nu in range(mu + 1, 4):
            U_mu = U[..., mu, :, :]
            U_nu = U[..., nu, :, :]

            # x+mu and x+nu shifts
            U_nu_shift_mu = np.roll(U_nu, shift=-1, axis=mu)
            U_mu_shift_nu = np.roll(U_mu, shift=-1, axis=nu)

            P = np.matmul(
                np.matmul(U_mu, U_nu_shift_mu),
                np.matmul(np.swapaxes(U_mu_shift_nu.conj(), -1, -2),
                          np.swapaxes(U_nu.conj(), -1, -2))
            )

            tr_half = np.real(P[..., 0, 0] + P[..., 1, 1]) / 2.0
            S_total += np.sum(1.0 - tr_half)

            diff = P - I2
            V_total += np.sum(np.abs(diff) ** 2)

    n_plaquettes = (L ** 4) * 6
    return beta * float(S_total), float(V_total), n_plaquettes


def commutator_overlap_from_alg(A1, A2):
    """
    Vectorized over lattice sites, loops only over the 6 (mu,nu) pairs.
    """
    total = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            A1_mu = A1[..., mu, :, :]
            A1_nu = A1[..., nu, :, :]
            A2_mu = A2[..., mu, :, :]
            A2_nu = A2[..., nu, :, :]

            comm1 = np.matmul(A1_mu, A2_nu) - np.matmul(A2_nu, A1_mu)
            comm2 = np.matmul(A1_nu, A2_mu) - np.matmul(A2_mu, A1_nu)
            diff = comm1 - comm2
            total += np.sum(np.abs(diff) ** 2)

    return float(total)


# =============================================================================
# SELF-TERM CACHES
# =============================================================================

_nonab_self_cache = {}
_ab_self_cache_unrot = {}
_ab_self_cache_rot = {}


def get_self_v_nonab(L, rho, beta=2.0):
    """
    Self-vorticity of a single non-Abelian lump.
    Independent of sep and theta in this construction.
    """
    key = (int(L), float(rho), float(beta))
    if key not in _nonab_self_cache:
        center = (L / 2.0, L / 2.0, L / 2.0, L / 2.0)
        omega = create_instanton_config(L, center, rho, charge=+1)
        U = build_link_field(omega)
        _, V, _ = compute_wilson_action_and_vorticity_from_links(U, beta=beta)
        _nonab_self_cache[key] = V
    return _nonab_self_cache[key]


def get_self_v_ab_unrot(L, rho, beta=2.0):
    """
    Self-vorticity of one Abelianized lump without relative color rotation.
    """
    key = (int(L), float(rho), float(beta))
    if key not in _ab_self_cache_unrot:
        center = (L / 2.0, L / 2.0, L / 2.0, L / 2.0)
        omega = create_instanton_config(L, center, rho, charge=+1)
        omega_ab = abelianize_projection(omega, axis=2)
        U = build_link_field(omega_ab)
        _, V, _ = compute_wilson_action_and_vorticity_from_links(U, beta=beta)
        _ab_self_cache_unrot[key] = V
    return _ab_self_cache_unrot[key]


def get_self_v_ab_rot(L, rho, theta, rotation_axis="z", beta=2.0):
    """
    Self-vorticity of the rotated-and-then-Abelianized second lump.
    Depends on theta and rotation axis.
    """
    key = (int(L), float(rho), float(theta), rotation_axis, float(beta))
    if key not in _ab_self_cache_rot:
        center = (L / 2.0, L / 2.0, L / 2.0, L / 2.0)
        omega = create_instanton_config(L, center, rho, charge=+1)
        R = get_rotation_matrix(theta, rotation_axis)
        omega_rot = rotate_color_field(omega, R)
        omega_ab = abelianize_projection(omega_rot, axis=2)
        U = build_link_field(omega_ab)
        _, V, _ = compute_wilson_action_and_vorticity_from_links(U, beta=beta)
        _ab_self_cache_rot[key] = V
    return _ab_self_cache_rot[key]


# =============================================================================
# FAST FAMILY METRICS
# =============================================================================

def compute_family_metrics_fast(
    L, sep, rho, theta, rotation_axis="z", beta=2.0, compute_overlap=False
):
    """
    Main fast evaluator.
    """
    L = int(L)

    # ---------- non-Abelian total ----------
    omega_tot_na, omega1_na, omega2_na = create_two_lump_family(
        L=L, sep=sep, rho=rho, theta=theta, rotation_axis=rotation_axis, abelianized=False
    )
    U_tot_na = build_link_field(omega_tot_na)
    _, V_nonAb, nplaq = compute_wilson_action_and_vorticity_from_links(U_tot_na, beta=beta)

    V_self_nonAb = get_self_v_nonab(L, rho, beta=beta)
    DeltaV_int_nonAb = V_nonAb - 2.0 * V_self_nonAb

    # ---------- Abelianized total ----------
    omega_tot_ab, omega1_ab, omega2_ab = create_two_lump_family(
        L=L, sep=sep, rho=rho, theta=theta, rotation_axis=rotation_axis, abelianized=True
    )
    U_tot_ab = build_link_field(omega_tot_ab)
    _, V_Ab, _ = compute_wilson_action_and_vorticity_from_links(U_tot_ab, beta=beta)

    V_self_ab_1 = get_self_v_ab_unrot(L, rho, beta=beta)
    V_self_ab_2 = get_self_v_ab_rot(L, rho, theta, rotation_axis=rotation_axis, beta=beta)
    DeltaV_int_Ab = V_Ab - V_self_ab_1 - V_self_ab_2

    # ---------- excess channels ----------
    DeltaV_NA = V_nonAb - V_Ab
    DeltaDeltaV_int = DeltaV_int_nonAb - DeltaV_int_Ab

    # ---------- optional overlap ----------
    O_nonAb = 0.0
    O_Ab = 0.0
    if compute_overlap:
        A1_na = build_alg_field(omega1_na)
        A2_na = build_alg_field(omega2_na)
        O_nonAb = commutator_overlap_from_alg(A1_na, A2_na)
        # Abelianized overlap is identically zero by construction
        O_Ab = 0.0

    eps = 1e-12
    return {
        "L": float(L),
        "sep": float(sep),
        "rho": float(rho),
        "theta": float(theta),
        "Np": float(nplaq),
        "V_nonAb": float(V_nonAb),
        "V_Ab": float(V_Ab),
        "O_nonAb": float(O_nonAb),
        "O_Ab": float(O_Ab),
        "DeltaV_NA": float(DeltaV_NA),
        "DeltaV_int_nonAb": float(DeltaV_int_nonAb),
        "DeltaV_int_Ab": float(DeltaV_int_Ab),
        "DeltaDeltaV_int": float(DeltaDeltaV_int),
        "R_NA": float(DeltaV_NA / max(abs(V_Ab), eps)),
        "R_int": float(DeltaDeltaV_int / max(abs(DeltaV_int_Ab), eps)),
        "d_NA": float(DeltaV_NA / max(float(nplaq), eps)),
        "d_int": float(DeltaDeltaV_int / max(float(nplaq), eps)),
    }


# =============================================================================
# SCALING PATHS
# =============================================================================

def path_fixed_units(L):
    """
    Localized object in a larger box.
    """
    return {"sep": 2.0, "rho": 0.8}


def path_fixed_fractions(L):
    """
    Geometric self-similarity.
    """
    return {"sep": L / 3.0, "rho": 0.13 * L}


def build_dataset(L_values, theta, rotation_axis, beta, path_mode, compute_overlap=False):
    rows = []
    for L_raw in L_values:
        L = int(L_raw)

        if path_mode == "fixed_units":
            params = path_fixed_units(L)
        elif path_mode == "fixed_fractions":
            params = path_fixed_fractions(L)
        else:
            raise ValueError("unknown path_mode")

        row = compute_family_metrics_fast(
            L=L,
            sep=params["sep"],
            rho=params["rho"],
            theta=theta,
            rotation_axis=rotation_axis,
            beta=beta,
            compute_overlap=compute_overlap,
        )
        row["path_mode"] = path_mode
        rows.append(row)

    return rows


# =============================================================================
# FIT / DIAGNOSTICS
# =============================================================================

def fit_linear_basis(X, y, label):
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coeff
    resid = y - yhat
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    r2 = 1.0 - rss / tss
    n = len(y)
    k = X.shape[1]
    bic = n * np.log(rss / max(n, 1) + 1e-12) + k * np.log(max(n, 2))
    return {
        "label": label,
        "coeff": coeff,
        "yhat": yhat,
        "rss": rss,
        "r2": r2,
        "bic": bic,
    }


def fit_power_growth(L, y):
    if np.any(y <= 0):
        return None
    lx = np.log(L)
    ly = np.log(y)
    A = np.column_stack([np.ones_like(lx), lx])
    coeff, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    logB, p = coeff
    yhat = np.exp(logB) * (L ** p)
    resid = y - yhat
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    r2 = 1.0 - rss / tss
    n = len(y)
    k = 2
    bic = n * np.log(rss / max(n, 1) + 1e-12) + k * np.log(max(n, 2))
    return {
        "label": "B*L^p",
        "coeff": np.array([np.exp(logB), p]),
        "yhat": yhat,
        "rss": rss,
        "r2": r2,
        "bic": bic,
    }


def scaling_fits(L, y):
    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    fits = []
    fits.append(fit_linear_basis(
        np.column_stack([np.ones_like(L)]), y, "A"
    ))
    fits.append(fit_linear_basis(
        np.column_stack([np.ones_like(L), 1.0 / L]), y, "A + B/L"
    ))
    fits.append(fit_linear_basis(
        np.column_stack([np.ones_like(L), 1.0 / (L ** 2)]), y, "A + B/L^2"
    ))
    fits.append(fit_linear_basis(
        np.column_stack([np.ones_like(L), 1.0 / np.log(L + 1e-12)]), y, "A + B/logL"
    ))

    power_fit = fit_power_growth(L, y)
    if power_fit is not None:
        fits.append(power_fit)

    fits = sorted(fits, key=lambda d: d["bic"])
    return fits


def plateau_score(y):
    y = np.asarray(y, dtype=np.float64)
    tail = y[-3:] if len(y) >= 3 else y
    return float(np.std(tail) / (abs(np.mean(tail)) + 1e-12))


def monotone_trend(y):
    dy = np.diff(y)
    if np.all(dy >= 0):
        return "monotone_increasing"
    if np.all(dy <= 0):
        return "monotone_decreasing"
    return "mixed"


def finite_limit_support_score(L, y):
    fits = scaling_fits(L, y)
    finite_candidates = [f for f in fits if f["label"] in {"A", "A + B/L", "A + B/L^2", "A + B/logL"}]
    best_finite = min(finite_candidates, key=lambda d: d["bic"])
    best_all = fits[0]
    pscore = plateau_score(y)
    trend = monotone_trend(y)

    penalty = 0.0
    if best_all["label"] == "B*L^p":
        penalty += 1.0
    if trend.startswith("monotone_"):
        penalty += 0.5
    penalty += min(pscore * 5.0, 3.0)

    return {
        "best_all": best_all,
        "best_finite": best_finite,
        "plateau_score": pscore,
        "trend": trend,
        "suspicion_score": penalty,
    }


# =============================================================================
# RUN DATASETS
# =============================================================================

dataset_fixed_units = build_dataset(
    L_VALUES, THETA, ROTATION_AXIS, BETA, "fixed_units", compute_overlap=False
)
dataset_fixed_fractions = build_dataset(
    L_VALUES, THETA, ROTATION_AXIS, BETA, "fixed_fractions", compute_overlap=False
)

observables = [
    ("DeltaV_NA", "DeltaV_NA"),
    ("DeltaDeltaV_int", "DeltaDeltaV_int"),
    ("R_NA", "DeltaV_NA / V_Ab"),
    ("R_int", "DeltaDeltaV_int / |DeltaV_int_Ab|"),
    ("d_NA", "DeltaV_NA / Np"),
    ("d_int", "DeltaDeltaV_int / Np"),
]

# =============================================================================
# PRINT RAW TABLES
# =============================================================================

def print_dataset(rows, name):
    print("\n" + "=" * 88)
    print(f"DATASET: {name}")
    print("=" * 88)
    print(
        f"{'L':>4} {'sep':>8} {'rho':>8} "
        f"{'DV_NA':>12} {'DDVint':>12} {'R_NA':>12} {'R_int':>12} {'d_NA':>12} {'d_int':>12}"
    )
    print("-" * 104)
    for r in rows:
        print(
            f"{int(r['L']):4d} {r['sep']:8.3f} {r['rho']:8.3f} "
            f"{r['DeltaV_NA']:12.6f} {r['DeltaDeltaV_int']:12.6f} "
            f"{r['R_NA']:12.6f} {r['R_int']:12.6f} "
            f"{r['d_NA']:12.6f} {r['d_int']:12.6f}"
        )


print_dataset(dataset_fixed_units, "fixed lattice units")
print_dataset(dataset_fixed_fractions, "fixed fractions")

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_dataset(rows, name):
    print("\n" + "=" * 88)
    print(f"FALSIFICATION ANALYSIS: {name}")
    print("=" * 88)

    L = np.array([r["L"] for r in rows], dtype=np.float64)
    analysis = {}

    for key, label in observables:
        y = np.array([r[key] for r in rows], dtype=np.float64)
        diag = finite_limit_support_score(L, y)
        analysis[key] = diag

        print(f"\nObservable: {label}")
        print(f"  trend          : {diag['trend']}")
        print(f"  plateau_score  : {diag['plateau_score']:.6f}   (smaller = more plateau-like)")
        print(f"  suspicion_score: {diag['suspicion_score']:.6f} (larger = more anti-plateau)")
        print(f"  best_all_model : {diag['best_all']['label']}")
        print(f"  best_all_BIC   : {diag['best_all']['bic']:.6f}")
        print(f"  best_finite    : {diag['best_finite']['label']}")
        print(f"  best_finite_BIC: {diag['best_finite']['bic']:.6f}")

    return analysis


analysis_units = analyze_dataset(dataset_fixed_units, "fixed lattice units")
analysis_fractions = analyze_dataset(dataset_fixed_fractions, "fixed fractions")

# =============================================================================
# FIGURES
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))


def get_series(rows, key):
    return np.array([r[key] for r in rows], dtype=np.float64)


# 1: DeltaV_NA
ax = axes[0, 0]
ax.plot(L_VALUES, get_series(dataset_fixed_units, "DeltaV_NA"), "o-", label="fixed units")
ax.plot(L_VALUES, get_series(dataset_fixed_fractions, "DeltaV_NA"), "s--", label="fixed fractions")
ax.set_title("DeltaV_NA vs L")
ax.set_xlabel("L")
ax.set_ylabel("DeltaV_NA")
ax.grid(alpha=0.3)
ax.legend()

# 2: DeltaDeltaV_int
ax = axes[0, 1]
ax.plot(L_VALUES, get_series(dataset_fixed_units, "DeltaDeltaV_int"), "o-", label="fixed units")
ax.plot(L_VALUES, get_series(dataset_fixed_fractions, "DeltaDeltaV_int"), "s--", label="fixed fractions")
ax.set_title("DeltaDeltaV_int vs L")
ax.set_xlabel("L")
ax.set_ylabel("DeltaDeltaV_int")
ax.grid(alpha=0.3)
ax.legend()

# 3: R_NA
ax = axes[0, 2]
ax.plot(L_VALUES, get_series(dataset_fixed_units, "R_NA"), "o-", label="fixed units")
ax.plot(L_VALUES, get_series(dataset_fixed_fractions, "R_NA"), "s--", label="fixed fractions")
ax.set_title("R_NA = DeltaV_NA / V_Ab")
ax.set_xlabel("L")
ax.set_ylabel("R_NA")
ax.grid(alpha=0.3)
ax.legend()

# 4: R_int
ax = axes[1, 0]
ax.plot(L_VALUES, get_series(dataset_fixed_units, "R_int"), "o-", label="fixed units")
ax.plot(L_VALUES, get_series(dataset_fixed_fractions, "R_int"), "s--", label="fixed fractions")
ax.set_title("R_int = DeltaDeltaV_int / |DeltaV_int_Ab|")
ax.set_xlabel("L")
ax.set_ylabel("R_int")
ax.grid(alpha=0.3)
ax.legend()

# 5: d_NA
ax = axes[1, 1]
ax.plot(L_VALUES, get_series(dataset_fixed_units, "d_NA"), "o-", label="fixed units")
ax.plot(L_VALUES, get_series(dataset_fixed_fractions, "d_NA"), "s--", label="fixed fractions")
ax.set_title("d_NA = DeltaV_NA / Np")
ax.set_xlabel("L")
ax.set_ylabel("d_NA")
ax.grid(alpha=0.3)
ax.legend()

# 6: d_int
ax = axes[1, 2]
ax.plot(L_VALUES, get_series(dataset_fixed_units, "d_int"), "o-", label="fixed units")
ax.plot(L_VALUES, get_series(dataset_fixed_fractions, "d_int"), "s--", label="fixed fractions")
ax.set_title("d_int = DeltaDeltaV_int / Np")
ax.set_xlabel("L")
ax.set_ylabel("d_int")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
out_path = "fast_finite_size_falsification_suite.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved figure: {out_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

def summary_line(analysis, key, label):
    d = analysis[key]
    return (
        f"- {label}: trend={d['trend']}, "
        f"best_all={d['best_all']['label']}, "
        f"best_finite={d['best_finite']['label']}, "
        f"plateau_score={d['plateau_score']:.4f}, "
        f"suspicion_score={d['suspicion_score']:.4f}"
    )

print("\n" + "=" * 88)
print("SUMMARY")
print("=" * 88)
print("Interpretation rule of thumb:")
print("  - plateau_score small + finite model preferred => naive finite limit is plausible")
print("  - large drift / monotone trend / power-growth preference => naive finite limit is suspicious")

print("\nFixed lattice units:")
for key, label in observables:
    print(summary_line(analysis_units, key, label))

print("\nFixed fractions:")
for key, label in observables:
    print(summary_line(analysis_fractions, key, label))

print("\nCaution:")
print("This code does NOT prove non-existence of a continuum Yang-Mills theory.")
print("It tests whether several natural excess observables in a controlled family")
print("support or fail to support simple finite-size extrapolation scenarios.")

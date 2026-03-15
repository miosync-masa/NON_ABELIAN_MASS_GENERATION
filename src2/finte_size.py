import numpy as np
import matplotlib.pyplot as plt

print("=" * 86)
print("FINITE-SIZE FALSIFICATION SUITE")
print("Naive continuum-scaling diagnostics for a controlled SU(2) two-lump family")
print("=" * 86)

# =============================================================================
# SU(2) BASICS
# =============================================================================

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
I2 = np.eye(2, dtype=np.complex64)

def su2_exp_np(omega):
    omega = np.asarray(omega, dtype=np.float32)
    norm = np.sqrt(np.sum(omega**2) + 1e-12)
    n = omega / norm
    cos_half = np.cos(norm / 2.0)
    sin_half = np.sin(norm / 2.0)
    n_dot_sigma = n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z
    return cos_half * I2 + 1j * sin_half * n_dot_sigma

def su2_alg_matrix(omega):
    return 0.5j * (
        omega[0] * SIGMA_X +
        omega[1] * SIGMA_Y +
        omega[2] * SIGMA_Z
    )

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

def rotate_color_field(omega, R):
    return np.einsum("...a,ab->...b", omega, R).astype(np.float32)

def abelianize_projection(omega, axis=2):
    out = np.zeros_like(omega)
    out[..., axis] = omega[..., axis]
    return out

# =============================================================================
# LATTICE HELPERS
# =============================================================================

def periodic_displacement(coord, center, L):
    return ((coord - center + L / 2.0) % L) - L / 2.0

def compute_plaquette_np(omega, L, x, y, z, t, mu, nu):
    def get_U(ix, iy, iz, it, idir):
        return su2_exp_np(omega[ix % L, iy % L, iz % L, it % L, idir])

    shift_mu = [1 if i == mu else 0 for i in range(4)]
    shift_nu = [1 if i == nu else 0 for i in range(4)]

    x_mu = (x + shift_mu[0]) % L
    y_mu = (y + shift_mu[1]) % L
    z_mu = (z + shift_mu[2]) % L
    t_mu = (t + shift_mu[3]) % L

    x_nu = (x + shift_nu[0]) % L
    y_nu = (y + shift_nu[1]) % L
    z_nu = (z + shift_nu[2]) % L
    t_nu = (t + shift_nu[3]) % L

    U1 = get_U(x, y, z, t, mu)
    U2 = get_U(x_mu, y_mu, z_mu, t_mu, nu)
    U3 = get_U(x_nu, y_nu, z_nu, t_nu, mu).conj().T
    U4 = get_U(x, y, z, t, nu).conj().T
    return U1 @ U2 @ U3 @ U4

def compute_wilson_action_and_vorticity(omega, L, beta=2.0):
    S_total = 0.0
    V_total = 0.0
    n_plaquettes = 0

    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            P = compute_plaquette_np(omega, L, x, y, z, t, mu, nu)
                            tr_half = np.real(np.trace(P)) / 2.0
                            S_total += 1.0 - tr_half
                            V_total += np.sum(np.abs(P - I2) ** 2)
                            n_plaquettes += 1

    return beta * float(S_total), float(V_total), n_plaquettes

# =============================================================================
# FIELD CONSTRUCTION
# =============================================================================
L_values = np.array([4, 5, 6, 7, 8, 10, 12], dtype=np.int32)

def create_instanton_config(L, center, rho, charge=1):
    omega = np.zeros((L, L, L, L, 4, 3), dtype=np.float32)
    cx, cy, cz, ct = center

    eta = np.zeros((3, 4, 4), dtype=np.float32)
    eta[0, 0, 1] = 1; eta[0, 1, 0] = -1
    eta[0, 2, 3] = 1; eta[0, 3, 2] = -1
    eta[1, 0, 2] = 1; eta[1, 2, 0] = -1
    eta[1, 3, 1] = 1; eta[1, 1, 3] = -1
    eta[2, 0, 3] = 1; eta[2, 3, 0] = -1
    eta[2, 1, 2] = 1; eta[2, 2, 1] = -1

    if charge == -1:
        eta = -eta

    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    dx = periodic_displacement(x, cx, L)
                    dy = periodic_displacement(y, cy, L)
                    dz = periodic_displacement(z, cz, L)
                    dt = periodic_displacement(t, ct, L)

                    r = np.array([dx, dy, dz, dt], dtype=np.float32)
                    r2 = np.sum(r ** 2)
                    f = 2.0 * rho ** 2 / (r2 + rho ** 2 + 1e-6)

                    for mu in range(4):
                        for a in range(3):
                            val = 0.0
                            for nu in range(4):
                                val += eta[a, mu, nu] * r[nu]
                            omega[x, y, z, t, mu, a] = val * f / (rho ** 2 + 0.1)

    return omega

def create_two_lump_components(L, sep, rho=0.8, theta=0.0, rotation_axis="z"):
    center = L / 2.0
    center1 = (center - sep / 2.0, center, center, center)
    center2 = (center + sep / 2.0, center, center, center)

    omega1 = create_instanton_config(L, center1, rho, charge=+1)
    omega2 = create_instanton_config(L, center2, rho, charge=-1)

    if rotation_axis == "x":
        R = rotation_matrix_x(theta)
    elif rotation_axis == "y":
        R = rotation_matrix_y(theta)
    else:
        R = rotation_matrix_z(theta)

    omega2_rot = rotate_color_field(omega2, R)
    return omega1, omega2_rot

def create_two_lump_family(L, sep, rho=0.8, theta=0.0, rotation_axis="z", abelianized=False):
    omega1, omega2 = create_two_lump_components(L, sep, rho, theta, rotation_axis)

    if abelianized:
        omega1 = abelianize_projection(omega1, axis=2)
        omega2 = abelianize_projection(omega2, axis=2)

    return omega1 + omega2, omega1, omega2

# =============================================================================
# OBSERVABLES
# =============================================================================

def commutator_overlap(omega1, omega2):
    L = omega1.shape[0]
    total = 0.0

    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            A1_mu = su2_alg_matrix(omega1[x, y, z, t, mu])
                            A1_nu = su2_alg_matrix(omega1[x, y, z, t, nu])
                            A2_mu = su2_alg_matrix(omega2[x, y, z, t, mu])
                            A2_nu = su2_alg_matrix(omega2[x, y, z, t, nu])

                            comm1 = A1_mu @ A2_nu - A2_nu @ A1_mu
                            comm2 = A1_nu @ A2_mu - A2_mu @ A1_nu
                            diff = comm1 - comm2
                            total += np.sum(np.abs(diff) ** 2)

    return float(total)

def compute_family_metrics(L, sep, rho, theta, rotation_axis="z", beta=2.0):
    omega_tot_na, omega1_na, omega2_na = create_two_lump_family(
        L=L, sep=sep, rho=rho, theta=theta, rotation_axis=rotation_axis, abelianized=False
    )
    _, V_na, nplaq = compute_wilson_action_and_vorticity(omega_tot_na, L, beta=beta)
    _, V1_na, _ = compute_wilson_action_and_vorticity(omega1_na, L, beta=beta)
    _, V2_na, _ = compute_wilson_action_and_vorticity(omega2_na, L, beta=beta)
    O_na = commutator_overlap(omega1_na, omega2_na)
    DVint_na = V_na - V1_na - V2_na

    omega_tot_ab, omega1_ab, omega2_ab = create_two_lump_family(
        L=L, sep=sep, rho=rho, theta=theta, rotation_axis=rotation_axis, abelianized=True
    )
    _, V_ab, _ = compute_wilson_action_and_vorticity(omega_tot_ab, L, beta=beta)
    _, V1_ab, _ = compute_wilson_action_and_vorticity(omega1_ab, L, beta=beta)
    _, V2_ab, _ = compute_wilson_action_and_vorticity(omega2_ab, L, beta=beta)
    O_ab = commutator_overlap(omega1_ab, omega2_ab)
    DVint_ab = V_ab - V1_ab - V2_ab

    DV_NA = V_na - V_ab
    DDVint = DVint_na - DVint_ab

    eps = 1e-12
    metrics = {
        "L": float(L),
        "sep": float(sep),
        "rho": float(rho),
        "theta": float(theta),
        "Np": float(nplaq),
        "V_nonAb": V_na,
        "V_Ab": V_ab,
        "O_nonAb": O_na,
        "O_Ab": O_ab,
        "DeltaV_NA": DV_NA,
        "DeltaV_int_nonAb": DVint_na,
        "DeltaV_int_Ab": DVint_ab,
        "DeltaDeltaV_int": DDVint,
        "R_NA": DV_NA / max(abs(V_ab), eps),
        "R_int": DDVint / max(abs(DVint_ab), eps),
        "d_NA": DV_NA / max(float(nplaq), eps),
        "d_int": DDVint / max(float(nplaq), eps),
    }
    return metrics

# =============================================================================
# SCALING PATHS
# =============================================================================

def path_fixed_units(L):
    """
    Localized object in a larger box:
    sep, rho fixed in lattice units.
    """
    return {"sep": 2.0, "rho": 0.8}

def path_fixed_fractions(L):
    """
    Geometric self-similarity:
    sep/L and rho/L fixed.
    """
    return {"sep": L / 3.0, "rho": 0.13 * L}

def build_dataset(L_values, theta, rotation_axis, beta, path_mode):
    rows = []
    for L_raw in L_values:
        L = int(L_raw)

        if path_mode == "fixed_units":
            params = path_fixed_units(L)
        elif path_mode == "fixed_fractions":
            params = path_fixed_fractions(L)
        else:
            raise ValueError("unknown path_mode")

        row = compute_family_metrics(
            L=L,
            sep=params["sep"],
            rho=params["rho"],
            theta=theta,
            rotation_axis=rotation_axis,
            beta=beta,
        )
        row["path_mode"] = path_mode
        rows.append(row)
    return rows

# =============================================================================
# FIT / DIAGNOSTICS
# =============================================================================

def fit_linear_basis(X, y, label):
    """
    Fit y ≈ X @ coeff by least squares.
    """
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
    """
    y ~ B * L^p, only if y positive.
    """
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
    """
    Smaller is more plateau-like in the last 3 points.
    """
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
    """
    Heuristic:
    lower score => better support for a simple finite limit.
    higher score => more suspicious.
    """
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
# RUN DATASET
# =============================================================================

beta = 2.0
rotation_axis = "y"
theta = np.pi / 4.0
L_values = np.array([4, 5, 6, 7, 8, 10, 12], dtype=np.float64)

dataset_fixed_units = build_dataset(L_values, theta, rotation_axis, beta, "fixed_units")
dataset_fixed_fractions = build_dataset(L_values, theta, rotation_axis, beta, "fixed_fractions")

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
    print("\n" + "=" * 86)
    print(f"DATASET: {name}")
    print("=" * 86)
    print(
        f"{'L':>4} {'sep':>8} {'rho':>8} "
        f"{'DV_NA':>12} {'DDVint':>12} {'R_NA':>12} {'R_int':>12} {'d_NA':>12} {'d_int':>12}"
    )
    print("-" * 100)
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
    print("\n" + "=" * 86)
    print(f"FALSIFICATION ANALYSIS: {name}")
    print("=" * 86)

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
ax.plot(L_values, get_series(dataset_fixed_units, "DeltaV_NA"), "o-", label="fixed units")
ax.plot(L_values, get_series(dataset_fixed_fractions, "DeltaV_NA"), "s--", label="fixed fractions")
ax.set_title("DeltaV_NA vs L")
ax.set_xlabel("L")
ax.set_ylabel("DeltaV_NA")
ax.grid(alpha=0.3)
ax.legend()

# 2: DeltaDeltaV_int
ax = axes[0, 1]
ax.plot(L_values, get_series(dataset_fixed_units, "DeltaDeltaV_int"), "o-", label="fixed units")
ax.plot(L_values, get_series(dataset_fixed_fractions, "DeltaDeltaV_int"), "s--", label="fixed fractions")
ax.set_title("DeltaDeltaV_int vs L")
ax.set_xlabel("L")
ax.set_ylabel("DeltaDeltaV_int")
ax.grid(alpha=0.3)
ax.legend()

# 3: R_NA
ax = axes[0, 2]
ax.plot(L_values, get_series(dataset_fixed_units, "R_NA"), "o-", label="fixed units")
ax.plot(L_values, get_series(dataset_fixed_fractions, "R_NA"), "s--", label="fixed fractions")
ax.set_title("R_NA = DeltaV_NA / V_Ab")
ax.set_xlabel("L")
ax.set_ylabel("R_NA")
ax.grid(alpha=0.3)
ax.legend()

# 4: R_int
ax = axes[1, 0]
ax.plot(L_values, get_series(dataset_fixed_units, "R_int"), "o-", label="fixed units")
ax.plot(L_values, get_series(dataset_fixed_fractions, "R_int"), "s--", label="fixed fractions")
ax.set_title("R_int = DeltaDeltaV_int / |DeltaV_int_Ab|")
ax.set_xlabel("L")
ax.set_ylabel("R_int")
ax.grid(alpha=0.3)
ax.legend()

# 5: d_NA
ax = axes[1, 1]
ax.plot(L_values, get_series(dataset_fixed_units, "d_NA"), "o-", label="fixed units")
ax.plot(L_values, get_series(dataset_fixed_fractions, "d_NA"), "s--", label="fixed fractions")
ax.set_title("d_NA = DeltaV_NA / Np")
ax.set_xlabel("L")
ax.set_ylabel("d_NA")
ax.grid(alpha=0.3)
ax.legend()

# 6: d_int
ax = axes[1, 2]
ax.plot(L_values, get_series(dataset_fixed_units, "d_int"), "o-", label="fixed units")
ax.plot(L_values, get_series(dataset_fixed_fractions, "d_int"), "s--", label="fixed fractions")
ax.set_title("d_int = DeltaDeltaV_int / Np")
ax.set_xlabel("L")
ax.set_ylabel("d_int")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
out_path = "finite_size_falsification_suite.png"
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

print("\n" + "=" * 86)
print("SUMMARY")
print("=" * 86)
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

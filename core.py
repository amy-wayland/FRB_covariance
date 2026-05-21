import numpy as np
import pyccl as ccl
import HaloProfiles as hp
import os
from functools import lru_cache
from pyccl.halos import Profile2pt
from scipy.interpolate import RegularGridInterpolator

# -----------------------------------------------------------
# Cosmology
# -----------------------------------------------------------

COSMO_P18 = {"Omega_c": 0.26066676,
             "Omega_b": 0.048974682,
             "h": 0.6766,
             "n_s": 0.9665,
             "sigma8": 0.8102,
             "matter_power_spectrum": "halofit"}
             
cosmo = ccl.Cosmology(**COSMO_P18)
cosmo.compute_growth()

# ------------------------------------------------------------
# Halo Model
# ------------------------------------------------------------

prof2pt = Profile2pt()
hmd = ccl.halos.MassDef200c

cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd)

pE = hp.HaloProfileDensityHE(
     mass_def=hmd, concentration=cM, 
     lMc=14.0, beta=0.6, A_star=0.03, eta_b=0.5)

hmc = ccl.halos.HMCalculator(
      mass_function=nM, halo_bias=bM, mass_def=hmd, 
      log10M_max=15.0, log10M_min=10.0, nM=32)

# ------------------------------------------------------------
# Kernels
# ------------------------------------------------------------

def _round(x, nd=5):
    '''
    Round inputs before caching.
    '''
    return float(np.round(x, nd))

@lru_cache(maxsize=256)
def rho_e_bar(a):
    '''
    Normalisation of the electron profile.
    '''
    a = _round(a)
    return pE.get_normalization(cosmo, a, hmc=hmc)

@lru_cache(maxsize=10000)
def P_lin(k, a):
    '''
    Linear matter power spectrum.
    '''
    k = _round(k)
    a = _round(a)
    return ccl.linear_matter_power(cosmo, k, a)

@lru_cache(maxsize=1000)
def P_e(k, a):
    '''
    Halo model electron power spectrum.
    '''
    return float(ccl.halos.halomod_power_spectrum(
        cosmo, hmc, _round(k), _round(a), prof=pE))

def P_e_array(k, a):
    '''
    Halo model electron power spectrum.
    '''
    return ccl.halos.halomod_power_spectrum(
        cosmo, hmc, k, a, prof=pE)



@lru_cache(maxsize=10000)
def I_1_1(k, a):
    k = _round(k)
    a = _round(a)
    return hmc.I_1_1(cosmo, k, a, prof=pE) / rho_e_bar(a)

@lru_cache(maxsize=10000)
def I_1_2(k1, k2, a):
    k1, k2 = sorted([_round(k1), _round(k2)])
    a = _round(a)
    k = np.array([k1, k2])
    return hmc.I_1_2(cosmo, k, a, prof=pE, prof_2pt=prof2pt, diag=False)[0, 1] / rho_e_bar(a)**2

@lru_cache(maxsize=10000)
def I_0_3(k1, k2, k3, a):
    k1, k2, k3 = sorted([_round(k1), _round(k2), _round(k3)])
    a = _round(a)
    norm = rho_e_bar(a)
    
    def integrand(M):
        u1 = pE.fourier(cosmo, k1, M, a) / norm
        u2 = pE.fourier(cosmo, k2, M, a) / norm
        u3 = pE.fourier(cosmo, k3, M, a) / norm
        return u1 * u2 * u3

    return hmc.integrate_over_massfunc(integrand, cosmo, a)

def F2(k1, k2, mu):
    '''
    Second order perturbation theory kernel.
    '''
    return (5/7 + 0.5 * (k1/k2 + k2/k1) * mu + 2/7 * mu**2)

# ------------------------------------------------------------
# Bispectrum
# ------------------------------------------------------------

def B_tree(k1, k2, k3, a):
    '''
    Tree level bispectrum.
    '''
    mu12 = (k3**2 - k1**2 - k2**2) / (2 * k1 * k2)
    mu23 = (k1**2 - k2**2 - k3**2) / (2 * k2 * k3)
    mu31 = (k2**2 - k3**2 - k1**2) / (2 * k3 * k1)
    
    P1 = P_lin(k1, a)
    P2 = P_lin(k2, a)
    P3 = P_lin(k3, a)
    
    return (2 * F2(k1, k2, mu12) * P1 * P2 +
            2 * F2(k2, k3, mu23) * P2 * P3 +
            2 * F2(k3, k1, mu31) * P3 * P1)

def B_1h(k1, k2, k3, a):
    '''
    1-halo term.
    '''
    return I_0_3(k1, k2, k3, a)

def B_2h(k1, k2, k3, a):
    '''
    2-halo term.
    '''
    return (I_1_1(k1, a) * I_1_2(k2, k3, a) * P_lin(k1, a) +
            I_1_1(k2, a) * I_1_2(k3, k1, a) * P_lin(k2, a) +
            I_1_1(k3, a) * I_1_2(k1, k2, a) * P_lin(k3, a))
    
def B_3h(k1, k2, k3, a):
    '''
    3-halo term.
    '''
    B_m = B_tree(k1, k2, k3, a)
    return I_1_1(k1, a) * I_1_1(k2, a) * I_1_1(k3, a) * B_m

def B_e(k1, k2, k3, a):
    '''
    Full electron bispectrum.
    '''
    return B_1h(k1, k2, k3, a) + B_2h(k1, k2, k3, a) + B_3h(k1, k2, k3, a)

# ------------------------------------------------------------
# Precompute Bispectrum Grid
# ------------------------------------------------------------

def build_bispectrum_grid(
    k_grid = np.logspace(-3, 1, 50),
    a_grid = np.linspace(0.2, 1.0, 20),
    phi_grid = np.linspace(0, 2*np.pi, 40)):
    
    cosphi = np.cos(phi_grid)
    B_grid = np.zeros((len(a_grid), len(k_grid), len(k_grid), len(phi_grid)))

    for ia, a in enumerate(a_grid):
        print(f"Precomputing a={a:.2f}", flush=True)
        for j, k2 in enumerate(k_grid):
            for l, k3 in enumerate(k_grid):
                k1 = np.sqrt(k2**2 + k3**2 + 2*k2*k3*cosphi)
                k1 = np.clip(k1, 1e-3, 1e2)
                B_grid[ia, j, l, :] = np.array([B_e(k1_val, k2, k3, a) for k1_val in k1])

    return B_grid, k_grid, a_grid, phi_grid

# ------------------------------------------------------------
# Save Bispectrum Grid
# ------------------------------------------------------------

def save_bispectrum(filename="bispectrum_grid.npz"):
    if os.path.exists(filename):
        print(f"Found existing grid: {filename}. Skipping recomputation.")
        return

    print("No existing grid found. Computing bispectrum.")

    B_grid, k_grid, a_grid, phi_grid = build_bispectrum_grid()

    np.savez_compressed(filename,
                         B_grid=B_grid,
                         k_grid=k_grid,
                         a_grid=a_grid,
                         phi_grid=phi_grid)

    print(f"Saved bispectrum grid as {filename}.")

# ------------------------------------------------------------
# Load and Interpolate Bispectrum
# ------------------------------------------------------------

class LogExtrapolatingBispectrumInterpolator:
    """
    Interpolates and extrapolates the bispectrum log-linearly in both k
    dimensions and in log(B) space.

    The two k axes are transformed to log(k) internally; the bispectrum
    values are stored as log(B).  Points outside the k grid are extrapolated
    using the local d(log B)/d(log k) slope computed from the two outermost
    grid slices.  The a and phi axes are interpolated linearly and are NOT
    extrapolated.
    """

    def __init__(self, a_grid, k_grid, phi_grid, B_grid):
        log_k = np.log(k_grid)
        log_B = np.log(np.maximum(B_grid, 1e-300))

        self._log_k_min = log_k[0]
        self._log_k_max = log_k[-1]

        kw = dict(method='linear', bounds_error=False, fill_value=None)
        self._interp = RegularGridInterpolator(
            (a_grid, log_k, log_k, phi_grid), log_B, **kw
        )

        # d(log_B)/d(log_k) at the lower and upper k boundaries.
        # Shape for k1 slopes: (na, nk2, nphi); for k2: (na, nk1, nphi).
        dk_lo = log_k[1] - log_k[0]
        dk_hi = log_k[-1] - log_k[-2]

        self._sk1_lo = RegularGridInterpolator(
            (a_grid, log_k, phi_grid),
            (log_B[:, 1, :, :] - log_B[:, 0, :, :]) / dk_lo, **kw)
        self._sk1_hi = RegularGridInterpolator(
            (a_grid, log_k, phi_grid),
            (log_B[:, -1, :, :] - log_B[:, -2, :, :]) / dk_hi, **kw)
        self._sk2_lo = RegularGridInterpolator(
            (a_grid, log_k, phi_grid),
            (log_B[:, :, 1, :] - log_B[:, :, 0, :]) / dk_lo, **kw)
        self._sk2_hi = RegularGridInterpolator(
            (a_grid, log_k, phi_grid),
            (log_B[:, :, -1, :] - log_B[:, :, -2, :]) / dk_hi, **kw)

    def __call__(self, pts):
        pts  = np.atleast_2d(np.asarray(pts, dtype=float))
        a    = pts[:, 0]
        lk1  = np.log(pts[:, 1])
        lk2  = np.log(pts[:, 2])
        phi  = pts[:, 3]

        lk1_c = np.clip(lk1, self._log_k_min, self._log_k_max)
        lk2_c = np.clip(lk2, self._log_k_min, self._log_k_max)

        log_B = self._interp(np.column_stack([a, lk1_c, lk2_c, phi]))
        # Log-linear extrapolation in k1
        d1 = lk1 - lk1_c
        m = d1 < 0
        if np.any(m):
            log_B[m] += self._sk1_lo(np.column_stack([a[m], lk2_c[m], phi[m]])) * d1[m]
        m = d1 > 0
        if np.any(m):
            log_B[m] += self._sk1_hi(np.column_stack([a[m], lk2_c[m], phi[m]])) * d1[m]

        # Log-linear extrapolation in k2
        d2 = lk2 - lk2_c
        m = d2 < 0
        if np.any(m):
            log_B[m] += self._sk2_lo(np.column_stack([a[m], lk1_c[m], phi[m]])) * d2[m]
        m = d2 > 0
        if np.any(m):
            log_B[m] += self._sk2_hi(np.column_stack([a[m], lk1_c[m], phi[m]])) * d2[m]
        return np.exp(log_B)


def load_bispectrum(filename="bispectrum_grid.npz"):
    d = np.load(filename)
    interp = LogExtrapolatingBispectrumInterpolator(
        d["a_grid"], d["k_grid"], d["phi_grid"], d["B_grid"]
    )
    return interp, d

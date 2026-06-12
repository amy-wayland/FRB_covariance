import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, eval_legendre, jv
from scipy.integrate import cumulative_trapezoid, simpson
from core import cosmo, P_e, load_bispectrum, P_e_array

# -----------------------------------------------------------
# Load Bispectrum
# -----------------------------------------------------------

B_interp, data = load_bispectrum("bispectrum_grid.npz")
k_grid = data["k_grid"]

# -----------------------------------------------------------
# FRB Kernel
# -----------------------------------------------------------

# FRB redshift distribution
alpha = 3.5
zz = np.linspace(0, 2, 128)
aa = 1/(1+zz)
nz = zz**2 * np.exp(-alpha*zz)
nz = nz/np.trapz(nz, zz)

# Want [G] = [cm^3 kg^{-1} s^{-2}]
G_m3_per_kg_per_s2 = ccl.physical_constants.GNEWT
G_cm3_per_kg_per_s2 = 1e6 * G_m3_per_kg_per_s2
G = G_cm3_per_kg_per_s2

# Want [m_p] = [kg]
mp_kg = 1.67262e-27
mp = mp_kg

# Want [H_0] = [Mpc s^{-1} Mpc^{-1}]
pc = 3.0857e13 # 1pc in km
km_to_Mpc = 1/(1e6*pc) # 1 km = 3.24078e-20 Mpc
H0_per_s = cosmo['H0'] * km_to_Mpc
H0 = H0_per_s

# Prefactor in units of [A] = [cm^{-3}]
xH = 0.75
A = (3*cosmo['Omega_b']*H0**2)/(8*np.pi*G*mp) * (1+xH)/2

# Cumulative integral of n(z)
nz_integrated = 1 - cumulative_trapezoid(nz, zz, initial=0)

# [W_{\chi}] = [A] = [cm^{-3}]
# Factor of 1e6 so that Cl is in units of [pc cm^{-3}]
h = cosmo['H0'] / 100
chis = ccl.comoving_radial_distance(cosmo, aa)
W_chi = A * (1+zz) * nz_integrated * 1e6

# Interpolate Kernel
chi_of_z_interp = interp1d(zz, chis, bounds_error=False, fill_value="extrapolate")
z_of_chi_interp = interp1d(chis, zz, bounds_error=False, fill_value="extrapolate")
W_interp = interp1d(chis, W_chi, bounds_error=False, fill_value=0.0)


k_interp = np.geomspace(1e-4,1e2,500)
a_interp = np.linspace(.2, 1, 100)
Pe_arr = P_e_array(k_interp, a_interp)
Pe_interpolator = ccl.pk2d.Pk2D(a_arr = a_interp,
                                lk_arr = np.log(k_interp),
                                pk_arr = np.log(Pe_arr),
                                extrap_order_lok=1,
                                extrap_order_hik=1,
                                is_logp=True)



def E_of_chi(chi):
    z = z_of_chi_interp(chi)
    a = 1/(1+z)
    return ccl.h_over_h0(cosmo, a)

# -----------------------------------------------------------
# DM-z Auto-Covariance
# -----------------------------------------------------------

def W_single_FRB(chi, chi_s):
    '''
    DM kernel for a single FRB at comoving distance chi_s.
    W_D(chi) = A * (1+z(chi)) for chi < chi_s, else 0.
    '''
    if chi >= chi_s:
        return 0.0
    z = float(z_of_chi_interp(chi))
    return A * (1+z) * 1e6  # same units as W_chi [pc cm^{-3}]

import time

def C_ij_ell(ell, zi, zj, Nchi=100):
    '''
    Angular power spectrum C_ij(ell) under the Limber approximation.
    '''
    chi_i = float(ccl.comoving_radial_distance(cosmo, 1/(1+zi)))
    chi_j = float(ccl.comoving_radial_distance(cosmo, 1/(1+zj)))
    chi_max = min(chi_i, chi_j)
    chi_arr = np.linspace(1e-2, chi_max, Nchi)

    z_arr = z_of_chi_interp(chi_arr)
    a_arr = 1 / (1 + z_arr)
    k_arr = np.clip((ell + 0.5) / chi_arr, 1e-3, 1e2)
    # All chi in chi_arr <= chi_max <= min(chi_i, chi_j), so both kernels are non-zero
    W_i = A * (1 + z_arr) * (chi_arr < chi_i) * 1e6
    W_j = A * (1 + z_arr) * (chi_arr < chi_j) * 1e6
    Pe_arr = np.diag(Pe_interpolator(k_arr,a_arr))
    integrand = W_i * W_j * Pe_arr / chi_arr**2
    return np.trapz(integrand, chi_arr)

def cov_DD(zi, zj, cos_theta, ell_max=500, Nchi=100, flat_sky=True):
    '''
    DM-DM auto-covariance summed over multipoles.

    By default this uses the original full-sky Legendre expansion.
    If flat_sky=True, it uses the flat-sky Bessel-integral approximation:
        Cov(theta) = int d ell [ell / (2 pi)] J_0(ell theta) C_ij(ell).
    '''
    ell_arr = np.unique(np.round(np.logspace(0, np.log10(500), 100)).astype(int))
    C_ell_arr = np.array([C_ij_ell(ell, zi, zj, Nchi=Nchi) for ell in ell_arr])

    if not flat_sky:
        P_ell_arr = np.array([float(eval_legendre(ell, cos_theta)) for ell in ell_arr])
        integrand = (2 * ell_arr + 1) / (4 * np.pi) * P_ell_arr * C_ell_arr
        return np.trapz(integrand, ell_arr)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    ell_integral = np.geomspace(ell_arr[0], ell_arr[-1], int(1e5))
    C_ell_interp = np.interp(ell_integral, ell_arr, C_ell_arr)
    bessel = jv(0, ell_integral * theta)
    integrand = ell_integral * C_ell_interp * bessel

    return simpson(integrand, x=ell_integral) / (2.0 * np.pi)

# -----------------------------------------------------------
# Cl^{DD} Auto-Covariance
# -----------------------------------------------------------

def C_ell_DD(ell, Nchi=100):
    '''
    DM angular power spectrum C_ell^DD evaluated at the given multipole:
        C_ell^DD = int dchi/chi^2 * W(chi)^2 * P_e((ell+0.5)/chi, z(chi))
    '''
    chi_max = float(ccl.comoving_radial_distance(cosmo, 1/(1+2.0)))
    chi_arr = np.linspace(1e-2, chi_max, Nchi)
    z_arr   = z_of_chi_interp(chi_arr)
    a_arr   = 1 / (1 + z_arr)
    W_arr   = W_interp(chi_arr)
    k_arr   = np.clip((ell + 0.5) / chi_arr, 1e-3, 1e2)
    Pe_arr  = np.diag(Pe_interpolator(k_arr, a_arr))
    return np.trapz(W_arr**2 * Pe_arr / chi_arr**2, chi_arr)

def cov_ClCl(ell, ell_prime, f_sky=1.0, Nchi=100, delta_ell=1):
    '''
    Gaussian (Knox) covariance of the bin-averaged DM power spectrum.
    C_ell^DD is evaluated at the central multipole; averaging over delta_ell
    modes reduces variance by 1/delta_ell:
        Cov[C-bar, C-bar] = 2 / ((2*ell+1) * delta_ell * f_sky) * C(ell)^2
    '''
    if ell != ell_prime:
        return 0.0
    C = C_ell_DD(ell, Nchi=Nchi)
    return 2.0 / ((2 * ell + 1) * delta_ell * f_sky) * C**2

# -----------------------------------------------------------
# Cross-Covariance
# -----------------------------------------------------------

Mpc_to_pc = 3.0857e18

def covariance_DM_Cl(ell, z1, z2, z3, Nchi=20, Nmu=40):
    '''
    Cross-covariance Cov[D(z1), C_ell^DD] evaluated at the central multipole.

    For ell > 100 uses the high-ell Bessel approximation:
        P_ell(mu) ≈ J_0((ell + 1/2) * arccos(mu))
    otherwise the exact Legendre polynomial via eval_legendre.

    Memory scales as O(Nchi^2 * Nmu) via a chi1 loop.
    '''
    chi1_max = ccl.comoving_radial_distance(cosmo, 1/(1+z1))
    chi2_max = ccl.comoving_radial_distance(cosmo, 1/(1+z2))
    chi3_max = ccl.comoving_radial_distance(cosmo, 1/(1+z3))
    chi_H    = ccl.physical_constants.CLIGHT_HMPC

    k_grid_max = float(np.max(k_grid))
    k_grid_min = float(np.min(k_grid))
    chi_min_ell = (ell + 0.5) / k_grid_max
    chi_max_ell = (ell + 0.5) / k_grid_min

    chi1_arr = np.linspace(max(chi_min_ell, 1e-2), min(chi1_max, chi_max_ell), Nchi)
    chi2_arr = np.linspace(max(chi_min_ell, 1e-2), min(chi2_max, chi_max_ell), Nchi)
    chi3_arr = np.linspace(max(chi_min_ell, 1e-2), min(chi3_max, chi_max_ell), Nchi)
    mu_arr   = np.linspace(-1, 1, Nmu)

    if chi1_arr[0] >= chi1_arr[-1] or len(chi1_arr) < 2: return 0.0
    if chi2_arr[0] >= chi2_arr[-1] or len(chi2_arr) < 2: return 0.0
    if chi3_arr[0] >= chi3_arr[-1] or len(chi3_arr) < 2: return 0.0

    z1_arr = z_of_chi_interp(chi1_arr)
    a1_arr = 1 / (1 + z1_arr)
    W1_arr = np.where(chi1_arr < chi1_max, A * (1 + z1_arr) * 1e6, 0.0)
    W2_arr = W_interp(chi2_arr)
    W3_arr = W_interp(chi3_arr)

    E1_arr = E_of_chi(chi1_arr)
    E2_arr = E_of_chi(chi2_arr)
    E3_arr = E_of_chi(chi3_arr)

    dchi1 = chi1_arr[1] - chi1_arr[0]
    dchi2 = chi2_arr[1] - chi2_arr[0]
    dchi3 = chi3_arr[1] - chi3_arr[0]

    G1  = W1_arr * E1_arr
    G2  = W2_arr / (chi2_arr**3 / E2_arr)
    G3  = W3_arr / (chi3_arr**3 / E3_arr)
    G23 = G2[:, None] * G3[None, :]                  # (Nchi, Nchi)

    # Limber wavenumbers and k1 triangle closing vector
    k2_arr = (ell + 0.5) / chi2_arr
    k3_arr = (ell + 0.5) / chi3_arr
    k2_3   = k2_arr[:, None, None]
    k3_3   = k3_arr[None, :, None]
    mu_3   = mu_arr[None, None, :]

    k1_3d = np.clip(
        np.sqrt(np.maximum(k2_3**2 + k3_3**2 - 2*k2_3*k3_3*mu_3, 0.0)),
        k_grid_min, k_grid_max)                       # (Nchi, Nchi, Nmu)

    # Bispectrum lookup angle and Legendre / high-ell Bessel approximation
    phi_arr  = np.arccos(-mu_arr)                     # minus: k1 = k3 - k2
    if ell > 100:
        Pell_arr = jv(0, (ell + 0.5) * np.arccos(mu_arr))
    else:
        Pell_arr = eval_legendre(int(ell), mu_arr)

    # Flat columns for B_interp pts — k2, k3, phi fixed for all chi1
    n_inner = Nchi * Nchi * Nmu
    k2_col  = np.broadcast_to(k2_3, (Nchi, Nchi, Nmu)).ravel().copy()
    k3_col  = np.broadcast_to(k3_3, (Nchi, Nchi, Nmu)).ravel().copy()
    phi_col = np.broadcast_to(phi_arr[None, None, :], (Nchi, Nchi, Nmu)).ravel().copy()

    prefactor = ((ell + 0.5)**3) / (4 * np.pi**2) / chi_H**3
    pts = np.empty((n_inner, 4))
    pts[:, 1] = k2_col
    pts[:, 2] = k3_col
    pts[:, 3] = phi_col

    result = 0.0
    for i1 in range(Nchi):
        if G1[i1] == 0.0:
            continue
        pts[:, 0] = a1_arr[i1]
        B_3d  = B_interp(pts).reshape(Nchi, Nchi, Nmu)
        j0_3d = spherical_jn(0, k1_3d * chi1_arr[i1])
        mu_int = np.trapz(j0_3d * Pell_arr * B_3d, mu_arr, axis=-1)
        result += G1[i1] * np.sum(G23 * mu_int)

    return result * prefactor * dchi1 * dchi2 * dchi3 / Mpc_to_pc

# -----------------------------------------------------------
# Build Full Covariance
# -----------------------------------------------------------

def build_covariance_matrix(ell_arr, z_frb, cos_theta_matrix,
                            f_sky=0.7, Nchi=50, Nmu=40,
                            flat_sky=False, delta_ell=1):
    '''
    Build the full (N_frb + N_ell) x (N_frb + N_ell) covariance matrix:

        C = [ Cov[D_i, D_j]      Cov[D_i, C_ell_b]   ]
            [ Cov[C_ell_a, D_j]  Cov[C_ell_a, C_ell_b] ]

    ell_arr : array-like of multipoles (length N_ell)
    z_frb   : array of FRB redshifts   (length N_frb)
    '''
    ell_arr = np.atleast_1d(np.asarray(ell_arr, dtype=float))
    N_frb = len(z_frb)
    N_ell = len(ell_arr)
    Ntot  = N_frb + N_ell
    cov   = np.zeros((Ntot, Ntot))

    # Block 1: Cov[D_i, D_j], indices 0..N_frb-1
    print("Computing Cov[D_i, D_j]...")
    for i in range(N_frb):
        for j in range(i, N_frb):
            val = cov_DD(
                z_frb[i], z_frb[j], cos_theta_matrix[i, j],
                Nchi=Nchi, flat_sky=flat_sky
            )
            cov[i, j] = val
            cov[j, i] = val
            print(f"  ({i},{j}): {val:.3e}")
    print()

    # Blocks 2 and 3: Cov[D_i, C_ell_b], indices (0..N_frb-1, N_frb..Ntot-1)
    print("Computing Cov[D_i, C_ell]...")
    z_max = float(zz[-1])
    for i in range(N_frb):
        for b, ell in enumerate(ell_arr):
            val = covariance_DM_Cl(ell, z_frb[i], z_max, z_max,
                                   Nchi=Nchi, Nmu=Nmu)
            cov[i, N_frb + b] = val
            cov[N_frb + b, i] = val
            print(f"  FRB {i} (z={z_frb[i]:.2f}), ell={ell:.0f}: {val:.3e}", end='\r')
    print()

    # Block 4: Cov[C_ell_a, C_ell_b], indices N_frb..Ntot-1 (diagonal in Knox)
    print("Computing Cov[C_ell, C_ell]...")
    for a, ell_a in enumerate(ell_arr):
        for b, ell_b in enumerate(ell_arr):
            val = cov_ClCl(ell_a, ell_b, f_sky=f_sky, Nchi=Nchi, delta_ell=delta_ell)
            cov[N_frb + a, N_frb + b] = val
            print(f"  ({a},{b}) ell=({ell_a:.0f},{ell_b:.0f}): {val:.3e}", end='\r')
    print()

    # Correlation coefficient
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)

    return cov, corr

# -----------------------------------------------------------
# Plot Correlation Matrix
# -----------------------------------------------------------

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.linewidth": 1.2
})

def plot_correlation_matrix(corr, z_frb, ell_arr, f_sky=1.0):
    '''
    Plot the correlation coefficient matrix r_ij.
    ell_arr may be a scalar or array of multipoles.
    '''
    ell_arr = np.atleast_1d(np.asarray(ell_arr, dtype=float))
    N_frb = len(z_frb)
    N_ell = len(ell_arr)
    Ntot  = N_frb + N_ell

    fig, ax = plt.subplots(figsize=(max(7, Ntot * 0.6), max(6, Ntot * 0.55)))

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$r_{ij}$')

    labels = [f'$\\mathcal{{D}}(z={z:.2f})$' for z in z_frb] \
           + [f'$C_{{\\ell={int(ell)}}}^{{\\mathcal{{DD}}}}$' for ell in ell_arr]
    ax.set_xticks(range(Ntot))
    ax.set_yticks(range(Ntot))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Highlight covariance blocks
    ax.add_patch(Rectangle((-0.5, -0.5), N_frb, N_frb,
                           fill=False, edgecolor='gold', lw=2,
                           label='$\\mathrm{Cov}[\\mathcal{D}_i, \\mathcal{D}_j]$'))
    ax.add_patch(Rectangle((N_frb - 0.5, -0.5), N_ell, N_frb,
                           fill=False, edgecolor='deepskyblue', lw=2,
                           label='$\\mathrm{Cov}[\\mathcal{D}_i, C_\\ell]$'))
    ax.add_patch(Rectangle((-0.5, N_frb - 0.5), N_frb, N_ell,
                           fill=False, edgecolor='deepskyblue', lw=2))
    ax.add_patch(Rectangle((N_frb - 0.5, N_frb - 0.5), N_ell, N_ell,
                           fill=False, edgecolor='blueviolet', lw=2,
                           label='$\\mathrm{Cov}[C_\\ell, C_\\ell]$'))

    ell_label = ','.join(f'{int(e)}' for e in ell_arr)
    #ax.set_title(f'Correlation matrix ($\\ell={ell_label}$, $f_{{\\rm sky}}={f_sky}$)', fontsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.tick_params(which='major', direction='in', length=5, width=0.8, top=True, right=True)
    ax.tick_params(which='minor', direction='in', length=2, width=0.6, top=True, right=True)

    plt.tight_layout()
    fname = f'cov/correlation_matrix_ell.pdf'
    plt.savefig(fname, format="pdf", bbox_inches="tight")

    return fig

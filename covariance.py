import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
from scipy.special import eval_legendre
from scipy.integrate import cumulative_trapezoid
from core import cosmo, P_e, load_bispectrum

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

def C_ij_ell(ell, zi, zj, Nchi=100):
    '''
    Angular power spectrum C_ij(ell) under the Limber approximation.
    '''
    chi_i = float(ccl.comoving_radial_distance(cosmo, 1/(1+zi)))
    chi_j = float(ccl.comoving_radial_distance(cosmo, 1/(1+zj)))
    chi_max = min(chi_i, chi_j)
    chi_arr = np.linspace(1e-2, chi_max, Nchi)

    integrand = np.zeros(Nchi)
    for idx, chi in enumerate(chi_arr):
        z = float(z_of_chi_interp(chi))
        a = 1/(1+z)
        k = (ell+0.5)/chi
        k = np.clip(k, 1e-3, 1e2)
        Wi = W_single_FRB(chi, chi_i)
        Wj = W_single_FRB(chi, chi_j)
        Pe = P_e(k, a)
        integrand[idx] = Wi*Wj*Pe/chi**2

    return np.trapz(integrand, chi_arr)

def cov_DD(zi, zj, cos_theta, ell_max=500, Nchi=100):
    '''
    Full DM-DM auto-covariance summed over multipoles 
    (Eq. 18 of Reischke & Hagstotz 2023).
    '''
    ell_arr = np.unique(np.round(np.logspace(0, np.log10(ell_max), 60)).astype(int))
    C_ell_arr = np.array([C_ij_ell(ell, zi, zj, Nchi=Nchi) for ell in ell_arr])
    P_ell_arr = np.array([float(eval_legendre(ell, cos_theta)) for ell in ell_arr])
    integrand = (2*ell_arr + 1) / (4*np.pi) * P_ell_arr * C_ell_arr
    return np.trapz(integrand, ell_arr)

# -----------------------------------------------------------
# Cl^{DD} Auto-Covariance
# -----------------------------------------------------------

def C_ell_DD(ell, Nchi=100):
    '''
    DM angular power spectrum C_ell^DD:
    C_ell^DD = int dchi/chi^2 * W(chi)^2 * P_e((ell+0.5)/chi, z(chi))
    '''
    chi_max = float(ccl.comoving_radial_distance(cosmo, 1/(1+2.0)))
    chi_arr = np.linspace(1e-2, chi_max, Nchi)

    integrand = np.zeros(Nchi)
    for idx, chi in enumerate(chi_arr):
        z = float(z_of_chi_interp(chi))
        a = 1 / (1 + z)
        k = np.clip((ell + 0.5) / chi, 1e-3, 1e2)
        W = float(W_interp(chi))
        Pe = P_e(k, a)
        integrand[idx] = W**2 * Pe / chi**2

    return np.trapz(integrand, chi_arr)

def cov_ClCl(ell, ell_prime, f_sky=1.0, Nchi=100):
    '''
    Gaussian (Knox) covariance of the DM power spectrum.
    '''
    if ell != ell_prime:
        return 0.0
    C = C_ell_DD(ell, Nchi=Nchi)
    return 2.0 / (2*ell + 1) / f_sky * C**2

# -----------------------------------------------------------
# Cross-Covariance
# -----------------------------------------------------------

Mpc_to_pc = 3.0857e18

def covariance_DM_Cl(ell, z1, z2, z3, Nchi=20, Nphi=40):
    '''
    Compute the cross-covariance between the DM-DM angular power spectrum
    and the DM-redshift relation under the flat sky approximation.
    '''
    chi1_max = ccl.comoving_radial_distance(cosmo, 1/(1+z1))
    chi2_max = ccl.comoving_radial_distance(cosmo, 1/(1+z2))
    chi3_max = ccl.comoving_radial_distance(cosmo, 1/(1+z3))
    chi_H = ccl.physical_constants.CLIGHT_HMPC

    # Set chi_min so that k = (ell+0.5)/chi stays within the bispectrum grid
    # k_grid max = 10 Mpc^{-1}, so chi_min = (ell+0.5)/10
    k_grid_max = float(np.max(k_grid))
    k_grid_min = float(np.min(k_grid))
    chi_min_ell = (ell + 0.5) / k_grid_max
    chi_max_ell = (ell + 0.5) / k_grid_min

    # Clip chi ranges to keep k within bispectrum grid
    chi1_arr = np.linspace(max(chi_min_ell, 1e-2), min(chi1_max, chi_max_ell), Nchi)
    chi2_arr = np.linspace(max(chi_min_ell, 1e-2), min(chi2_max, chi_max_ell), Nchi)
    chi3_arr = np.linspace(max(chi_min_ell, 1e-2), min(chi3_max, chi_max_ell), Nchi)
    phi_arr = np.linspace(0, 2*np.pi, Nphi)

    # Check grids are valid
    if chi1_arr[0] >= chi1_arr[-1] or len(chi1_arr) < 2:
        return 0.0
    if chi2_arr[0] >= chi2_arr[-1] or len(chi2_arr) < 2:
        return 0.0
    if chi3_arr[0] >= chi3_arr[-1] or len(chi3_arr) < 2:
        return 0.0

    # Precompute geometry
    W1_arr = W_interp(chi1_arr)
    W2_arr = W_interp(chi2_arr)
    W3_arr = W_interp(chi3_arr)

    z1_arr = z_of_chi_interp(chi1_arr)
    a1_arr = 1 / (1 + z1_arr)

    E1_arr = E_of_chi(chi1_arr)
    E2_arr = E_of_chi(chi2_arr)
    E3_arr = E_of_chi(chi3_arr)

    k2_arr = (ell + 0.5) / chi2_arr
    k3_arr = (ell + 0.5) / chi3_arr

    dchi1 = chi1_arr[1] - chi1_arr[0]
    dchi2 = chi2_arr[1] - chi2_arr[0]
    dchi3 = chi3_arr[1] - chi3_arr[0]

    G1 = W1_arr / E1_arr
    G2 = W2_arr / (chi2_arr * E2_arr)
    G3 = W3_arr / (chi3_arr**2 * E3_arr)

    phi_template = np.zeros((len(phi_arr), 4))
    phi_template[:, 3] = phi_arr

    result = 0.0
    prefactor = ((ell + 0.5)**2) / (2 * np.pi) * chi_H**3

    for i1 in range(Nchi):
        a1 = a1_arr[i1]
        for i2 in range(Nchi):
            k2 = k2_arr[i2]
            for i3 in range(Nchi):
                k3 = k3_arr[i3]
                phi_template[:, 0] = a1
                phi_template[:, 1] = k2
                phi_template[:, 2] = k3
                B_vals = B_interp(phi_template)
                phi_integral = np.trapz(B_vals, phi_arr) / (2*np.pi)
                weight = G1[i1] * G2[i2] * G3[i3]
                result += weight * phi_integral

    result *= prefactor * dchi1 * dchi2 * dchi3
    return result / Mpc_to_pc

# -----------------------------------------------------------
# Build Full Covariance
# -----------------------------------------------------------

def build_covariance_matrix(ell, z_frb, cos_theta_matrix,
                            f_sky=0.7, Nchi=50, Nphi=40):
    '''
    Build the full (N+1)x(N+1) covariance matrix:
    C = [Cov[D_i, D_j]    Cov[D_i, C_ell]  ]
        [Cov[C_ell, D_j]  Cov[C_ell, C_ell]].
    '''
    N = len(z_frb)
    Ntot = N + 1
    cov = np.zeros((Ntot, Ntot))

    # Block 1: Cov[D_i, D_j], shape (N, N)
    print("Computing Cov[D_i, D_j]...")
    for i in range(N):
        for j in range(i, N):
            val = cov_DD(z_frb[i], z_frb[j], cos_theta_matrix[i, j], Nchi=Nchi)
            cov[i, j] = val
            cov[j, i] = val
            print(f"  ({i},{j}): {val:.3e}", end='\r')
    print()

    # Blocks 2 and 3: Cov[D_i, C_ell],  shape (N, 1) and (1, N)
    # z2=z3=z_max: upper limits of chi2, chi3 integrals correspond to
    # the maximum redshift of the survey over which C_ell^DD is defined.
    print("Computing Cov[D_i, C_ell]...")
    z_max = float(np.max(z_frb))
    for i in range(N):
        val = covariance_DM_Cl(ell, z_frb[i], z_max, z_max,
                                Nchi=Nchi, Nphi=Nphi)
        cov[i, N] = val
        cov[N, i] = val
        print(f"  FRB {i} (z={z_frb[i]:.2f}): {val:.3e}", end='\r')
    print()

    # Block 4: Cov[C_ell, C_ell], scalar
    print("Computing Cov[C_ell, C_ell]...")
    cov[N, N] = cov_ClCl(ell, ell, f_sky=f_sky, Nchi=Nchi)
    print(f"  {cov[N,N]:.3e}")

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

def plot_correlation_matrix(corr, z_frb, ell, f_sky=1.0):
    '''
    Plot the correlation coefficient matrix r_ij.
    '''
    N = len(z_frb)
    Ntot = N + 1

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot r_ij with symmetric color scale
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$r_{ij}$')

    labels = [f'$\\mathcal{{D}}(z={z:.2f})$' for z in z_frb] \
           + [f'$C_{{\\ell={ell}}}^{{\\mathcal{{DD}}}}$']
    ax.set_xticks(range(Ntot))
    ax.set_yticks(range(Ntot))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Highlight covariance blocks
    ax.add_patch(Rectangle((-0.5, -0.5), N, N,
                           fill=False, edgecolor='gold', lw=2,
                           label='$\\mathrm{Cov}[\\mathcal{D}_i, \\mathcal{D}_j]$'))

    ax.add_patch(Rectangle((N-0.5, -0.5), 1, N,
                           fill=False, edgecolor='deepskyblue', lw=2,
                           label='$\\mathrm{Cov}[\\mathcal{D}_i, C_\\ell]$'))
    ax.add_patch(Rectangle((-0.5, N-0.5), N, 1,
                           fill=False, edgecolor='deepskyblue', lw=2))

    ax.add_patch(Rectangle((N-0.5, N-0.5), 1, 1,
                           fill=False, edgecolor='blueviolet', lw=2,
                           label='$\\mathrm{Cov}[C_\\ell, C_\\ell]$'))

    ax.set_title(f'Correlation matrix ($\\ell={ell}$, $f_{{\\rm sky}}={f_sky}$)')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.tick_params(which='major', direction='in', length=5, width=0.8, top=True, right=True)
    ax.tick_params(which='minor', direction='in', length=2, width=0.6, top=True, right=True)

    plt.tight_layout()
    plt.savefig(f'cov/correlation_matrix_ell{ell}.pdf', format="pdf", bbox_inches="tight")
    plt.show()

    return fig

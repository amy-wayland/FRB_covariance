import numpy as np
import os

from covariance import build_covariance_matrix, plot_correlation_matrix, zz, nz

if __name__ == "__main__":

    # ---------------------------------------------------
    # FRB sample
    # ---------------------------------------------------
    rng = np.random.default_rng(42)
    N = 6   # Fiducial number of FRBs
    z_frb = rng.choice(zz, size=N, p=nz / np.sum(nz))
    z_frb = np.sort(z_frb)
    print("Sampled FRB redshifts:", z_frb)
    f_sky = 1

    # ---------------------------------------------------
    # Sky geometry  —  sample N points inside a spherical cap
    # of fractional area f_sky, centred on the north pole.
    #
    # ---------------------------------------------------
    cos_theta_max = 1.0 - 2.0 * f_sky
    cos_colat = rng.uniform(cos_theta_max, 1.0, N)   # uniform in solid angle
    phi = rng.uniform(0, 2 * np.pi, N)
    ra  = phi
    dec = np.pi / 2.0 - np.arccos(cos_colat)         # colatitude → declination

    cos_theta = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            cos_theta[i, j] = (
                np.sin(dec[i]) * np.sin(dec[j]) +
                np.cos(dec[i]) * np.cos(dec[j]) * np.cos(ra[i] - ra[j]))

    # ---------------------------------------------------
    # Multipoles to compute
    # ---------------------------------------------------
    ell_boundaries = np.unique(np.geomspace(10, 1000, 20)).astype(int) 
    #ell_boundaries = np.array([1000, 2000, 5000, 10000])
    delta_ell = np.diff(ell_boundaries)
    ell_centers = 0.5 * (ell_boundaries[:-1] + ell_boundaries[1:])

    # Build one joint covariance matrix over all ell bins.
    # delta_ell is passed as the mean bin width (Knox formula denominator).
    # For per-bin widths, pass a scalar representative value here.
    mean_delta_ell = int(np.round(np.mean(delta_ell)))

    print("=" * 50)
    print(f"Computing joint covariance for ell={ell_centers}, delta_ell={mean_delta_ell}")
    print("=" * 50)

    cov, corr = build_covariance_matrix(
        ell_centers,
        z_frb,
        cos_theta,
        f_sky=f_sky,
        Nchi=50,
        Nmu=40,
        delta_ell=mean_delta_ell,
    )

    os.makedirs("cov", exist_ok=True)
    save_file = "cov/covariance_matrix.npz"
    np.savez(
        save_file,
        cov=cov,
        corr=corr,
        z_frb=z_frb,
        ell_centers=ell_centers,
        cos_theta=cos_theta,
    )
    print(f"Saved {save_file}")

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------
    plot_correlation_matrix(corr, z_frb, ell_centers, f_sky=f_sky)

    # ---------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------
    N_frb = len(z_frb)
    N_ell = len(ell_centers)

    print("\nCorrelation coefficient summary:")
    print(f"  Max |r| DD block:    {np.max(np.abs(corr[:N_frb, :N_frb] - np.eye(N_frb))):.4f}")
    print(f"  Max |r| cross block: {np.max(np.abs(corr[:N_frb, N_frb:])):.4f}")
    print(f"  Cov[C_ell, C_ell] diagonal: {np.diag(cov[N_frb:, N_frb:])}")

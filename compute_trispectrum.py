import numpy as np
import pyccl as ccl
import argparse
import os
import sys

# --------------------------------------------------
# Location of the BFC covariance code
# --------------------------------------------------

bfc_dir = "/mnt/users/waylanda/FRB_covariance_project/FRB_covariance_bfc"
sys.path.insert(0, bfc_dir)

from core import cosmo, hmc, pE
from covariance import W_interp, chis, C_ell_DD

# --------------------------------------------------
# Global settings
# --------------------------------------------------

nside = 512
ell_max = 3 * nside - 1

# --------------------------------------------------
# Build the CCL DM tracer
# --------------------------------------------------

def dm_tracer(chi_min=1.0, n_chi=2048):
    """
    CCL Tracer carrying the FRB DM kernel W_D(chi).
    """
    chi_arr = np.linspace(chi_min, float(chis.max()), n_chi)
    w_arr = W_interp(chi_arr)
    tr = ccl.Tracer()
    tr.add_tracer(cosmo, kernel=(chi_arr, w_arr))
    return tr

# --------------------------------------------------
# Build the halo-model trispectrum
# --------------------------------------------------

def build_trispectrum(n_a, n_k, term="cNG", k_min=1e-4, k_max=1e3):
    """
    T(k1,k2,a) for the BFC electron profile.

    term: 'cNG' for the full 1h+2h+3h+4h sum, or '1h'/'2h'/'3h'/'4h' to
          isolate a single contribution.
    """
    # Select the required contribution
    builder = {"cNG": ccl.halos.halomod_Tk3D_cNG,
               "1h": ccl.halos.halomod_Tk3D_1h,
               "2h": ccl.halos.halomod_Tk3D_2h,
               "3h": ccl.halos.halomod_Tk3D_3h,
               "4h": ccl.halos.halomod_Tk3D_4h}[term]

    a_arr = np.linspace(1.0 / (1.0 + 2.0), 1.0, n_a)
    lk_arr = np.log(np.geomspace(k_min, k_max, n_k))
    return builder(cosmo, hmc, prof=pE, a_arr=a_arr, lk_arr=lk_arr)

# --------------------------------------------------
# Choose multipole sample points inside each bin
# --------------------------------------------------

def bin_nodes(edges, n_sub=6):
    """
    Multipole sample points inside each bin.
    
    The covariance integral is formally over every ell in the bin;
    evaluating the CCL trispectrum covariance at every ell would be
    expensive however, so we choose a smaller number of representative
    'nodes' inside each bin.
    """
    nodes = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ells = np.arange(lo, hi)
        if ells.size == 0:
            # uses lo itself if there are no integer multipoles between
            # lo and hi
            ells = np.array([lo])
        elif ells.size > n_sub:
            idx = np.unique(np.round(np.linspace(0, ells.size - 1,
                                                 n_sub)).astype(int))
            ells = ells[idx]
        nodes.append(ells.astype(float))
    # Complete set of multipoles to evaluate the covariance
    union = np.unique(np.concatenate(nodes))
    return nodes, union

# --------------------------------------------------
# Average the multipole-level covariance over bins
# --------------------------------------------------

def bin_average(cov_ll, union, nodes):
    """
    Weighted double average of cov_ll over each pair of bins;
    Eq. (73) of arXiv:2410.06962.
    """
    nb = len(nodes)
    out = np.zeros((nb, nb))
    idx = {l: i for i, l in enumerate(union)}

    for b1 in range(nb):
        i1 = [idx[l] for l in nodes[b1]]
        w1 = nodes[b1]
        for b2 in range(nb):
            i2 = [idx[l] for l in nodes[b2]]
            w2 = nodes[b2]
            sub = cov_ll[np.ix_(i1, i2)]
            if len(i1) > 1 and len(i2) > 1:
                num = np.trapezoid(np.trapezoid(sub * w2[None, :], nodes[b2],
                                                axis=1) * w1, nodes[b1])
                den = (np.trapezoid(w1, nodes[b1]) *
                       np.trapezoid(w2, nodes[b2]))
            else:
                num = float(np.sum(np.outer(w1, w2) * sub))
                den = float(np.sum(w1) * np.sum(w2))
            out[b1, b2] = num / den
    return out

# --------------------------------------------------
# Gaussian covariance
# --------------------------------------------------

def gaussian_block(edges, cl_b, f_sky=1.0):
    """
    Gaussian covariance from the Knox formula.
    """
    n_modes = np.array([np.sum(2 * np.arange(lo, hi) + 1)
                        for lo, hi in zip(edges[:-1], edges[1:])],
                       dtype=float)
    n_modes[n_modes == 0] = 1.0
    return np.diag(2.0 * cl_b**2 / (f_sky * n_modes)), n_modes


# --------------------------------------------------
# Perform the computation
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sub", type=int, default=6,
                   help="multipole samples per bin for the averaging")
    p.add_argument("--term", default="cNG",
                   choices=["cNG", "1h", "2h", "3h", "4h"],
                   help="full trispectrum, or a single halo term")
    p.add_argument("--kmin", type=float, default=1e-4)
    p.add_argument("--kmax", type=float, default=1e3)
    p.add_argument("--chi-min", type=float, default=1.0)
    p.add_argument("--out", default="cov/trispectrum_cov.npz")
    args = p.parse_args()

    # Numer of scale factor and k points for the trispectrum grid
    n_a, n_k = (16, 48)

    # Define the multipole bins
    edges = np.unique(np.geomspace(2, ell_max, 15).astype(int))
    centres = 0.5 * (edges[:-1] + edges[1:])
    nodes, union = bin_nodes(edges, n_sub=args.n_sub)

    # DM tracer
    tr = dm_tracer(chi_min=args.chi_min)

    # Halo-model trispectrum
    tkk = build_trispectrum(n_a, n_k, term=args.term,
                            k_min=args.kmin, k_max=args.kmax)

    # Evaluate the covariance at the individual multipole nodes
    print(f"\n Evaluating {args.term} covariance at every multipole pair...", flush=True)
    cov_ll = ccl.covariances.angular_cl_cov_cNG(
        cosmo, tr, tr, ell=union, t_of_kk_a=tkk,
        tracer3=tr, tracer4=tr, ell2=union,
        integration_method="spline")

    # Average the connected covariance over the multipole bins
    cov_ng = bin_average(np.asarray(cov_ll), union, nodes)

    # Calculate the power spectrum at the bin centres
    cl_b = np.array([C_ell_DD(l, Nchi=400) for l in centres])

    # Calculate the Gaussian covariance
    cov_g, n_modes = gaussian_block(edges, cl_b)

    # Total covariance
    cov_tot = cov_g + cov_ng
    s = np.sqrt(np.diag(cov_tot))
    corr = cov_tot / np.outer(s, s)

    print(f"{'ell':>8} {'n_modes':>9} {'C_ell':>12} {'Gauss':>12} "
          f"{'cNG':>12} {'cNG/Gauss':>10}")
    for b, l in enumerate(centres):
        print(f"{l:8.0f} {n_modes[b]:9.0f} {cl_b[b]:12.4e} "
              f"{cov_g[b, b]:12.4e} {cov_ng[b, b]:12.4e} "
              f"{cov_ng[b, b]/cov_g[b, b]:10.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, ell_edges=edges, ell_centres=centres, term=args.term,
             cl=cl_b, cov_gauss=cov_g, cov_cng=cov_ng, cov_total=cov_tot,
             corr=corr, n_modes=n_modes, chi_min=args.chi_min, 
             kmin=args.kmin, kmax=args.kmax)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

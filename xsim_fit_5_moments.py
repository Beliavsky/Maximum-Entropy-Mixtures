#!/usr/bin/env python3
"""
Simulate nsim times from a true two‐component Gaussian mixture, fit mixtures by
moment‐matching and EM (optionally initialized at the moment fit), compute average
log‐likelihood and cross‐entropy for each, and print an aligned table of parameters
(with w1>=w2), average loglik, and cross‐entropies. Prints N and whether moment fit
is used to initialize EM. Optionally plots, for each simulation, the true mixture
density, the EM‐fit density, and a Gaussian matching the sample mean and std.
"""
import numpy as np
from scipy.optimize import fsolve, least_squares
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Simulation parameters
nsim = 1     # number of Monte Carlo simulations
N    = 10**3 # observations per simulation

# True mixture parameters
w_true, mu1_true, mu2_true, s1_true, s2_true = 0.3, -0.7, 0.3, 3.0, 1.0

# Flags
use_mom_init = True    # if True, initialize EM at moment‐matched fit
do_plot      = True    # if True, plot densities for each simulation

def sample_true_mixture(N):
    u = np.random.rand(N)
    x = np.empty(N)
    mask = u < w_true
    x[mask]  = np.random.randn(mask.sum())*s1_true + mu1_true
    x[~mask] = np.random.randn((~mask).sum())*s2_true + mu2_true
    return x

def empirical_moments(x, k):
    return [np.mean(x**j) for j in range(1, k+1)]

def fit_by_moments(m1, m2, m3, m4, m5):
    def eqs(vars):
        w, mu1, mu2, s1, s2 = vars
        m1_1, m1_2 = mu1, mu2
        m2_1, m2_2 = mu1**2 + s1**2, mu2**2 + s2**2
        m3_1 = mu1**3 + 3*mu1*s1**2;  m3_2 = mu2**3 + 3*mu2*s2**2
        m4_1 = mu1**4 + 6*mu1**2*s1**2 + 3*s1**4
        m4_2 = mu2**4 + 6*mu2**2*s2**2 + 3*s2**4
        m5_1 = mu1**5 + 10*mu1**3*s1**2 + 15*mu1*s1**4
        m5_2 = mu2**5 + 10*mu2**3*s2**2 + 15*mu2*s2**4
        M1 = w*m1_1 + (1-w)*m1_2
        M2 = w*m2_1 + (1-w)*m2_2
        M3 = w*m3_1 + (1-w)*m3_2
        M4 = w*m4_1 + (1-w)*m4_2
        M5 = w*m5_1 + (1-w)*m5_2
        return [M1 - m1, M2 - m2, M3 - m3, M4 - m4, M5 - m5]

    guess = [0.5, -0.5, 0.5, 1.0, 1.0]
    sol, info, ier, _ = fsolve(eqs, guess, full_output=True)
    if ier != 1:
        def res(vars): return eqs(vars)
        lb, ub = [1e-6, -5, -5, 1e-6, 1e-6], [1-1e-6, 5, 5, 5, 5]
        sol = least_squares(res, guess, bounds=(lb, ub)).x
    w, mu1, mu2, s1, s2 = sol
    return w, mu1, mu2, abs(s1), abs(s2)

def fit_by_em(x, init_params=None):
    if init_params is not None and use_mom_init:
        w, mu1, mu2, s1, s2 = init_params
        weights_init = np.array([w, 1-w])
        means_init   = np.array([[mu1], [mu2]])
        precisions_init = np.array([[[1/s1**2]], [[1/s2**2]]])
        gm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            init_params="random",
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=0
        )
    else:
        gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gm.fit(x.reshape(-1,1))
    w_em = gm.weights_[0]
    mu1_em, mu2_em = gm.means_.flatten()
    s1_em = np.sqrt(gm.covariances_[0].flatten()[0])
    s2_em = np.sqrt(gm.covariances_[1].flatten()[0])
    return w_em, mu1_em, mu2_em, s1_em, s2_em

def mixture_pdf(x, params):
    w, mu1, mu2, s1, s2 = params
    p1 = w/(np.sqrt(2*np.pi)*s1)*np.exp(-0.5*((x-mu1)/s1)**2)
    p2 = (1-w)/(np.sqrt(2*np.pi)*s2)*np.exp(-0.5*((x-mu2)/s2)**2)
    return p1 + p2

def cross_entropy_and_ll(x, params):
    pdf_vals = mixture_pdf(x, params)
    logpdf = np.log(pdf_vals + 1e-300)
    avg_ll = np.mean(logpdf)
    return avg_ll, -avg_ll

def sort_params(params):
    w, mu1, mu2, s1, s2 = params
    w1, w2 = w, 1-w
    if w1 < w2:
        return (w2, w1, mu2, mu1, s2, s1)
    else:
        return (w1, w2, mu1, mu2, s1, s2)

def main():
    np.random.seed(0)
    print(f"N = {N}")
    print(f"Use moment fit to initialize EM: {use_mom_init}\n")

    hdr = f"{'Model':<8}{'w1':>8}{'w2':>8}{'mu1':>10}{'mu2':>10}{'s1':>8}{'s2':>8}{'AvgLL':>12}{'CrossEnt':>12}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    sums = {'True':[0,0], 'MomFit':[0,0], 'EMFit':[0,0]}

    xs = np.linspace(-5, 5, 1000)
    for _ in range(nsim):
        x = sample_true_mixture(N)
        m1, m2, m3, m4, m5 = empirical_moments(x, 5)

        p_true = (w_true, mu1_true, mu2_true, s1_true, s2_true)
        p_mom  = fit_by_moments(m1, m2, m3, m4, m5)
        p_em   = fit_by_em(x, init_params=p_mom)

        ll_true, ce_true = cross_entropy_and_ll(x, p_true)
        ll_mom,  ce_mom  = cross_entropy_and_ll(x, p_mom)
        ll_em,   ce_em   = cross_entropy_and_ll(x, p_em)

        for label, params, ll, ce in [
            ('True',   p_true, ll_true,  ce_true),
            ('MomFit', p_mom,  ll_mom,   ce_mom),
            ('EMFit',  p_em,   ll_em,    ce_em),
        ]:
            w1, w2, mu1, mu2, s1, s2 = sort_params(params)
            print(f"{label:<8}{w1:8.4f}{w2:8.4f}{mu1:10.4f}{mu2:10.4f}"
                  f"{s1:8.4f}{s2:8.4f}{ll:12.6f}{ce:12.6f}")
            sums[label][0] += ll
            sums[label][1] += ce

        print(sep)

        if do_plot:
            # true PDF
            pdf_true = mixture_pdf(xs, p_true)
            # EM fit PDF
            pdf_em   = mixture_pdf(xs, p_em)
            # normal matching sample mean/std
            m_data, s_data = np.mean(x), np.std(x)
            pdf_norm = (1/np.sqrt(2*np.pi*s_data**2)) * np.exp(-0.5*((xs-m_data)/s_data)**2)

            plt.figure()
            plt.plot(xs, pdf_true, label="True mixture")
            plt.plot(xs, pdf_em,   label="EM fit mixture")
            plt.plot(xs, pdf_norm, '--', label="Normal match mean/std")
            plt.title("Density comparison")
            plt.xlabel("x")
            plt.ylabel("Density")
            plt.legend()
            plt.show()

    # averages
    print("\nAverage over simulations:")
    avg_hdr = f"{'Model':<8}{'AvgLL':>12}{'CrossEnt':>12}"
    print(avg_hdr)
    print("-"*len(avg_hdr))
    for label in ('True','MomFit','EMFit'):
        avg_ll = sums[label][0]/nsim
        avg_ce = sums[label][1]/nsim
        print(f"{label:<8}{avg_ll:12.6f}{avg_ce:12.6f}")

if __name__ == "__main__":
    main()

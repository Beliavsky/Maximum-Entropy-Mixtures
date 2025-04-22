#!/usr/bin/env python3
"""
Scan a grid of target third- and fourth-moment values and for each compute the
two-component Gaussian mixture (overall mean=0, variance=1) that matches m3 and m4
and minimizes max(s2/s1, s1/s2). Print a table of m3, m4, mixture weights,
component means, standard deviations, and the minimized ratio. Optionally plot,
for each m4, the densities across m3 values together with the standard normal.
"""
import numpy as np
from scipy.optimize import fsolve, minimize_scalar
import matplotlib.pyplot as plt

# Scanning parameters for m3 and m4:
m3_min = 0.0    # minimum third moment
m3_inc = 0.5     # third-moment increment
nm3   = 3        # number of m3 values to try
m4_min = 9.0    # minimum fourth moment (must exceed 3)
m4_inc = 0.5     # fourth-moment increment
nm4   = 1        # number of m4 values to try

# Plot flag: set to True to display density plots per m4
do_plot = True

# Large finite penalty to avoid invalid objective calls
PENALTY = 1e6

def solve_params(w: float, m3_target: float, m4_target: float):
    """
    Given weight w, solve for mu1, mu2, s1, s2 so that the mixture
    has mean=0, var=1, third moment=m3_target, fourth=m4_target.
    Returns (mu1, mu2, s1, s2) or None if infeasible.
    """
    def eqs(vars):
        mu1, mu2, s1, s2 = vars
        eq1 = w*mu1 + (1-w)*mu2
        eq2 = w*(s1**2 + mu1**2) + (1-w)*(s2**2 + mu2**2) - 1.0
        eq3 = w*(mu1**3 + 3*mu1*s1**2) + (1-w)*(mu2**3 + 3*mu2*s2**2) - m3_target
        eq4 = (
            w*(mu1**4 + 6*mu1**2*s1**2 + 3*s1**4) +
            (1-w)*(mu2**4 + 6*mu2**2*s2**2 + 3*s2**4)
            - m4_target
        )
        return [eq1, eq2, eq3, eq4]

    guess = [-0.1, 0.1, 0.9, 1.1]
    try:
        sol, infodict, ier, _ = fsolve(eqs, guess, full_output=True)
    except Exception:
        return None
    if ier != 1 or np.isnan(sol).any():
        return None
    mu1, mu2, s1, s2 = sol
    if s1 <= 0 or s2 <= 0:
        return None
    return mu1, mu2, s1, s2

def ratio_for_w(w: float, m3: float, m4: float) -> float:
    """
    Return max(s2/s1, s1/s2) for the mixture at weight w matching (m3,m4),
    or a large finite penalty if infeasible.
    """
    sol = solve_params(w, m3, m4)
    if sol is None:
        return PENALTY
    _, _, s1, s2 = sol
    ratio = max(s2/s1, s1/s2)
    return float(ratio) if np.isfinite(ratio) else PENALTY

def find_optimal_mixture(m3: float, m4: float):
    """
    Find the mixture weight w that minimizes max(s2/s1, s1/s2) for given (m3,m4).
    Returns (w, mu1, mu2, s1, s2, ratio) or None if infeasible.
    """
    eps = 1e-6
    res = minimize_scalar(
        lambda w: ratio_for_w(w, m3, m4),
        bounds=(eps, 1-eps),
        method='bounded',
        options={'xatol':1e-6}
    )
    w_opt = res.x
    sol = solve_params(w_opt, m3, m4)
    if sol is None:
        return None
    mu1, mu2, s1, s2 = sol
    R = max(s2/s1, s1/s2)
    return w_opt, mu1, mu2, s1, s2, R

def mixture_pdf(x: np.ndarray, w: float, mu1: float, mu2: float, s1: float, s2: float) -> np.ndarray:
    """
    Density of the two-component Gaussian mixture at x with means mu1, mu2
    and std devs s1, s2.
    """
    norm1 = np.exp(-((x-mu1)**2)/(2*s1*s1)) / (np.sqrt(2*np.pi)*s1)
    norm2 = np.exp(-((x-mu2)**2)/(2*s2*s2)) / (np.sqrt(2*np.pi)*s2)
    return w*norm1 + (1-w)*norm2

def main():
    # Print table header
    print("{:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "m3", "m4", "w1", "w2", "mu1", "mu2", "s1", "s2", "ratio"
    ))
    print("-"*74)

    xs = np.linspace(-5, 5, 1000)
    pdf_norm = np.exp(-xs**2/2) / np.sqrt(2*np.pi)

    for j in range(nm4):
        m4 = m4_min + j*m4_inc
        if m4 <= 3.0:
            continue

        # collect plot data for this m4
        plot_data = []

        for i in range(nm3):
            m3 = m3_min + i*m3_inc
            result = find_optimal_mixture(m3, m4)
            if result is None:
                print(f"{m3:6.3f} {m4:6.3f}    skipped")
            else:
                w, mu1, mu2, s1, s2, R = result
                print(f"{m3:6.3f} {m4:6.3f} {w:8.5f} {1-w:8.5f} "
                      f"{mu1:8.5f} {mu2:8.5f} {s1:8.5f} {s2:8.5f} {R:8.5f}")
                if do_plot:
                    pdf_mix = mixture_pdf(xs, w, mu1, mu2, s1, s2)
                    plot_data.append((pdf_mix, f"m3={m3:.3f}"))

        if do_plot and plot_data:
            plt.figure()
            for pdf_mix, label in plot_data:
                plt.plot(xs, pdf_mix, label=label)
            plt.plot(xs, pdf_norm, 'k--', label="Standard normal")
            plt.title(f"Densities for m4 = {m4:.3f}")
            plt.xlabel("x")
            plt.ylabel("Density")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    main()

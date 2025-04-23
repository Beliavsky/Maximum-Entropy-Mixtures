#!/usr/bin/env python3
"""
Fit a zero‐mean, unit‐variance two‐component Gaussian mixture to match
target third moment m3_target, fourth moment m4_target, and average absolute
deviation abs_m1_target = E[|X|]. Uses multiple initial guesses and fallback
strategies. Prints the resulting mixture parameters in a table with each
component on its own line and optionally plots the fitted density vs standard normal.
"""
import numpy as np
from scipy.optimize import fsolve, least_squares
from scipy.special import erf
import matplotlib.pyplot as plt

# Plot flag
do_plot = True

def abs_moment(mu: float, sigma: float) -> float:
    """
    E[|Y|] for Y ~ N(mu, sigma^2).
    = sigma*sqrt(2/pi)*exp(-mu^2/(2*sigma^2)) + |mu|*erf(|mu|/(sigma*sqrt(2))).
    """
    a = abs(mu)/(sigma*np.sqrt(2))
    return sigma*np.sqrt(2/np.pi)*np.exp(-0.5*(mu/sigma)**2) + abs(mu)*erf(a)

def fit_normal_mixture_abs(m3_target: float, m4_target: float, abs_m1_target: float):
    """
    Solve for (w, mu1, mu2, s1, s2) such that
      E[X]=0, Var[X]=1,
      E[X^3]=m3_target, E[X^4]=m4_target,
      E[|X|]=abs_m1_target.
    """
    def equations(vars):
        w, mu1, mu2, s1, s2 = vars
        # component raw moments
        m1_1 = mu1
        m2_1 = mu1**2 + s1**2
        m3_1 = mu1**3 + 3*mu1*s1**2
        m4_1 = mu1**4 + 6*mu1**2*s1**2 + 3*s1**4
        m1_2 = mu2
        m2_2 = mu2**2 + s2**2
        m3_2 = mu2**3 + 3*mu2*s2**2
        m4_2 = mu2**4 + 6*mu2**2*s2**2 + 3*s2**4
        # mixture moments
        M1 = w*m1_1 + (1-w)*m1_2
        M2 = w*m2_1 + (1-w)*m2_2
        M3 = w*m3_1 + (1-w)*m3_2
        M4 = w*m4_1 + (1-w)*m4_2
        # average absolute
        A1 = abs_moment(mu1, s1)
        A2 = abs_moment(mu2, s2)
        MA = w*A1 + (1-w)*A2
        return [
            M1,               # = 0
            M2 - 1.0,         # = 0
            M3 - m3_target,   # = 0
            M4 - m4_target,   # = 0
            MA - abs_m1_target
        ]

    def is_valid(sol):
        if sol is None or any(np.isnan(sol)):
            return False
        w, mu1, mu2, s1, s2 = sol
        return 0 < w < 1 and s1 > 0 and s2 > 0

    # build initial guesses
    skew_est = np.sign(m3_target) * abs(m3_target)**(1/3) if m3_target != 0 else 0.1
    guesses = [
        [0.5, -skew_est, skew_est, 0.8, 1.2],
        [0.5,  skew_est, -skew_est, 1.2, 0.8],
        [0.3, -0.5*skew_est, 1.5*skew_est, 0.5, 1.5],
        [0.7, -1.5*skew_est, 0.5*skew_est, 1.5, 0.5],
    ]

    # try fsolve
    for guess in guesses:
        try:
            sol, infodict, ier, _ = fsolve(equations, guess, full_output=True)
        except Exception:
            continue
        if ier == 1 and is_valid(sol):
            return tuple(sol)

    # fallback: least_squares with bounds
    def residuals(vars):
        return equations(vars)
    lb = [1e-6, -5, -5, 1e-6, 1e-6]
    ub = [1-1e-6,  5,  5,  5,   5]
    for guess in guesses:
        try:
            res = least_squares(residuals, guess, bounds=(lb, ub))
        except Exception:
            continue
        sol = res.x
        if res.success and is_valid(sol):
            return tuple(sol)

    raise RuntimeError("Mixture fit did not converge or produced invalid parameters")

def mixture_pdf(x: np.ndarray, w: float, mu1: float, mu2: float, s1: float, s2: float) -> np.ndarray:
    """Density of the two‐component Gaussian mixture at x."""
    norm1 = np.exp(-((x-mu1)**2)/(2*s1*s1)) / (np.sqrt(2*np.pi)*s1)
    norm2 = np.exp(-((x-mu2)**2)/(2*s2*s2)) / (np.sqrt(2*np.pi)*s2)
    return w*norm1 + (1-w)*norm2

def main():
    # Example target
    m3 = 0.5
    m4 = 4.0
    abs_m1 = 0.8

    print(f"Targets: m3={m3}, m4={m4}, E|X|={abs_m1}\n")
    w, mu1, mu2, s1, s2 = fit_normal_mixture_abs(m3, m4, abs_m1)

    # Print component table
    print("{:<10}{:>10}{:>10}{:>10}".format("Component","Weight","Mean","StdDev"))
    print("-"*40)
    print("{:<10}{:>10.6f}{:>10.6f}{:>10.6f}".format("1", w,    mu1, s1))
    print("{:<10}{:>10.6f}{:>10.6f}{:>10.6f}".format("2", 1-w, mu2, s2))

    if do_plot:
        xs = np.linspace(-5, 5, 1000)
        pdf_mix = mixture_pdf(xs, w, mu1, mu2, s1, s2)
        pdf_norm = np.exp(-xs**2/2)/np.sqrt(2*np.pi)

        plt.figure()
        plt.plot(xs, pdf_mix, label="Fitted mixture")
        plt.plot(xs, pdf_norm, 'k--', label="Std normal")
        plt.title("Mixture vs. standard normal")
        plt.xlabel("x"); plt.ylabel("Density")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()

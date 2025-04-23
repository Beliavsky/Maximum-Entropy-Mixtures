"""
Fit a zero‐mean, unit‐variance two‐component Gaussian mixture to match
target third, fourth, and fifth moments (m3_target, m4_target, m5_target),
using multiple initial guesses and fallback strategies. Print the resulting
mixture parameters in a table with each component on its own line and optionally
plot the fitted density against the standard normal.
"""
import numpy as np
from scipy.optimize import fsolve, least_squares
import matplotlib.pyplot as plt

# Plot flag: set to True to display the fitted density
do_plot = True

def fit_normal_mixture(m3_target: float, m4_target: float, m5_target: float):
    """
    Solve for mixture parameters (w, mu1, mu2, s1, s2) such that
      E[X]=0, Var[X]=1, E[X^3]=m3_target, E[X^4]=m4_target, E[X^5]=m5_target.
    Tries multiple initial guesses and both fsolve and least_squares.
    Returns (w, mu1, mu2, s1, s2).
    Raises RuntimeError if no valid solution found.
    """
    def moments_eq(vars):
        w, mu1, mu2, s1, s2 = vars
        # moments for component 1
        m1_1 = mu1
        m2_1 = mu1**2 + s1**2
        m3_1 = mu1**3 + 3*mu1*s1**2
        m4_1 = mu1**4 + 6*mu1**2*s1**2 + 3*s1**4
        m5_1 = mu1**5 + 10*mu1**3*s1**2 + 15*mu1*s1**4
        # moments for component 2
        m1_2 = mu2
        m2_2 = mu2**2 + s2**2
        m3_2 = mu2**3 + 3*mu2*s2**2
        m4_2 = mu2**4 + 6*mu2**2*s2**2 + 3*s2**4
        m5_2 = mu2**5 + 10*mu2**3*s2**2 + 15*mu2*s2**4

        # mixture moments
        M1 = w*m1_1 + (1-w)*m1_2
        M2 = w*m2_1 + (1-w)*m2_2
        M3 = w*m3_1 + (1-w)*m3_2
        M4 = w*m4_1 + (1-w)*m4_2
        M5 = w*m5_1 + (1-w)*m5_2

        return [
            M1,                # = 0
            M2 - 1.0,          # = 0
            M3 - m3_target,    # = 0
            M4 - m4_target,    # = 0
            M5 - m5_target     # = 0
        ]

    def is_valid(sol):
        if sol is None or any(np.isnan(sol)):
            return False
        w, mu1, mu2, s1, s2 = sol
        return (0 < w < 1) and (s1 > 0) and (s2 > 0)

    # build initial guesses
    skew_est = np.sign(m3_target) * (abs(m3_target)**(1/3)) if m3_target != 0 else 0.1
    guesses = [
        [0.5, -skew_est, skew_est, 0.8, 1.2],
        [0.5,  skew_est, -skew_est, 1.2, 0.8],
        [0.3, -0.5*skew_est, 1.5*skew_est, 0.5, 1.5],
        [0.7, -1.5*skew_est, 0.5*skew_est, 1.5, 0.5]
    ]

    # try fsolve
    for guess in guesses:
        try:
            sol, infodict, ier, _ = fsolve(moments_eq, guess, full_output=True)
        except Exception:
            continue
        if ier == 1 and is_valid(sol):
            return tuple(sol)

    # fallback: least_squares with bounds
    def residuals(vars):
        return moments_eq(vars)
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
    # Example target moments
    m3 = 0.5
    m4 = 4.0
    m5 = 5.0

    print(f"Target moments: m3={m3}, m4={m4}, m5={m5}\n")
    w, mu1, mu2, s1, s2 = fit_normal_mixture(m3, m4, m5)

    # Print table of component parameters
    print("{:<10} {:>10} {:>10} {:>10}".format("Component", "Weight", "Mean", "StdDev"))
    print("-"*44)
    print("{:<10} {:>10.6f} {:>10.6f} {:>10.6f}".format("1", w, mu1, s1))
    print("{:<10} {:>10.6f} {:>10.6f} {:>10.6f}".format("2", 1-w, mu2, s2))

    if do_plot:
        xs = np.linspace(-5, 5, 1000)
        pdf_mix = mixture_pdf(xs, w, mu1, mu2, s1, s2)
        pdf_norm = np.exp(-xs**2/2) / np.sqrt(2*np.pi)

        plt.figure()
        plt.plot(xs, pdf_mix, label="Fitted mixture")
        plt.plot(xs, pdf_norm, 'k--', label="Standard normal")
        plt.title(f"Fitted mixture vs standard normal")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()

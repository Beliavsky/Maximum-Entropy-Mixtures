"""
Scan a range of fourth‐moment values and for each compute the minimal feasible
ratio s2/s1 for a zero-mean two-component Gaussian mixture (variance=1),
then output m4, mixture weights, component scales, ratio s2/s1, and Shannon entropy.
Finally plot all mixture densities together with the standard normal.
"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Parameters for scanning m4 values:
m4_min = 10.0    # minimum fourth moment (must exceed 3)
m4_inc = 3.0    # increment between successive m4 values
nm4   = 1      # number of m4 values to try

def mixture_pdf(x: np.ndarray, w: float, s1: float, s2: float) -> np.ndarray:
    norm1 = np.exp(-x*x/(2.0*s1*s1)) / (np.sqrt(2.0*np.pi) * s1)
    norm2 = np.exp(-x*x/(2.0*s2*s2)) / (np.sqrt(2.0*np.pi) * s2)
    return w * norm1 + (1.0 - w) * norm2

def entropy_mixture(w: float, s1: float, s2: float) -> float:
    def integrand(x):
        p = mixture_pdf(x, w, s1, s2)
        return -p * np.log(p) if p > 0.0 else 0.0
    H_half, _ = quad(integrand, 0.0, np.inf, limit=200)
    return 2.0 * H_half

def min_ratio_mixture(m4: float):
    if m4 <= 3.0:
        raise ValueError("m4 must exceed 3")
    phi = m4/3.0 - 1.0
    a = phi + np.sqrt(phi * (phi + 1.0))
    w1 = a*a / (a*a + phi)
    w2 = 1.0 - w1
    s1 = np.sqrt((a - phi) / a)
    s2 = np.sqrt(1.0 + a)
    R = s2 / s1
    return R, w1, w2, s1, s2

def main():
    # Table header with s2/s1 after s2
    print("{:>6}   {:>8}   {:>8}   {:>8}   {:>8}   {:>8}   {:>12}".format(
        "m4", "w1", "w2", "s1", "s2", "s2/s1", "Entropy"
    ))
    print("-" * 76)

    xs = np.linspace(-5, 5, 1000)
    pdf_norm = np.exp(-xs**2 / 2.0) / np.sqrt(2.0 * np.pi)

    plt.figure()
    for i in range(nm4):
        m4 = m4_min + i * m4_inc
        if m4 <= 3.0:
            continue
        try:
            R, w1, w2, s1, s2 = min_ratio_mixture(m4)
            H = entropy_mixture(w1, s1, s2)
            print("{:6.3f}   {:8.5f}   {:8.5f}   {:8.5f}   {:8.5f}   {:8.5f}   {:12.6f}".format(
                m4, w1, w2, s1, s2, R, H
            ))
            pdf_mix = mixture_pdf(xs, w1, s1, s2)
            plt.plot(xs, pdf_mix, label=f"m4={m4:.1f}")
        except ValueError:
            pass

    # plot standard normal
    plt.plot(xs, pdf_norm, 'k--', label="Standard normal")
    plt.title("Mixture densities for various m4")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

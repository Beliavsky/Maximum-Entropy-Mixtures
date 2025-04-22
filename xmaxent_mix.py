"""
Scan a range of target fourth‐moment values and for each compute the maximum‐entropy
zero‐mean two‐component Gaussian mixture (variance 1, fourth moment m4) subject to
an upper bound on component scales (s_max), then print a table of m4, mixture weights,
component standard deviations, and entropy.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, brentq

# User-set parameters for scanning m4 values and scale cap:
m4_min = 3.1    # minimum fourth moment (must exceed 3)
m4_inc = 1.0    # increment between successive m4 values
nm4   = 10     # number of m4 values to try
s_max = 3.0    # maximum allowed standard deviation for any component
sqrt_2_pi = np.sqrt(2.0*np.pi)

print("s_max:", s_max, end="\n\n")

def get_scales(w: float, m4: float) -> tuple[float, float]:
    """
    Solve for component std devs s1,s2 given weight w and target fourth moment m4.
    Returns (s1, s2) or (None, None) if infeasible.
    """
    phi = m4/3.0 - 1.0
    try:
        s1_sq = 1.0 - np.sqrt((1.0 - w) * phi / w)
        s2_sq = 1.0 + np.sqrt(w * phi / (1.0 - w))
    except (ZeroDivisionError, ValueError):
        return None, None
    if s1_sq <= 0.0 or s2_sq <= 0.0:
        return None, None
    return np.sqrt(s1_sq), np.sqrt(s2_sq)

def mixture_pdf(x: float, w: float, s1: float, s2: float) -> float:
    """Mixture density at x for weights w and std devs s1, s2."""
    norm1 = np.exp(-x*x/(2.0*s1*s1)) / (sqrt_2_pi * s1)
    norm2 = np.exp(-x*x/(2.0*s2*s2)) / (sqrt_2_pi * s2)
    return w * norm1 + (1.0 - w) * norm2

def entropy(w: float, m4: float) -> float:
    """
    Shannon entropy H = -∫ p log p dx of the mixture,
    computed over [0,∞) and doubled.
    """
    s1, s2 = get_scales(w, m4)
    if s1 is None:
        return -np.inf

    def integrand(x):
        p = mixture_pdf(x, w, s1, s2)
        return -p * np.log(p) if p > 0.0 else 0.0

    H_half, _ = quad(integrand, 0.0, np.inf, limit=200)
    return 2.0 * H_half

def find_maxent_mixture(m4: float) -> tuple[float, float, float, float]:
    """
    Find (w_opt, s1, s2, H_max) maximizing H subject to:
      - variance = 1, fourth moment = m4,
      - 0 < w < 1, and s1, s2 <= s_max.
    Raises ValueError if no feasible solution.
    """
    if m4 <= 3.0:
        raise ValueError("m4 must exceed 3.")
    w_min = (m4 - 3.0) / m4
    eps = 1e-8

    # find upper bound w_max so that s2(w_max) = s_max
    def s2_minus_max(w):
        s1_, s2_ = get_scales(w, m4)
        return (s2_ - s_max) if s2_ is not None else -s_max

    try:
        w_max = brentq(s2_minus_max, w_min + eps, 1.0 - eps)
    except ValueError:
        # if at w_min s2 > s_max, no feasible mixture
        s1_test, s2_test = get_scales(w_min + eps, m4)
        if s2_test is not None and s2_test > s_max:
            raise ValueError("No mixture with s2 <= {}".format(s_max))
        # otherwise s2 never reaches s_max in domain -> no need to cap
        w_max = 1.0 - eps

    # optimize entropy over w in [w_min, w_max]
    res = minimize_scalar(
        lambda w: -entropy(w, m4),
        bounds=(w_min + eps, w_max),
        method='bounded',
        options={'xatol': 1e-6}
    )
    w_opt = res.x
    s1_opt, s2_opt = get_scales(w_opt, m4)
    H_max = entropy(w_opt, m4)
    return w_opt, s1_opt, s2_opt, H_max

def main():
    # Table header
    print("{:>6}   {:>8}   {:>8}   {:>8}   {:>8}   {:>12}".format(
        "m4", "w1", "w2", "s1", "s2", "Entropy"
    ))
    print("-" * 64)

    for i in range(nm4):
        m4 = m4_min + i * m4_inc
        if m4 <= 3.0:
            print("{:6.3f}      skipped (m4 must exceed 3)".format(m4))
            continue
        try:
            w1, s1, s2, H = find_maxent_mixture(m4)
            w2 = 1.0 - w1
            print("{:6.3f}   {:8.5f}   {:8.5f}   {:8.5f}   {:8.5f}   {:12.6f}".format(
                m4, w1, w2, s1, s2, H
            ))
        except Exception as e:
            print("{:6.3f}      error: {}".format(m4, e))

if __name__ == "__main__":
    main()

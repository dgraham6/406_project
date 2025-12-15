import numpy as np
from scipy.stats import norm

def hypothesis_test(test_statistic, std_dev, two_sided=True):
    """
    Performs a z-test for a given test statistic.

    Parameters:
        test_statistic (float): Observed test statistic (e.g. difference in MAE)
        std_dev (float): Standard deviation of the test statistic
        two_sided (bool): Whether to perform a two-sided test

    Returns:
        z_score (float)
        p_value (float)
    """
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive.")

    z_score = test_statistic / std_dev

    if two_sided:
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
    else:
        p_value = 1 - norm.cdf(z_score)

    return z_score, p_value

# Example: difference in MAE
test_stat = abs(0.00917 - 0.009805)    # observed difference (Replace with difference in MAE or RMSE)
std_dev =  0.00082          # estimated SD from simulations

z, p = hypothesis_test(test_stat, std_dev, two_sided=False)

print(f"Z-score: {z:.6f}")
print(f"P-value: {p:.6f}")

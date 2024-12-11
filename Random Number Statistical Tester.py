import numpy as np
from scipy.stats import kstest, chi2

class RandomNumberTester:
    @staticmethod
    def kolmogorov_smirnov_test(random_numbers, alpha=0.05):
        """
        Perform the Kolmogorov-Smirnov test for uniformity.
        """
        n = len(random_numbers)
        random_numbers.sort()

        # Compute D+ and D-
        D_plus = [(i + 1) / n - random_numbers[i] for i in range(n)]
        D_minus = [random_numbers[i] - i / n for i in range(n)]

        # D = max(D+, D-)
        D = max(max(D_plus), max(D_minus))

        # Critical value
        critical_value = 1.36 / np.sqrt(n)  # Approximation for alpha = 0.05

        # Decision
        decision = "Accept H0" if D < critical_value else "Reject H0"

        return {
            "D": D,
            "Critical Value": critical_value,
            "Decision": decision
        }

    @staticmethod
    def chi_square_test(random_numbers, intervals=10, alpha=0.05):
        """
        Perform the Chi-Square test for uniformity.
        """
        n = len(random_numbers)
        expected_count = n / intervals

        # Calculate observed counts
        observed_counts, _ = np.histogram(random_numbers, bins=np.linspace(0, 1, intervals + 1))

        # Chi-Square statistic
        chi_square_stat = sum(((observed_counts[i] - expected_count) ** 2) / expected_count for i in range(intervals))

        # Critical value
        critical_value = chi2.ppf(1 - alpha, intervals - 1)

        # Decision
        decision = "Accept H0" if chi_square_stat < critical_value else "Reject H0"

        return {
            "Chi-Square Statistic": chi_square_stat,
            "Critical Value": critical_value,
            "Decision": decision
        }

    @staticmethod
    def autocorrelation_test(random_numbers, lag=1, alpha=0.05):
        """
        Perform the autocorrelation test for independence.
        """
        n = len(random_numbers)
        m = (n - lag - 1) // lag

        # Compute autocorrelation estimate
        auto_corr = sum(random_numbers[i] * random_numbers[i + lag] for i in range(m)) / m - 0.25

        # Compute standard deviation
        std_dev = np.sqrt((13 * m + 7) / (12 * m))

        # Z-value
        z_value = auto_corr / std_dev

        # Critical Z value (two-tailed)
        critical_z = 1.96  # For alpha = 0.05

        # Decision
        decision = "Accept H0" if -critical_z <= z_value <= critical_z else "Reject H0"

        return {
            "Autocorrelation": auto_corr,
            "Z-Value": z_value,
            "Critical Z": critical_z,
            "Decision": decision
        }

# Example Usage
if __name__ == "__main__":
    # Generate random numbers
    np.random.seed(15)
    random_sequence = np.random.random(100)

    # Initialize tester
    tester = RandomNumberTester()

    # Kolmogorov-Smirnov Test
    ks_result = tester.kolmogorov_smirnov_test(random_sequence)
    print("Kolmogorov-Smirnov Test:", ks_result)

    # Chi-Square Test
    chi_square_result = tester.chi_square_test(random_sequence)
    print("Chi-Square Test:", chi_square_result)

    # Autocorrelation Test
    auto_corr_result = tester.autocorrelation_test(random_sequence, lag=3)
    print("Autocorrelation Test:", auto_corr_result)

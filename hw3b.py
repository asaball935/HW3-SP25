import math


def gamma_fn(x):
    """Approximate the Gamma function using factorial or Stirling's approximation."""
    if x == int(x) and x > 0:
        product = 1
        for i in range(1, int(x)):
            product *= i
        return product
    else:
        # Stirling's approximation for real-valued gamma function
        return math.sqrt(2 * math.pi / x) * (x / math.e) ** x


def t_dist_cdf(df, limit):
    """Estimate cumulative probability P(T ≤ z) using trapezoidal integration."""
    # Normalizing constant
    const = gamma_fn((df + 1) / 2) / (math.sqrt(df * math.pi) * gamma_fn(df / 2))

    def t_pdf(u):
        return (1 + (u ** 2) / df) ** (-(df + 1) / 2)

    def trapezoid_area(f, x_start, x_end, steps=1000):
        width = (x_end - x_start) / steps
        total = 0.5 * (f(x_start) + f(x_end))
        for j in range(1, steps):
            total += f(x_start + j * width)
        return total * width

    return const * trapezoid_area(t_pdf, -100, limit)


def run_t_distribution_tool():
    """Main interactive loop to compute t-distribution probabilities."""
    print("T-Distribution CDF Calculator")
    while True:
        try:
            df = int(input("Degrees of freedom (positive integer): "))
            t_val = float(input("z (T-value): "))

            prob = t_dist_cdf(df, t_val)
            print(f"P(T ≤ {t_val} | df = {df}) ≈ {prob:.4f}")
        except ValueError:
            print("Invalid input. Please enter numeric values only.")

        repeat = input("Run another calculation? (y/n): ").strip().lower()
        if repeat not in {"y", "yes"}:
            break


if __name__ == "__main__":
    run_t_distribution_tool()

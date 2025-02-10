import scipy.integrate as integrate
import scipy.special as special
import numpy as np


def gamma_function(alpha):
    """
    Compute the Gamma function for a given alpha.
    This is a mathematical function that generalizes the factorial function.
    """
    return special.gamma(alpha)


def t_distribution_probability(m, z):
    """
    Compute the cumulative distribution function (CDF) for a t-distribution with m degrees of freedom.
    This function calculates the probability P(T ≤ z) for a given t-distribution.
    """
    # Gamma constant factor
    Km = gamma_function(0.5 * m + 0.5) / (np.sqrt(m * np.pi) * gamma_function(0.5 * m))

    def integrand(u):
        """
        The integrand function for the t-distribution probability calculation.
        This represents the probability density function (PDF) for the t-distribution.
        """
        return (1 + (u ** 2) / m) ** (-(m + 1) / 2)

    # Perform the numerical integration to calculate the CDF from -∞ to z
    result, _ = integrate.quad(integrand, -np.inf, z)

    # Return the final probability scaled by the constant Km
    return Km * result


def prompt_user_input():
    """
    Function to handle user input, validate, and ensure proper numeric input.
    """
    while True:
        try:
            m = int(input("Enter the degrees of freedom (m): "))
            z = float(input("Enter the value of z: "))
            if m <= 0:
                print("Degrees of freedom (m) should be a positive integer.")
                continue
            return m, z
        except ValueError:
            print("Invalid input. Please enter valid numeric values.")


def main():
    """
    Main driver function to interact with the user, perform computations, and display results.
    """
    print("Welcome to the T-Distribution Probability Calculator!")

    while True:
        # Prompt for user input
        m, z = prompt_user_input()

        # Calculate the probability
        probability = t_distribution_probability(m, z)

        # Display the result
        print(f"P(T ≤ {z} | m = {m}) = {probability:.4f}")

        # Ask the user if they want to compute another probability
        again = input("Would you like to calculate another probability? (y/n): ").strip().lower()
        if again not in ["y", "yes"]:
            print("Thank you for using the T-Distribution Probability Calculator. Goodbye!")
            break


if __name__ == "__main__":
    main()


from numericalMethods import GPDF, Probability, Secant


# Secant Method for Numerical Solution
def secant_method(func, x0, x1, tol=1e-5, max_iter=100):
    """
    Secant method to find the root of a function
    :param func: The function to find the root of
    :param x0: The initial guess
    :param x1: The second initial guess
    :param tol: The tolerance for convergence
    :param max_iter: The maximum number of iterations
    :return: The root of the function, or None if max iterations are reached
    """
    for i in range(max_iter):
        fx0 = func(x0)
        fx1 = func(x1)
        if abs(fx1 - fx0) < tol:
            return x1  # Convergence check
        # Secant update
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x2
    return None  # Return None if it doesn't converge


# Define a function for the probability difference (used in secant method)
def find_probability_difference(c, mean, stDev, target_prob, GT, OneSided):
    """
    This function returns the difference between the desired probability and the computed probability for a given c.
    :param c: The value of c to test
    :param mean: The population mean
    :param stDev: The standard deviation
    :param target_prob: The target probability (P)
    :param GT: Whether the user wants P(x > c)
    :param OneSided: Whether the integration is one-sided
    :return: The difference between the computed probability and the target probability
    """
    if OneSided:
        prob = Probability(GPDF, (mean, stDev), c, GT=GT)
    else:
        prob = Probability(GPDF, (mean, stDev), c, GT=True)
        prob = 1 - 2 * prob
    return prob - target_prob


def main():
    """
    This program allows the user to either specify a probability and solve for c,
    or specify a c value and solve for the probability.
    """
    Again = True
    mean = 0
    stDev = 1.0
    OneSided = True
    GT = False
    yesOptions = ["y", "yes", "true"]

    while Again:
        # Ask user if they are specifying c or P
        response = input(
            "Specify c and solve for P (enter 'c') or specify P and solve for c (enter 'P'): ").strip().lower()

        if response == 'c':  # User specifies c, we compute the probability
            c = float(input(f"Enter value for c: ({0.5}) ").strip() or "0.5")
            prob = Probability(GPDF, (mean, stDev), c, GT=GT) if OneSided else Probability(GPDF, (mean, stDev), c,
                                                                                           GT=True)
            prob = prob if OneSided else 1 - 2 * prob
            print(f"P(x{'>' if GT else '<'}{c:0.2f}|{mean:0.2f},{stDev:0.2f}) = {prob:0.3f}")

        elif response == 'P':  # User specifies P, we solve for c
            target_prob = float(input("Enter the target probability P (e.g., 0.95): ").strip())

            # Find initial guesses for the secant method
            x0 = 0.1  # A reasonable initial guess
            x1 = 5.0  # A second reasonable initial guess

            # Secant method to find the value of c
            func = lambda c: find_probability_difference(c, mean, stDev, target_prob, GT, OneSided)
            c_solution = secant_method(func, x0, x1)

            if c_solution is not None:
                print(f"Value of c that gives probability {target_prob:0.3f} is c = {c_solution:0.3f}")
            else:
                print("Secant method did not converge.")

        # Ask if user wants to go again
        response = input(f"Go again? (Y/N): ").strip().lower()
        Again = True if response in yesOptions else False


if __name__ == "__main__":
    main()

#region imports
from math import sqrt, pi, exp

# ========== hw3a.py ==========
def GPDF(x, mu, sig):
    return (1 / (sig * sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sig) ** 2)

def Simpson(f, mu, sig, a, b, N=100):
    if N % 2: N += 1
    h = (b - a) / N
    result = f(a, mu, sig) + f(b, mu, sig)
    for i in range(1, N):
        coeff = 4 if i % 2 else 2
        result += coeff * f(a + i * h, mu, sig)
    return (h / 3) * result

def Probability(PDF, args, c, GT=True):
    mu, sig = args
    p = Simpson(PDF, mu, sig, mu - 5 * sig, c)
    return 1 - p if GT else p

def Secant(f, x0, x1, tol=1e-5, max_iter=50):
    for _ in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if fx1 - fx0 == 0: break
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(x2 - x1) < tol: return x2
        x0, x1 = x1, x2
    return x1

def main():
    Again = True
    mean = 0.0
    stDev = 1.0
    c = 0.5
    probability = 0.5
    OneSided = True
    GT = False
    yesOptions = ["y", "yes", "true"]

    while Again:
        response = input(f"Population mean? ({mean:0.3f}) ").strip().lower()
        mean = float(response) if response else mean

        response = input(f"Standard deviation? ({stDev:0.3f}) ").strip().lower()
        stDev = float(response) if response else stDev

        mode = input("Are you specifying c to find P or specifying P to find c? (Enter 'c' or 'p') ").strip().lower()

        if mode == 'c':
            response = input(f"c value? ({c:0.3f}) ").strip().lower()
            c = float(response) if response else c

            response = input(f"Probability greater than c? ({GT}) ").strip().lower()
            GT = True if response in yesOptions else False

            response = input(f"One sided? ({OneSided}) ").strip().lower()
            OneSided = True if response in yesOptions else False

            if OneSided:
                prob = Probability(GPDF, (mean, stDev), c, GT=GT)
                print(f"P(x {'>' if GT else '<'} {c:0.2f} | {mean:0.2f}, {stDev:0.2f}) = {prob:0.3f}")
            else:
                prob = Probability(GPDF, (mean, stDev), c, GT=True)
                prob = 1 - 2 * prob
                if GT:
                    print(f"P({mean - (c - mean)} > x > {mean + (c - mean)} | {mean:0.2f}, {stDev:0.2f}) = {1 - prob:0.3f}")
                else:
                    print(f"P({mean - (c - mean)} < x < {mean + (c - mean)} | {mean:0.2f}, {stDev:0.2f}) = {prob:0.3f}")

        elif mode == 'p':
            response = input(f"Desired probability? ({probability:0.3f}) ").strip().lower()
            probability = float(response) if response else probability

            response = input(f"Probability greater than c? ({GT}) ").strip().lower()
            GT = True if response in yesOptions else False

            response = input(f"One sided? ({OneSided}) ").strip().lower()
            OneSided = True if response in yesOptions else False

            if OneSided:
                c = Secant(lambda c: Probability(GPDF, (mean, stDev), c, GT=GT) - probability, 0.0, 1.0)
                print(f"c value for P(x {'>' if GT else '<'} {c:0.2f} | {mean:0.2f}, {stDev:0.2f}) = {c:0.3f}")
            else:
                func = lambda c: (1 - 2 * Probability(GPDF, (mean, stDev), c, GT=True)) - probability
                c = Secant(func, 0.0, 1.0)
                print(f"c value for P({mean - (c - mean)} < x < {mean + (c - mean)} | {mean:0.2f}, {stDev:0.2f}) = {c:0.3f}")

        response = input("Go again? (Y/N) ").strip().lower()
        Again = response in yesOptions

if __name__ == "__main__":
    main()

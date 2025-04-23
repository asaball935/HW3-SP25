import math


def check_symmetry(matrix):
    """Return True if matrix is symmetric."""
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True


def cholesky_factor(matrix):
    """Return the Cholesky factor L where matrix = L * L^T."""
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for row in range(n):
        for col in range(row + 1):
            total = sum(L[row][k] * L[col][k] for k in range(col))
            if row == col:
                val = matrix[row][row] - total
                if val <= 0:
                    raise ValueError("Matrix not positive definite")
                L[row][col] = math.sqrt(val)
            else:
                L[row][col] = (matrix[row][col] - total) / L[col][col]
    return L


def is_pos_def(matrix):
    """Try to perform Cholesky; if it fails, matrix isn't positive definite."""
    try:
        _ = cholesky_factor(matrix)
        return True
    except ValueError:
        return False


def forward_sub(L, b):
    """Solve L * y = b"""
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y


def backward_sub(U, y):
    """Solve U * x = y"""
    n = len(y)
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


def solve_by_cholesky(A, b):
    """Solve Ax = b using Cholesky method."""
    L = cholesky_factor(A)
    y = forward_sub(L, b)
    LT = [list(col) for col in zip(*L)]
    return backward_sub(LT, y)


def doolittle_factor(A):
    """Return L and U such that A = L * U using Doolittle's method."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    return L, U


def solve_by_doolittle(A, b):
    """Solve Ax = b using Doolittle decomposition."""
    L, U = doolittle_factor(A)
    y = forward_sub(L, b)
    return backward_sub(U, y)


def determine_and_solve(A, b):
    """Decide method based on matrix properties and solve Ax = b."""
    if check_symmetry(A) and is_pos_def(A):
        print("Using Cholesky method (matrix is symmetric and positive definite).")
        return solve_by_cholesky(A, b)
    else:
        print("Using Doolittle method (matrix is not symmetric positive definite).")
        return solve_by_doolittle(A, b)


def run_hw3c():
    A1 = [
        [1, -1, 3, 2],
        [-1, 5, -5, -2],
        [3, -5, 19, 3],
        [2, -2, 3, 21]
    ]
    b1 = [15, -35, 94, 1]

    A2 = [
        [4, 1, 4, 0],
        [2, 2, 3, 2],
        [4, 3, 6, 3],
        [0, 2, 3, 9]
    ]
    b2 = [20, 36, 60, 122]

    print("Solving System 1:")
    x1 = determine_and_solve(A1, b1)
    print("x1 =", x1)

    print("\n" + "=" * 50 + "\n")

    print("Solving System 2:")
    x2 = determine_and_solve(A2, b2)
    print("x2 =", x2)


if __name__ == "__main__":
    run_hw3c()


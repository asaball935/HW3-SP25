import copy
import numpy as np
import scipy.linalg as linalg
import numericalMethods as nm
import matrixOperations as mo

# Gauss-Seidel method
def GaussSeidel(Aaug, x, Niter=15, epsilon=1e-5):
    AA = copy.deepcopy(Aaug)
    AA = DiagDominant(AA)  # ensure matrix is diagonal dominant

    n = len(x)
    for j in range(Niter):
        maxErr = 0
        for r in range(n):
            xOld = x[r]
            rhs = AA[r][n]  # the value from last col of row r
            for c in range(n):
                if c != r:
                    rhs -= AA[r][c] * x[c]
            x[r] = rhs / AA[r][r]
            maxErr = max(maxErr, abs(xOld - x[r]))
        if maxErr <= epsilon:
            break
    return x

def DiagDominant(A):
    AA = copy.deepcopy(A)
    rows = len(AA)
    for i in range(rows):
        c = abs(AA[i][i])
        for k in range(i + 1, rows):
            if abs(AA[k][i]) > c:
                row = AA.pop(k)
                AA.insert(i, row)
                c = abs(AA[i][i])
    return AA

def separateAugmented(Aaug):
    A = copy.deepcopy(Aaug)
    b = []
    n = len(A[0]) - 1
    for r in A:
        b.append(r.pop(n))
    b = mo.makeColumnVector(b)
    return (A, b)

def checkMatrixSoln(A, x, augmented=True):
    if augmented:
        AA, b = separateAugmented(A)
    else:
        AA = A
    B = []
    for r in AA:
        s = 0
        rCntr = 0
        for c in r:
            s += c * x[rCntr]
            rCntr += 1
        B.append(s)
    return B

# Doolittle LU Decomposition method
def LUFactorization(A):
    n = len(A)
    U = [([0 for c in range(n)] if not r == 0 else [a for a in A[0]]) for r in range(n)]
    L = [[(1 if c == r else (A[r][0] / U[0][0] if c == 0 else 0)) for c in range(n)] for r in range(n)]

    for j in range(1, n):
        for k in range(j, n):
            U[j][k] = A[j][k]
            for s in range(j):
                U[j][k] -= L[j][s] * U[s][k]

        for i in range(k + 1, n):
            sig = 0
            for s in range(k):
                sig += L[i][s] * U[s][k]
            L[i][k] = (1 / U[k][k]) * (A[i][k] - sig)
    return (L, U)

def BackSolve(A, b, UT=True):
    nRows = len(b)
    nCols = nRows
    x = [0] * nRows
    if UT:
        for nR in range(nRows - 1, -1, -1):
            s = 0
            for nC in range(nR + 1, nRows):
                s += A[nR][nC] * x[nC]
            x[nR] = 1 / A[nR][nR] * (b[nR] - s)
    else:
        for nR in range(nRows):
            s = 0
            for nC in range(nR):
                s += A[nR][nC] * x[nC]
            x[nR] = 1 / A[nR][nR] * (b[nR] - s)
    return x

def Doolittle(Aaug):
    A, b = mo.separateAugmented(Aaug)
    L, U = LUFactorization(A)
    y = BackSolve(L, b, UT=False)
    x = BackSolve(U, y, UT=True)
    return x

# Checking if the matrix is symmetric and positive definite
def is_symmetric(A):
    return np.allclose(A, np.transpose(A))

def is_positive_definite(A):
    try:
        linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Cholesky method
def cholesky_method(A, b):
    L = linalg.cholesky(A, lower=True)
    y = linalg.solve(L, b)
    x = linalg.solve(L.T, y)
    return x

# Main solver function based on matrix properties
def solve_matrix_equation(Aaug, x):
    A, b = mo.separateAugmented(Aaug)
    if is_symmetric(A) and is_positive_definite(A):
        print("Matrix is symmetric and positive definite. Using Cholesky method:")
        return cholesky_method(A, b)
    else:
        print("Matrix is not symmetric or positive definite. Using Doolittle method:")
        return Doolittle(Aaug)

def main():
    # Example matrices for testing
    A1 = [[3, 1, -1, 2],
          [1, 4, 1, 12],
          [2, 1, 2, 10]]
    x1 = [0, 0, 0]

    A2 = [[1, -10, 2, 4, 2],
          [3, 1, 4, 12, 12],
          [9, 2, 3, 4, 21],
          [-1, 2, 7, 3, 37]]
    x2 = [1, 1, 1, 1]

    print("Solving system 1 using appropriate method:")
    xSoln1 = solve_matrix_equation(A1, x1)
    print("Solution vector x1: ", [round(x, 4) for x in xSoln1])

    print("\nSolving system 2 using appropriate method:")
    xSoln2 = solve_matrix_equation(A2, x2)
    print("Solution vector x2: ", [round(x, 4) for x in xSoln2])

if __name__ == "__main__":
    main()

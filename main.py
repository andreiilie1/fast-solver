import numpy as np
from scipy.sparse import *
from scipy import *
import matplotlib.pyplot as plt
import sys
import math


# What I have tried for CG: 
# 1) corrected some float divisions which were getting converted to int's in auxilliary steps
# 2) checked that the matrix is correct for small sizes, checked for N's up to 20 that it is positive definite and symmetric
# 3) checked that the solution for g = - 2 sin x sin y converges (fast) to the correct solution f = sin x sin y
# 4) tried 3 different implementations (one taken exactly from Golub - practical CG notes) of CG, they all get the same ||Ax - b|| (iteration) graphs

# TODO: function JacobiIteration(D,R,b) to solve Mx=b

# https://people.eecs.berkeley.edu/~demmel/cs267/lectureSparseLU/lectureSparseLU1.html for 
# Cholesky on sparse matrices

# https://en.wikipedia.org/wiki/Successive_over-relaxation
# SOR generalization of GS / J

# http://cpsc.yale.edu/sites/default/files/files/tr48.pdf
# Gauss efficient sparse implementation

# We construct matrix M to approximate the solution of a differential equation
# We'll get the equation Mx = valueVector2D and try to solve it
# by different methods
# The scope is to get a fast solver for a class of differential equations

globalIterationConstant = 20
COMPLETE_MATRIX = 'COMPLETE_MATRIX'
LOWER_MATRIX = 'LOWER_MATRIX'
STRICTLY_UPPER_MATRIX = 'STRICTLY_UPPER_MATRIX'
DIAGONAL_MATRIX = 'DIAGONAL_MATRIX'
REMAINDER_MATRIX = 'REMAINDER_MATRIX'

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# Helper vector operations functions
def vectorSum(a, b):
	return map(sum, zip(a, b))


def multiplyVectorByScalar(vect, scalar):
	return map (lambda x: x * scalar, vect)
#____________________________________________#


# Helper functions for creating the complete, lower, upper and diagonal matrices
def addEntry(type, row, column, value):
    if(type == COMPLETE_MATRIX):
        rowList.append(row)
        colList.append(column)
        dataList.append(value)
    elif(type == LOWER_MATRIX):
        rowListLower.append(row)
        colListLower.append(column)
        dataListLower.append(value)
    elif(type == STRICTLY_UPPER_MATRIX):
        rowListUpper.append(row)
        colListUpper.append(column)
        dataListUpper.append(value)
    elif(type == DIAGONAL_MATRIX):
        rowListDiagonal.append(row)
        colListDiagonal.append(column)
        dataListDiagonal.append(value)
    elif(type == REMAINDER_MATRIX):
        rowListRemainder.append(row)
        colListRemainder.append(column)
        dataListRemainder.append(value)


def addEntryToMatrices(row, column, value):
    addEntry(COMPLETE_MATRIX, row, column, value)
    if(row == column):
        addEntry(DIAGONAL_MATRIX, row, column, value)
        addEntry(LOWER_MATRIX, row, column, value)
    if(row > column):
        addEntry(LOWER_MATRIX, row, column, value)
        addEntry(REMAINDER_MATRIX, row, column, value)
    if(row < column):
        addEntry(STRICTLY_UPPER_MATRIX, row, column, value)
        addEntry(REMAINDER_MATRIX, row, column, value)
#____________________________________________#


# Iterative methods for solving a linear system
def JacobiIterate(D, R, M, b):
    x = []
    d = D.diagonal()
    iterationConstant = globalIterationConstant
    # Initial guess is x = (0,0,...0)
    x = np.zeros_like(b)
    # Iterate constant number of times (TODO: iterate while big error Mx-b)
    for i in range(iterationConstant):
        err = np.subtract(M.dot(x), b)
        absErr = math.sqrt(err.dot(err))
        errorDataJacobi.append(math.log(absErr))
        y = R.dot(x)
        r = np.subtract(b, y)
        x = [r_i / d_i for r_i, d_i in zip(r, d)]
    err = np.subtract(M.dot(x), b)
    absErr = math.sqrt(err.dot(err))
    errorDataJacobi.append(math.log(absErr))
    return x, absErr



def GaussSeidelIterate(L, U, M, b):
    x = []
    d = L.diagonal()
    iterationConstant = globalIterationConstant
    # Initial guess is x = (0,0,...0)
    x = np.zeros_like(b)
    # Iterate constant number of times (TODO: iterate while big error Mx-b)
    for i in range(iterationConstant):
        err = np.subtract(M.dot(x), b)
        absErr = math.sqrt(err.dot(err))
        errorDataGaussSeidel.append(math.log(absErr))
        xNew = np.zeros_like(x)
        for j in range(L.shape[0]):
            currentLowerRow = L.getrow(j)
            currentUperRow = U.getrow(j)
            rowSum = currentLowerRow.dot(xNew) + currentUperRow.dot(x)
            xNew[j] = (b[j] - rowSum) / d[j]
        x = xNew
    err = np.subtract(M.dot(x), b)
    absErr = math.sqrt(err.dot(err))
    errorDataGaussSeidel.append(math.log(absErr))
    return x, absErr


def SSORIterate(L, U, M, b):
	omega = 1.0
	errorDataSSOR = []
	x = []
	d = L.diagonal()
	iterationConstant = globalIterationConstant

	x = np.zeros_like(b)


	for k in range(iterationConstant):

		print(k)
		err = np.subtract(M.dot(x), b)
		absErr = math.sqrt(err.dot(err))
		errorDataSSOR.append(math.log(absErr))

		xNew = np.zeros_like(x)

		for i in range(L.shape[0]):
			currentLowerRow = L.getrow(i)
			currentUpperRow = U.getrow(i)

			currSum = currentLowerRow.dot(xNew) + currentUpperRow.dot(x)
			currSum = (b[i] - currSum) / d[i]
			xNew[i] = x[i] + omega * (currSum - x[i])

		x = xNew
		xNew = np.zeros_like(x)

		for i in reversed(range(L.shape[0])):
			currSum = 0
			currentLowerRow = L.getrow(i)
			currentUpperRow = U.getrow(i)

			currSum = currentLowerRow.dot(x) + currentUpperRow.dot(xNew) - d[i] * x[i]
			currSum = (b[i] - currSum) / d[i]
			xNew[i] = x[i] + omega * (currSum - x[i])

		x = xNew

	err = np.subtract(b, M.dot(x))
	absErr = math.sqrt(err.dot(err))
	errorDataSSOR.append(math.log(absErr))
	return x, absErr, errorDataSSOR, err


def SteepestDescent(M, b):
    avoidDivByZeroError = 0.0000000000000000001
    x = np.zeros_like(b)
    r = np.subtract(b, M.dot(x))
    print(r)
    iterationConstant = globalIterationConstant
    for i in range(iterationConstant):
    	err = np.subtract(M.dot(x), b)
        absErr = math.sqrt(err.dot(err))
        errorDataSteepestDescent.append(absErr)
    	alpha_numerator = r.dot(r)	
    	alpha_denominator = r.dot(M.dot(r))
    	if(alpha_denominator < avoidDivByZeroError):
    		break
    	alpha = alpha_numerator / alpha_denominator
    	x = vectorSum(x, multiplyVectorByScalar(r, alpha))
    	r = np.subtract(b, M.dot(x))
    err = np.subtract(M.dot(x), b)
    absErr = math.sqrt(err.dot(err))
    errorDataSteepestDescent.append(absErr)
    return x, absErr

# ActualSolution will hold the discretization for f = sin x sin y
# and is used for debugging purposes
actualSolution = []

def ConjugateGradientsHS(M, b):
    avoidDivByZeroError = 0.0000000001
    errorDataConjugateGradients = []
    x = np.zeros_like(b, dtype=np.float)
    r = np.subtract(b, M.dot(x))
    d = np.subtract(b, M.dot(x))
    for i in range((N+1)*(N+1)):
        solutionError = np.subtract(x, actualSolution)
        absErr = np.linalg.norm(solutionError)
        errorDataConjugateGradients.append(absErr)

        alpha_numerator = r.dot(r)
        alpha_denominator = d.dot(M.dot(d))
        if(alpha_denominator < avoidDivByZeroError):
    		break
    	alpha = 1.0 * alpha_numerator / alpha_denominator

    	x = np.add(x, np.multiply(d, alpha))
    	r_new = np.subtract(r, np.multiply(M.dot(d), alpha))

    	beta_numerator = r_new.dot(r_new)
    	beta_denominator = r.dot(r)
    	if(beta_denominator < avoidDivByZeroError):
    		break
    	beta = 1.0 * beta_numerator / beta_denominator

    	d = r_new + np.multiply(d, beta)
    	r = r_new

    solutionError = np.subtract(x, actualSolution)
    absErr = np.linalg.norm(solutionError)
    errorDataConjugateGradients.append(absErr)
    return x, absErr, errorDataConjugateGradients


def ConjugateGradients_Golub(A, b):
	errorDataConjugateGradients = []
	tol = 0.000001
	k = 0
	x = np.zeros_like(b)
	r = np.subtract(b, A.dot(x))
	ro_c = r.dot(r)
	delta = tol * np.linalg.norm(b)
	while math.sqrt(ro_c) > delta:
		err = np.subtract(M.dot(x), b)
		absErr = np.linalg.norm(err)
		errorDataConjugateGradients.append(absErr)
		k = k + 1
		if(k == 1):
			p = r
		else:
			tau = ro_c / ro_minus
			p = np.add(r, np.multiply(p, tau))
		w = A.dot(p)
		miu_nominator = ro_c
		miu_denominator = w.dot(p)
		miu = miu_nominator / miu_denominator
		x = np.add(x, np.multiply(p, miu))
		r = np.subtract(r, np.multiply(w, miu))
		ro_minus = ro_c
		ro_c = r.dot(r)

	err = np.subtract(M.dot(x), b)
	absErr = np.linalg.norm(err)
	errorDataConjugateGradients.append(absErr)
	return x, absErr, errorDataConjugateGradients
#____________________________________________#


# Value of the border function on values x,y
def borderFunction(x, y):
    # Assert (x,y) is on border
    value = 1.0 * math.sin(x) * math.sin(y)
    return value


# Nabla value of the differential equation at points x, y
def nablaFunction(x, y):
    value = - 2.0 * math.sin(x) * math.sin(y)
    return value


# NablaValueVector from Mx = valueVector2D
valueVector2D = []


# Check if a(i,j) is on border
def isOnBorder(i, j):
    # print(i, j)
    if(i == 0 or j == 0 or i == N or j == N):
        return True
    else:
        return False


# Get the coordinates of the variable around which the row-th row is created
def getCoordinates(row):
    return int(row / (N + 1)), row % (N + 1)


# Get the row of a(i, j)'s equation
def getRow(i, j):
    return(i * (N + 1) + j)


# Compute M and valueVector2D (in Mx = valueVector2D) and
# computer L, U, D (lower, strictly upper and diagonal matrices of M)
def computeMatrixAndVector():
    for currentRow in range((N + 1) * (N + 1)):
        computeRow(currentRow)


# Compute the elements of row-th row in (rowList, colList, dataList) for -nabla f(x,t) = g(x,t) problem
def computeRow(row):
    (x, y) = getCoordinates(row)
    if(isOnBorder(x, y)):
        addEntryToMatrices(row, row, 1.0)
        # The value of the border on point x/N, y/N is known,
        # so append the equation variable = value to the system
        valueVector2D.append(borderFunction((1.0) * x / N, (1.0) * y / N))
    else:
        value = - nablaFunction((1.0) * x / N, (1.0) * y / N) * h * h
        addEntryToMatrices(row, row, 4.0)

        for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if(not(isOnBorder(x + dX, y + dY))):
                addEntryToMatrices(row, getRow(x + dX, y + dY), -1.0)
            else:
                localValue = borderFunction((1.0) * (x + dX) / N, (1.0) * (y + dY) / N)
                value += localValue
        valueVector2D.append(value)


# Current value vector i.e. current value (at current time k * dT) of f(x,y,t) in du/dt - laplace(u) = f
valueVector = []

def solveHeatTimeEquation():
	return 'to implement'
    # To implement by encapsulating the data structures and methods in the main body

# Returns a vector with the initial solution at t = 0 for all x,y
def initialHeatTimeSolution():
	initSol = []
	for currentIndex in range(N * N):
		initSol.append(0)
	return initSol

# Value of f(x,y,t) in du/dt - laplace(u) = f
def rhsFunctionHeatTimeEquation(x, y, t):
	value = 0
	return value


# Value of the border function on values x,y and at time t
# used for the equation which also introduces time
def borderTimeFunction(x, y, t):
    # Assert (x,y) is on border
    value = 1
    return value


# Compute M and valueVector2D (in Mx = valueVector2D) and
# computer L, U, D (lower, strictly upper and diagonal matrices of M)
def computeMatrixHeatTimeEquation():
    for currentRow in range(N * N):
        computeRowHeatTimeEquation(currentRow)


# Computes the RHS vector at time k*dT w.r.t. the prevSol (sol vector at time = (k-1)*dT)
def computeVectorAtTimestep(k, prevSol):
	valueVector = []
	for currentIndex in range(N * N):
		(x, y) = getCoordinates(currentIndex)

		if(isOnBorder(x, y)):
			value = borderTimeFunction(x / N, y / N, k * dT)
		else:
			value = (h * h) * rhsFunctionHeatTimeEquation(x / N, y / N, k * dT)
			value = value + (h * h) / dT * prevSol[getRow(x, y)]

			for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
			 	if((isOnBorder(x + dX, y + dY))):
					localValue = borderTimeFunction((x + dX) / N, (y + dY) / N, k * dT)
					value += localValue

		valueVector.append(value)

	return valueVector


# Compute the elements of row-th row in (rowList, colList, dataList) for the heat equation with time
def computeRowHeatTimeEquation(row):
    (x, y) = getCoordinates(row)
    if(isOnBorder(x, y)):
        addEntryToMatrices(row, row, 1)
    else:
        addEntryToMatrices(row, row, 4 + (h * h) / dT)

        for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if(not(isOnBorder(x + dX, y + dY))):
                addEntryToMatrices(row, getRow(x + dX, y + dY), -1)


#____________________________________________#

def displayFunction3D(solution):
	return false
	#TODO: Implement 3d visualization of data


N = int(input("Enter inverse of coordinates sample rate\n"))
T = int(input("Enter inverse of time sample rate\n"))
h = 1.0 / N
dT = 1.0 / T

rowList = []
colList = []
dataList = []

rowListDiagonal = []
colListDiagonal = []
dataListDiagonal = []

rowListRemainder = []
colListRemainder = []
dataListRemainder = []

rowListUpper = []
colListUpper = []
dataListUpper = []

rowListLower = []
colListLower = []
dataListLower = []

# In the current format, just call one of the two computeMatrix functions below
computeMatrixAndVector()
# computeMatrixHeatTimeEquation()


# Instantiate sparse matrix M according to data kept in (rowList, colList, dataList)
M = csr_matrix((np.array(dataList), (np.array(rowList), np.array(colList))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
Mt = M.transpose()
Msym = np.subtract(M, Mt)
maxMSym = Msym.max()
minMSym = Msym.min()

if(maxMSym != 0 or minMSym != 0):
	print('Matrix not symmetric!!')
	sys.exit(1)
# print(is_pos_def(M.toarray()))

# Diagonal and remainging matrices D, R with D + R = M
D = csr_matrix((np.array(dataListDiagonal), (np.array(rowListDiagonal), np.array(colListDiagonal))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
R = csr_matrix((np.array(dataListRemainder), (np.array(rowListRemainder), np.array(colListRemainder))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))

# Lower and strictly upper matrices L, U with L + U = M
L = csr_matrix((np.array(dataListLower), (np.array(rowListLower), np.array(colListLower))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
U = csr_matrix((np.array(dataListUpper), (np.array(rowListUpper), np.array(colListUpper))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))

# Error data after each iteration
errorDataJacobi = []
errorDataGaussSeidel = []
errorDataSteepestDescent = []
errorDataConjugateGradients = []
errorDataHeatConjugateGradients = []
errorDataConjugateGradients2 = []
# solHeat = initialHeatTimeSolution()
# for k in range(1, T + 1):
# 	t = k * dT
# 	valueVector = computeVectorAtTimestep(k, solHeat)
# 	(solHeat, errHeat, errorDataHeatConjugateGradients) = ConjugateGradientsHS(M, valueVector)
# 	print(errHeat)

(solution, error) = JacobiIterate(D, R, M, valueVector2D)
(solution2, error2, errorDataSSOR, _) = SSORIterate(L, U, M, valueVector2D)
(solution3, error3) = GaussSeidelIterate(L,U,M,valueVector2D)

# (solution3, error3) = SteepestDescent(M, valueVector2D)
# for i in range(N + 1):
# 	for j in range(N + 1):
# 		actualSolution.append(math.sin((1.0) * i / N) * math.sin((1.0) * j / N))
# (solution4, error4, errorDataConjugateGradients) = ConjugateGradientsHS(M, valueVector2D)
# (solution5, error5, errorDataConjugateGradients2) = ConjugateGradientsHS2(M, valueVector2D)
# (solution6, error6, errorDataConjugateGradients3) = ConjugateGradients_Golub(M, valueVector2D)

plt.plot(errorDataJacobi, label = 'Jacobi')
plt.plot(errorDataSSOR, label = 'SSOR')
plt.plot(errorDataGaussSeidel, label = 'Gauss-Seidel')
# plt.plot(errorDataSteepestDescent, label = 'Steepest Descent')
# plt.plot(errorDataConjugateGradients, label = 'Conjugate Gradients - x error')
# plt.plot(errorDataConjugateGradients2, label = 'Conjugate Gradients - Ax error')
# plt.plot(errorDataConjugateGradients3, label = 'Conjugate Gradients 3')

plt.legend(loc='upper right')
plt.show()
# Next lines are for debugging purpose

# print('Solution of Jacobi Iteration:')
# #print(solution)
# print('Error of Jacobi Iteration:')
# print(error)
# print('_________')

# print('Solution of Gauss Seidel Iteration:')
# #print(solution2)
# print('Error of Gauss Seidel Iteration:')
# print(error2)
# print('_________')

# print('Solution of Steepest Descent Iteration:')
# #print(solution3)
# print('Error of Steepest Descent Iteration:')
# print(error3)
# print('_________')

# print('Solution of CG Iteration:')
# print(solution4)

# print actualSolution

# print('Error of CG Iteration:')
# print(error4)
# print('_________')


# print(M.toarray())
# print(D.toarray())
# print(R.toarray())
# print(U.toarray())
# print(D.toarray())
#
# print(valueVector2D)
# print(M.data())
# print(*rowList, sep=' ')
# print(*colList, sep=' ')

# The development plan is as follows:
# 1) Gauss-Seidel, 2) Jacobi, 3) Steepest Descent, 4) Gauss Elimination
# 5) Encapsulate the whole state in a class to which we can pass as paramters
#    the functions for the border values and for the nabla values so we are able
#    to instantiate it for different problems and test the efficiency easier

# Sample output for N = 4 (we're showing the matrix and the nabla vector)

# Enter inverse of sample rate
# 4
# [[ 1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  4 -1  0  0 -1  0  0  0  0  0  0]
#  [ 0  0  0  0  0 -1  4  0  0  0 -1  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0 -1  0  0  0  4 -1  0  0  0  0  0]
#  [ 0  0  0  0  0  0 -1  0  0 -1  4  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1]]
# 0.0625 0.0625 0.0625 0.0625 0.0625 0.125 0.125 0.0625 0.0625 0.125 0.125 0.0625 0.0625 0.0625 0.0625 0.0625

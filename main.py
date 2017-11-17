import numpy as np
from scipy.sparse import *
from scipy import *
import matplotlib.pyplot as plt
import sys

# TODO: function JacobiIteration(D,R,b) to solve Mx=b

# https://people.eecs.berkeley.edu/~demmel/cs267/lectureSparseLU/lectureSparseLU1.html for 
# Cholesky on sparse matrices

# https://en.wikipedia.org/wiki/Successive_over-relaxation
# SOR generalization of GS / J

# http://cpsc.yale.edu/sites/default/files/files/tr48.pdf
# Gauss efficient sparse implementation

# We construct matrix M to approximate the solution of a differential equation
# We'll get the equation Mx = nablaValueVector and try to solve it
# by different methods
# The scope is to get a fast solver for a class of differential equations

globalIterationConstant = 10000
COMPLETE_MATRIX = 'COMPLETE_MATRIX'
LOWER_MATRIX = 'LOWER_MATRIX'
STRICTLY_UPPER_MATRIX = 'STRICTLY_UPPER_MATRIX'
DIAGONAL_MATRIX = 'DIAGONAL_MATRIX'
REMAINDER_MATRIX = 'REMAINDER_MATRIX'

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
        errorDataJacobi.append(absErr)
        y = R.dot(x)
        r = np.subtract(b, y)
        x = [r_i / d_i for r_i, d_i in zip(r, d)]
    err = np.subtract(M.dot(x), b)
    absErr = math.sqrt(err.dot(err))
    errorDataJacobi.append(absErr)
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
        errorDataGaussSeidel.append(absErr)
        xNew = np.zeros_like(x)
        for j in range(L.shape[0]):
            currentLowerRow = L.getrow(j)
            currentUperRow = U.getrow(j)
            rowSum = currentLowerRow.dot(xNew) + currentUperRow.dot(x)
            xNew[j] = (b[j] - rowSum) / d[j]
        x = xNew
    err = np.subtract(M.dot(x), b)
    absErr = math.sqrt(err.dot(err))
    errorDataGaussSeidel.append(absErr)
    return x, absErr



def SteepestDescent(M, b):
    avoidDivByZeroError = 0.0000000000000000001
    x = np.zeros_like(b)
    r = np.subtract(b, M.dot(x))
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



def ConjugateGradientsHS(M, b):
    avoidDivByZeroError = 0.000001
    errorDataConjugateGradients = []
    x = np.zeros_like(b)
    r = np.subtract(b, M.dot(x))
    d = np.subtract(b, M.dot(x))
    iterationConstant = globalIterationConstant
    for i in range(iterationConstant):
    	err = np.subtract(M.dot(x), b)
        absErr = math.sqrt(err.dot(err))
        errorDataConjugateGradients.append(absErr)

        alpha_numerator = r.dot(r)
        alpha_denominator = d.dot(M.dot(d))
        if(alpha_denominator < avoidDivByZeroError):
    		break
    	alpha = alpha_numerator / alpha_denominator

    	x = np.add(x, np.multiply(d, alpha))
    	r_new = np.subtract(r, np.multiply(M.dot(d), alpha))

    	beta_numerator = r_new.dot(r_new)
    	beta_denominator = r.dot(r)
    	if(beta_denominator < avoidDivByZeroError):
    		break
    	beta = beta_numerator / beta_denominator

    	d = r_new + np.multiply(d, beta)
    	r = r_new
    err = np.subtract(M.dot(x), b)
    absErr = math.sqrt(err.dot(err))
    errorDataConjugateGradients.append(absErr)
    return x, absErr, errorDataConjugateGradients

#____________________________________________#


# Value of the border function on values x,y
def borderFunction(x, y):
    # Assert (x,y) is on border
    value = 1
    return value


# Nabla value of the differential equation at points x, y
def nablaFunction(x, y):
    value = 0
    return value


# NablaValueVector from Mx = nablaValueVector
nablaValueVector = []


# Check if a(i,j) is on border
def isOnBorder(i, j):
    if(i == 0 or j == 0 or i == N - 1 or j == N - 1):
        return True
    else:
        return False


# Get the coordinates of the variable around which the row-th row is created
def getCoordinates(row):
    return int(row / N), row % N


# Get the row of a(i, j)'s equation
def getRow(i, j):
    return(i * N + j)


# Compute M and nablaValueVector (in Mx = nablaValueVector) and
# computer L, U, D (lower, strictly upper and diagonal matrices of M)
def computeMatrixAndVector():
    for currentRow in range(N * N):
        computeRow(currentRow)


# Compute the elements of row-th row in (rowList, colList, dataList) for -nabla f(x,t) = g(x,t) problem
def computeRow(row):
    (x, y) = getCoordinates(row)
    if(isOnBorder(x, y)):
        addEntryToMatrices(row, row, 1)
        # The value of the border on point x/N, y/N is known,
        # so append the equation variable = value to the system
        nablaValueVector.append(borderFunction(x / N, y / N))
    else:
        value = - nablaFunction(x / N, y / N) * h * h
        addEntryToMatrices(row, row, 4)

        for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if(not(isOnBorder(x + dX, y + dY))):
                addEntryToMatrices(row, getRow(x + dX, y + dY), -1)
            else:
                localValue = borderFunction((x + dX) / N, (y + dY) / N)
                value += localValue
        nablaValueVector.append(value)


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


# Compute M and nablaValueVector (in Mx = nablaValueVector) and
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
# computeMatrixAndVector()
computeMatrixHeatTimeEquation()

# Instantiate sparse matrix M according to data kept in (rowList, colList, dataList)
M = csr_matrix((np.array(dataList), (np.array(rowList), np.array(colList))), shape = (N * N, N * N))
Mt = M.transpose()
Msym = np.subtract(M, Mt)
maxMSym = Msym.max()
minMSym = Msym.min()

if(maxMSym != 0 or minMSym != 0):
	print('Matrix not symmetric!!')
	sys.exit(1)

# Diagonal and remainging matrices D, R with D + R = M
D = csr_matrix((np.array(dataListDiagonal), (np.array(rowListDiagonal), np.array(colListDiagonal))), shape = (N * N, N * N))
R = csr_matrix((np.array(dataListRemainder), (np.array(rowListRemainder), np.array(colListRemainder))), shape = (N * N, N * N))

# Lower and strictly upper matrices L, U with L + U = M
L = csr_matrix((np.array(dataListLower), (np.array(rowListLower), np.array(colListLower))), shape = (N * N, N * N))
U = csr_matrix((np.array(dataListUpper), (np.array(rowListUpper), np.array(colListUpper))), shape = (N * N, N * N))

# Error data after each iteration
errorDataJacobi = []
errorDataGaussSeidel = []
errorDataSteepestDescent = []
errorDataConjugateGradients = []
errorDataHeatConjugateGradients = []

solHeat = initialHeatTimeSolution()
for k in range(1, T + 1):
	t = k * dT
	valueVector = computeVectorAtTimestep(k, solHeat)
	(solHeat, errHeat, errorDataHeatConjugateGradients) = ConjugateGradientsHS(M, valueVector)
	print(errHeat)

# (solution, error) = JacobiIterate(D, R, M, nablaValueVector)
# (solution2, error2) = GaussSeidelIterate(L, U, M, nablaValueVector)
# (solution3, error3) = SteepestDescent(M, nablaValueVector)
# (solution4, error4, errorDataConjugateGradients) = ConjugateGradientsHS(M, nablaValueVector)

# plt.plot(errorDataJacobi, label = 'Jacobi')
# plt.plot(errorDataGaussSeidel, label = 'Gauss-Seidel')
# plt.plot(errorDataSteepestDescent, label = 'Steepest Descent')
# plt.plot(errorDataConjugateGradients, label = 'Conjugate Gradients')
# plt.legend(loc='upper right')
# plt.show()
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
# print('Error of CG Iteration:')
# print(error4)
# print('_________')


# print(M.toarray())
# print(D.toarray())
# print(R.toarray())
# print(U.toarray())
# print(D.toarray())
#
# print(nablaValueVector)
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

import numpy as np
from scipy.sparse import *
from scipy import *
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

COMPLETE_MATRIX = 'COMPLETE_MATRIX'
LOWER_MATRIX = 'LOWER_MATRIX'
STRICTLY_UPPER_MATRIX = 'STRICTLY_UPPER_MATRIX'
DIAGONAL_MATRIX = 'DIAGONAL_MATRIX'
REMAINDER_MATRIX = 'REMAINDER_MATRIX'

tol = 0.000001

class MultiGrid:

	def __init__(self, borderFunction, valueFunction, niu1 = 5, niu2 = 5):
		self.borderFunction = borderFunction
		self.valueFunction = valueFunction
		self.niu1 = niu1
		self.niu2 = niu2


	def getCoordinates(self, row, N):
	    return int(row / (N + 1)), row % (N + 1)

	def getRow(self, i, j, N):
	    return(i * (N + 1) + j)


	def restrict(self, r, N):
		restr = []

		for(i,elem) in enumerate(r):
			(x, y) = self.getCoordinates(i, N)

			if(x%2 == 0 and y%2 ==0):
				restr.append(elem)

		return restr

	def interpolate(self, r, fineN, coarseN):
		interp = []
		for i in range((fineN + 1) * (fineN + 1)):
			(x, y) = self.getCoordinates(i, fineN)
			if(x % 2 == 0 and y % 2 == 0):
				index = self.getRow(x / 2, y / 2, coarseN)
				interp.append(r[index])
			elif(x % 2 == 1 and y % 2 == 0):
				index1 = self.getRow((x-1) / 2, y /2, coarseN)
				index2 = self.getRow((x+1) / 2, y /2, coarseN)
				interp.append((r[index1] + r[index2]) / 2.0)
			elif(x % 2 == 0 and y % 2 == 1):
				index1 = self.getRow(x / 2, (y-1) /2, coarseN)
				index2 = self.getRow(x / 2, (y+1) /2, coarseN)
				interp.append((r[index1] + r[index2]) / 2.0)
			else:
				index1 = self.getRow((x-1) / 2, (y-1) /2, coarseN)
				index2 = self.getRow((x+1) / 2, (y-1) /2, coarseN)
				index3 = self.getRow((x-1) / 2, (y+1) /2, coarseN)
				index4 = self.getRow((x+1) / 2, (y+1) /2, coarseN)
				interp.append((r[index1] + r[index2] + r[index3] + r[index4])/4.0)

		return interp


	def vcycle(self, N, f, initSol = []):
		discr = SimpleEquationDiscretizer(N, self.borderFunction, self.valueFunction)

		fSize = len(f)
		if(fSize < 20):
			v = la.solve(discr.M.todense(), f)
			return v

		solver1 = SolverMethods(self.niu1, discr, f, initSol)
		v, _, _, _ = solver1.JacobiIterate()

		assert(N % 2 == 0)
		coarseN = N  / 2

		Mv = discr.M.dot(v)
		residual = np.subtract(np.array(f), Mv)
		coarseResidual = self.restrict(residual, N)

		coarseV = self.vcycle(coarseN, coarseResidual)
		fineV = self.interpolate(coarseV, N, coarseN)
		v = np.add(v, fineV)

		solver2 = SolverMethods(self.niu2, discr, f, v)
		v2, _, _, _ = solver2.JacobiIterate()
		return v2

	def iterateVCycles(self, N, t):
		initSol = []
		vErrors = []
		discr = SimpleEquationDiscretizer(N, self.borderFunction, self.valueFunction)
		f = discr.valueVector2D

		for i in range(t):
			currSol = self.vcycle(N, f, initSol)

			err = np.subtract(discr.M.dot(currSol), discr.valueVector2D)
			absErr = math.sqrt(err.dot(err))
			if(absErr < tol):
				break
			vErrors.append(math.log(absErr))

			initSol = currSol

		return currSol, vErrors

class SolverMethods:
	# Iterative methods for solving a linear system

	def __init__(self, iterationConstant, eqDiscretizer, b = [], initSol = []):
		self.iterationConstant = iterationConstant
		self.M = eqDiscretizer.M
		self.D = eqDiscretizer.D
		self.R = eqDiscretizer.R
		self.L = eqDiscretizer.L
		self.U = eqDiscretizer.U
		if(b == []) :
			self.b = eqDiscretizer.valueVector2D
		else:
			self.b = b
		self.initSol = initSol

	def JacobiIterate(self):
	    errorDataJacobi = []
	    x = []
	    d = self.D.diagonal()
	    iterationConstant = self.iterationConstant
	    # Initial guess is x = (0,0,...0) if not provided as a parameter
	    if(self.initSol == []):
	    	x = np.zeros_like(self.b)
	    else:
	    	x = self.initSol
	    # Iterate constant number of times (TODO: iterate while big error Mx-b)
	    for i in range(iterationConstant):
	        err = np.subtract(self.M.dot(x), self.b)
	        absErr = math.sqrt(err.dot(err))
	        errorDataJacobi.append(absErr)
	        y = self.R.dot(x)
	        r = np.subtract(self.b, y)
	        x = [r_i / d_i for r_i, d_i in zip(r, d)]
	    err = np.subtract(self.b, self.M.dot(x))
	    absErr = math.sqrt(err.dot(err))
	    errorDataJacobi.append(absErr)
	    return x, absErr, errorDataJacobi, err

	def GaussSeidelIterate(self):
	    errorDataGaussSeidel = []
	    x = []
	    d = self.L.diagonal()
	    iterationConstant = self.iterationConstant
	    # Initial guess is x = (0,0,...0)
	    if(self.initSol == []):
	    	x = np.zeros_like(self.b)
	    else:
	    	x = self.initSol
	    # Iterate constant number of times (TODO: iterate while big error Mx-b)
	    for i in range(iterationConstant):
	        err = np.subtract(self.M.dot(x), self.b)
	        absErr = math.sqrt(err.dot(err))
	        errorDataGaussSeidel.append(absErr)
	        xNew = np.zeros_like(x)
	        for j in range(self.L.shape[0]):
	            currentLowerRow = self.L.getrow(j)
	            currentUperRow = self.U.getrow(j)
	            rowSum = currentLowerRow.dot(xNew) + currentUperRow.dot(x)
	            xNew[j] = (self.b[j] - rowSum) / d[j]

	        if np.allclose(x, xNew, rtol=1e-6):
        	    break

	        x = xNew

	    err = np.subtract(self.b, self.M.dot(x))
	    absErr = math.sqrt(err.dot(err))
	    errorDataGaussSeidel.append(absErr)
	    return x, absErr, errorDataGaussSeidel, err


class SimpleEquationDiscretizer:

	def __init__(self, N, borderFunction, valueFunction):
		self.N = N
		self.h = 1.0 / N
		self.borderFunction = borderFunction
		self.valueFunction = valueFunction

		self.rowList = []
		self.colList = []
		self.dataList = []

		self.rowListDiagonal = []
		self.colListDiagonal = []
		self.dataListDiagonal = []

		self.rowListRemainder = []
		self.colListRemainder = []
		self.dataListRemainder = []

		self.rowListUpper = []
		self.colListUpper = []
		self.dataListUpper = []

		self.rowListLower = []
		self.colListLower = []
		self.dataListLower = []

		self.valueVector2D = []

		self.computeMatrixAndVector()
		self.M = csr_matrix((np.array(self.dataList), (np.array(self.rowList), np.array(self.colList))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		self.D = csr_matrix((np.array(self.dataListDiagonal), (np.array(self.rowListDiagonal), np.array(self.colListDiagonal))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		self.R = csr_matrix((np.array(self.dataListRemainder), (np.array(self.rowListRemainder), np.array(self.colListRemainder))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))

		# Lower and strictly upper matrices L, U with L + U = M
		self.L = csr_matrix((np.array(self.dataListLower), (np.array(self.rowListLower), np.array(self.colListLower))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		self.U = csr_matrix((np.array(self.dataListUpper), (np.array(self.rowListUpper), np.array(self.colListUpper))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))

	# Helper functions for creating the complete, lower, upper and diagonal matrices
	def addEntry(self, type, row, column, value):
	    if(type == COMPLETE_MATRIX):
	        self.rowList.append(row)
	        self.colList.append(column)
	        self.dataList.append(value)
	    elif(type == LOWER_MATRIX):
	        self.rowListLower.append(row)
	        self.colListLower.append(column)
	        self.dataListLower.append(value)
	    elif(type == STRICTLY_UPPER_MATRIX):
	        self.rowListUpper.append(row)
	        self.colListUpper.append(column)
	        self.dataListUpper.append(value)
	    elif(type == DIAGONAL_MATRIX):
	        self.rowListDiagonal.append(row)
	        self.colListDiagonal.append(column)
	        self.dataListDiagonal.append(value)
	    elif(type == REMAINDER_MATRIX):
	        self.rowListRemainder.append(row)
	        self.colListRemainder.append(column)
	        self.dataListRemainder.append(value)


	def addEntryToMatrices(self, row, column, value):
	    self.addEntry(COMPLETE_MATRIX, row, column, value)
	    if(row == column):
	        self.addEntry(DIAGONAL_MATRIX, row, column, value)
	        self.addEntry(LOWER_MATRIX, row, column, value)
	    if(row > column):
	        self.addEntry(LOWER_MATRIX, row, column, value)
	        self.addEntry(REMAINDER_MATRIX, row, column, value)
	    if(row < column):
	        self.addEntry(STRICTLY_UPPER_MATRIX, row, column, value)
	        self.addEntry(REMAINDER_MATRIX, row, column, value)

	valueVector2D = []


	# Check if a(i,j) is on border
	def isOnBorder(self, i, j):
	    # print(i, j)
	    if(i == 0 or j == 0 or i == self.N or j == self.N):
	        return True
	    else:
	        return False


	# Get the coordinates of the variable around which the row-th row is created
	def getCoordinates(self, row):
	    return int(row / (self.N + 1)), row % (self.N + 1)


	# Get the row of a(i, j)'s equation
	def getRow(self, i, j):
	    return(i * (self.N + 1) + j)


	# Compute M and valueVector2D (in Mx = valueVector2D) and
	# computer L, U, D (lower, strictly upper and diagonal matrices of M)
	def computeMatrixAndVector(self):
	    for currentRow in range((self.N + 1) * (self.N + 1)):
	        self.computeRow(currentRow)


	# Compute the elements of row-th row in (rowList, colList, dataList) for -nabla f(x,t) = g(x,t) problem
	def computeRow(self, row):
	    (x, y) = self.getCoordinates(row)
	    if(self.isOnBorder(x, y)):
	        self.addEntryToMatrices(row, row, 1.0)
	        # The value of the border on point x/N, y/N is known,
	        # so append the equation variable = value to the system
	        self.valueVector2D.append(self.borderFunction((1.0) * x / self.N, (1.0) * y / self.N))
	    else:
	        value = - self.valueFunction((1.0) * x / self.N, (1.0) * y / self.N) * self.h * self.h
	        self.addEntryToMatrices(row, row, 4.0)

	        for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
	            if(not(self.isOnBorder(x + dX, y + dY))):
	                self.addEntryToMatrices(row, self.getRow(x + dX, y + dY), -1.0)
	            else:
	                localValue = self.borderFunction((1.0) * (x + dX) / self.N, (1.0) * (y + dY) / self.N)
	                value += localValue
	        self.valueVector2D.append(value)


# Value of the border function on values x,y
def sinBorderFunction(x, y):
    # Assert (x,y) is on border
    value = 1.0 * math.sin(x) * math.sin(y)
    return value


# RHS value of the differential equation at points x, y
def sinValueFunction(x, y):
    value = - 2.0 * math.sin(x) * math.sin(y)
    return value

def borderFunction1(x, y):
	value = 1.0 * x * y * (x + y)
	return value

def valueFunction1(x, y):
	value = 2.0 * x + 2.0 * y
	return value



N = int(input("Enter inverse of coordinates sample rate for the coarser grid\n"))

sinEquationDiscr = SimpleEquationDiscretizer(N, sinBorderFunction, sinValueFunction)

# solver = SolverMethods(100, sinEquationDiscr)
# (xFine, absErr, errorDataJacobi, rFine) = solver.JacobiIterate()

# print(errorDataJacobi)
n = 2
i = 0
fig = plt.figure()
while(n < N):
	i = i + 1
	n = n * 2
	h = 1.0 / n
	ax = fig.add_subplot(220 + i, projection='3d')
	ax.title.set_text("N = " + str(n))
	x = y = np.arange(0.0, 1.0 + h, h)
	X, Y = np.meshgrid(x, y)

	sinEquationDiscr = SimpleEquationDiscretizer(n, sinBorderFunction, sinValueFunction)
	mg = MultiGrid(sinBorderFunction, sinValueFunction)
	solMG, vErrors = mg.iterateVCycles(n, 600)
	Z = np.reshape(solMG, (n+1, n+1))

	ax.plot_wireframe(X, Y, Z)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	print('last error:', math.exp(vErrors[len(vErrors) - 1]))
	# solMG = mg.vcycle(N, sinEquationDiscr.valueVector2D)
	# print(solMG)
	# print('last error:', math.exp(vErrors[len(vErrors) - 1]))
	# plt.plot(vErrors, label = str(n))
	# plt.legend(loc='upper right')

# while(n < N):
# 	n = n * 2
# 	eqDiscr = SimpleEquationDiscretizer(n, borderFunction1, valueFunction1)
# 	mg = MultiGrid(borderFunction1, valueFunction1)
# 	solMG, vErrors = mg.iterateVCycles(n, eqDiscr.valueVector2D, 600)
# 	# solMG = mg.vcycle(N, sinEquationDiscr.valueVector2D)
# 	# print(solMG)
# 	print('last error:', math.exp(vErrors[len(vErrors) - 1]))
# 	# plt.plot(vErrors, label = str(n))
# 	# plt.legend(loc='upper right')
# 	print(vErrors)

plt.show()
# # TODO: Define restrict to maxe the grid 2x coarser for height and width
# rCoarse = restrict(rFine)
# sinEquationDiscrCoarse = SimpleEquationDiscretizer(N, sinBorderFunction, sinValueFunction)
# sinEquationDiscrCoarse.valueVector2D = rCoarse

# xCoarse = solve(sinEquationDiscrCoarse, rCoarse)

# xFineCorrection = interpolate(xCoarse)

# x = np.add(xFine, xFineCorrection)
# xSol = solver.JacobiIterateWithInitSol(x)

# Q: GS slow because I'm not using matrix operations, in 50 iterations for N = 100
# the error norm for the correct sol is 22, and there are 10.000 entries
# ___________________________________________________
# Debug tools:
# print('norm(Mx-b) error:')
# print(errorDataJacobi[:10])
# print ''

# actualSolution = []
# for i in range(N + 1):
# 	for j in range(N + 1):
# 		actualSolution.append(math.sin((1.0) * i / N) * math.sin((1.0) * j / N))

# solutionError = np.subtract(solMG, actualSolution)
# absErr = np.linalg.norm(solutionError)

# print('Solution obtained:')
# print x
# print ''
# print('Actual solution:')
# print actualSolution
# print ''
# print 'norm(x-x*)'
# print(absErr)
import numpy as np
from scipy.sparse import *
from scipy import *


COMPLETE_MATRIX = 'COMPLETE_MATRIX'
LOWER_MATRIX = 'LOWER_MATRIX'
STRICTLY_UPPER_MATRIX = 'STRICTLY_UPPER_MATRIX'
DIAGONAL_MATRIX = 'DIAGONAL_MATRIX'
REMAINDER_MATRIX = 'REMAINDER_MATRIX'


class TimeEquationDiscretizer:
	# Current value vector i.e. current value (at current time k * dT) of f(x,y,t) in du/dt - laplace(u) = f
	def __init__(self, N, T, borderFunction, valueFunction):
		self.N = N
		self.h = 1.0 / N

		self.T = T
		self.dT  = 1.0 / T

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

		self.computeMatrixHeatTimeEquation()
		self.M = csr_matrix((np.array(self.dataList), (np.array(self.rowList), np.array(self.colList))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		self.D = csr_matrix((np.array(self.dataListDiagonal), (np.array(self.rowListDiagonal), np.array(self.colListDiagonal))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		self.R = csr_matrix((np.array(self.dataListRemainder), (np.array(self.rowListRemainder), np.array(self.colListRemainder))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))

		# Lower and strictly upper matrices L, U with L + U = M
		self.L = csr_matrix((np.array(self.dataListLower), (np.array(self.rowListLower), np.array(self.colListLower))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		self.U = csr_matrix((np.array(self.dataListUpper), (np.array(self.rowListUpper), np.array(self.colListUpper))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))


	valueVector = []

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

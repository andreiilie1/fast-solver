import numpy as np
from scipy.sparse import *
from scipy import *


COMPLETE_MATRIX = 'COMPLETE_MATRIX'
LOWER_MATRIX = 'LOWER_MATRIX'
STRICTLY_UPPER_MATRIX = 'STRICTLY_UPPER_MATRIX'
DIAGONAL_MATRIX = 'DIAGONAL_MATRIX'
REMAINDER_MATRIX = 'REMAINDER_MATRIX'


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


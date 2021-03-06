# TimeEquationDiscretizer.py

#  _____________________________________________
# |USED TO GENERATE HEAT EQUATION DISCRETIZATION|
# |_____________________________________________|

import numpy as np
from scipy.sparse import *
from scipy import *


COMPLETE_MATRIX = 'COMPLETE_MATRIX'
LOWER_MATRIX = 'LOWER_MATRIX'
STRICTLY_UPPER_MATRIX = 'STRICTLY_UPPER_MATRIX'
DIAGONAL_MATRIX = 'DIAGONAL_MATRIX'
REMAINDER_MATRIX = 'REMAINDER_MATRIX'

# Class encapsulating the discretization arising from
# a 2D Heat Equation, including methods to get the
# RHS vector for the Backward Euler method
class TimeEquationDiscretizer:

	# Current value vector i.e. current value (at current time k * dT) of f(x,y,t) in du/dt - laplace(u) = f
	def __init__(self, N, T, borderTimeFunction, rhsHeatEquationFunction, initialHeatTimeFunction):
		self.N = N
		self.h = 1.0 / N

		self.T = T
		self.dT  = 1.0 / T

		self.borderTimeFunction = borderTimeFunction
		self.rhsHeatEquationFunction = rhsHeatEquationFunction
		self.initialHeatTimeFunction = initialHeatTimeFunction

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

		self.computeMatrixHeatTimeEquation()
		self.M = csr_matrix((np.array(self.dataList), (np.array(self.rowList), np.array(self.colList))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		self.D = csr_matrix((np.array(self.dataListDiagonal), (np.array(self.rowListDiagonal), np.array(self.colListDiagonal))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		self.R = csr_matrix((np.array(self.dataListRemainder), (np.array(self.rowListRemainder), np.array(self.colListRemainder))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))

		# Lower and strictly upper matrices L, U with L + U = M
		self.L = csr_matrix((np.array(self.dataListLower), (np.array(self.rowListLower), np.array(self.colListLower))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		self.U = csr_matrix((np.array(self.dataListUpper), (np.array(self.rowListUpper), np.array(self.colListUpper))), shape = ((N + 1) * (N + 1), (N + 1) * (N + 1)))
		
		print(self.M.todense())
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

	# Check if a(i,j) is on border
	def isOnBorder(self, i, j):
		# print(i, j)
		if(i == 0 or j == 0 or i == (self.N) or j == (self.N)):
			return True
		else:
			return False


	# Get the coordinates of the variable around which the row-th row is created
	def getCoordinates(self, row):
		return int(row / (self.N + 1)), row % (self.N + 1)


	# Get the row of a(i, j)'s equation
	def getRow(self, i, j):
		return(i * (self.N + 1) + j)

	valueVector = []

	# Returns a vector with the initial solution at t = 0 for all x,y
	def initialHeatTimeSolution(self):
		initSol = []
		for currentIndex in range((self.N + 1) * (self.N + 1)):
			(x, y) = self.getCoordinates(currentIndex)
			initSol.append(self.initialHeatTimeFunction(x, y, 0))
		return initSol


	# Compute M and valueVector2D (in Mx = valueVector2D) and
	# compute L, U, D (lower, strictly upper and diagonal matrices of M)
	def computeMatrixHeatTimeEquation(self):
		for currentRow in range((self.N + 1) * (self.N + 1)):
			self.computeRowHeatTimeEquation(currentRow)


	# Computes the RHS vector at time k*dT w.r.t. the prevSol (sol vector at time = (k-1)*dT)
	def computeVectorAtTimestep(self, k, prevSol):
		valueVector = []
		for currentIndex in range((self.N + 1) * (self.N + 1)):
			(x, y) = self.getCoordinates(currentIndex)

			if(self.isOnBorder(x, y)):
				value = self.borderTimeFunction((1.0) * x / (self.N + 1), (1.0) * y / (self.N + 1), (1.0) * k * self.dT)
			else:
				value = 1.0* (self.h * self.h) * self.rhsHeatEquationFunction((1.0) * x / (self.N + 1), (1.0) * y / (self.N + 1), (1.0) * k * self.dT)
				value = value + 1.0 * (self.h * self.h) / self.dT * prevSol[self.getRow(x, y)]

				for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
				 	if((self.isOnBorder(x + dX, y + dY))):
						localValue = self.borderTimeFunction((1.0) * (x + dX) / (self.N + 1), (1.0) * (y + dY) / (self.N + 1), k * self.dT)
						value += localValue

			valueVector.append(value)

		return valueVector


	# Compute the elements of row-th row in (rowList, colList, dataList) for the heat equation with time
	def computeRowHeatTimeEquation(self, row):
		(x, y) = self.getCoordinates(row)
		if(self.isOnBorder(x, y)):
			self.addEntryToMatrices(row, row, 1)
		else:
			self.addEntryToMatrices(row, row, 4 + (1.0) * (self.h * self.h) / self.dT)

			for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
				if(not(self.isOnBorder(x + dX, y + dY))):
					self.addEntryToMatrices(row, self.getRow(x + dX, y + dY), -1)
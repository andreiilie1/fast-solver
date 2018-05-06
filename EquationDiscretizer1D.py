# EquationDiscretizer1D.py

#  ___________________________________________________
# |USED TO GENERATE POISSON 1D EQUATION DISCRETIZATION|
# |___________________________________________________|

import numpy as np
from scipy.sparse import *
from scipy import *


COMPLETE_MATRIX = 'COMPLETE_MATRIX'
LOWER_MATRIX = 'LOWER_MATRIX'
STRICTLY_UPPER_MATRIX = 'STRICTLY_UPPER_MATRIX'
DIAGONAL_MATRIX = 'DIAGONAL_MATRIX'
REMAINDER_MATRIX = 'REMAINDER_MATRIX'

# Class encapsulating the sparse matrix
# arising from discretizing a 1D model problem
class EquationDiscretizer1D:

	def __init__(self, N, borderFunction, valueFunction, printMatrix = False):
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

		self.valueVector1D = []

		self.computeMatrixAndVector()
		self.M = csr_matrix((np.array(self.dataList), (np.array(self.rowList), np.array(self.colList))), shape = ((N + 1), (N + 1)))
		self.D = csr_matrix((np.array(self.dataListDiagonal), (np.array(self.rowListDiagonal), np.array(self.colListDiagonal))), shape = ((N + 1), (N + 1)))
		self.R = csr_matrix((np.array(self.dataListRemainder), (np.array(self.rowListRemainder), np.array(self.colListRemainder))), shape = ((N + 1), (N + 1)))

		# Lower and strictly upper matrices L, U with L + U = M
		self.L = csr_matrix((np.array(self.dataListLower), (np.array(self.rowListLower), np.array(self.colListLower))), shape = ((N + 1), (N + 1)))
		self.U = csr_matrix((np.array(self.dataListUpper), (np.array(self.rowListUpper), np.array(self.colListUpper))), shape = ((N + 1), (N + 1)))

		if(printMatrix):
			print("Discretization matrix: ")
			print(self.M.todense())
			print("RHS vector: ") 
			print(self.valueVector1D)
			
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

	valueVector1D = []


	# Check if a(i) is on border
	def isOnBorder(self, i):
		if(i == 0 or i == self.N):
			return True
		else:
			return False



	# Compute M and valueVector1D (in Mx = valueVector1D) and
	# computer L, U, D (lower, strictly upper and diagonal matrices of M)
	def computeMatrixAndVector(self):
		for currentRow in range((self.N + 1)):
			self.computeRow(currentRow)


	# Compute the elements of row-th row in (rowList, colList, dataList) for -nabla f(x,t) = g(x,t) problem
	def computeRow(self, row):
		x = row
		if(self.isOnBorder(x)):
			self.addEntryToMatrices(row, row, 1.0)
			# The value of the border on point x/N, y/N is known,
			# so append the equation variable = value to the system
			self.valueVector1D.append(self.borderFunction((1.0) * x / self.N))
		else:
			value = - self.valueFunction((1.0) * x / self.N) * self.h * self.h
			self.addEntryToMatrices(row, row, 2.0)

			for (dX) in [-1, 1]:
				if(not(self.isOnBorder(x + dX))):
					self.addEntryToMatrices(row, x + dX, -1.0)
				else:
					localValue = self.borderFunction((1.0) * (x + dX) / self.N)
					value += localValue
			self.valueVector1D.append(value)


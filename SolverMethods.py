import numpy as np
from scipy.sparse import *
from scipy import *
import math

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

	def JacobiIterate(self, dampFactor = 0):
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
			xPrev = np.copy(x)
			x = [r_i / d_i for r_i, d_i in zip(r, d)]
			x = np.add(np.multiply(xPrev, dampFactor), np.multiply(x, (1-dampFactor)))
		
		err = np.subtract(self.b, self.M.dot(x))
		absErr = math.sqrt(err.dot(err))
		errorDataJacobi.append(absErr)
		return x, absErr, errorDataJacobi, err

	def GaussSeidelIterate(self):
		errorDataGaussSeidel = []
		x = []
		d = self.L.diagonal()
		iterationConstant = self.iterationConstant

		if(self.initSol == []):
			x = np.zeros_like(self.b)
		else:
			x = self.initSol

		for i in range(iterationConstant):
			err = np.subtract(self.M.dot(x), self.b)
			absErr = math.sqrt(err.dot(err))
			errorDataGaussSeidel.append(absErr)

			xNew = np.zeros_like(x)
			for j in range(self.L.shape[0]):
				currentLowerRow = self.L.getrow(j)
				currentUpperRow = self.U.getrow(j)

				rowSum = currentLowerRow.dot(xNew) + currentUpperRow.dot(x)
				xNew[j] = 1.0 * (self.b[j] - rowSum) / d[j]

			# if np.allclose(x, xNew, rtol=1e-6):
			#	 break

			x = xNew

		err = np.subtract(self.b, self.M.dot(x))
		absErr = math.sqrt(err.dot(err))
		errorDataGaussSeidel.append(absErr)
		return x, absErr, errorDataGaussSeidel, err

	def SSORIterate(self, omega = 1.0):
		errorDataSSOR = []
		x = []
		d = self.D.diagonal()
		iterationConstant = self.iterationConstant

		if(self.initSol == []):
			x = np.zeros_like(self.b)
		else:
			x = self.initSol

		for k in range(iterationConstant):

			err = np.subtract(self.M.dot(x), self.b)
			absErr = math.sqrt(err.dot(err))
			errorDataSSOR.append(absErr)

			xNew = np.zeros_like(x)

			for i in range(self.L.shape[0]):
				currentLowerRow = self.L.getrow(i)
				currentUpperRow = self.U.getrow(i)

				currSum = currentLowerRow.dot(xNew) + currentUpperRow.dot(x)
				currSum = 1.0 * (self.b[i] - currSum) / d[i]
				xNew[i] = x[i] + omega * (currSum - x[i])

			x = xNew
			xNew = np.zeros_like(x)

			for i in reversed(range(self.L.shape[0])):
				currSum = 0
				currentLowerRow = self.L.getrow(i)
				currentUpperRow = self.U.getrow(i)

				currSum = currentLowerRow.dot(x) + currentUpperRow.dot(xNew) - d[i] * x[i]
				currSum = 1.0 * (self.b[i] - currSum) / d[i]
				xNew[i] = x[i] + omega * (currSum - x[i])

			x = xNew

		err = np.subtract(self.b, self.M.dot(x))
		absErr = math.sqrt(err.dot(err))
		errorDataSSOR.append(absErr)
		return x, absErr, errorDataSSOR, err



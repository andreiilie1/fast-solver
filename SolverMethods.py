import numpy as np
from scipy.sparse import *
from scipy import *
import math

tol = 0.000001
# https://www.ibiblio.org/e-notes/webgl/gpu/mg/poisson_rel.html : eigenvalues for the poisson matrix
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

	def JacobiIterate(self, dampFactor = 1.0):
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
			absErr = np.linalg.norm(err) / np.linalg.norm(self.b)
			errorDataJacobi.append(math.log(absErr))

			if(absErr < tol):
				break

			y = self.R.dot(x)
			r = np.subtract(self.b, y)
			xPrev = np.copy(x)
			x = [r_i / d_i for r_i, d_i in zip(r, d)]
			xNew = np.add(np.multiply(xPrev, (1.0 - dampFactor)), np.multiply(x, dampFactor))
			x = np.copy(xNew)


		
		err = np.subtract(self.b, self.M.dot(x))
		absErr = np.linalg.norm(err) / np.linalg.norm(self.b)
		errorDataJacobi.append(math.log(absErr))
		return x, absErr, errorDataJacobi, err

	def JacobiIterate2(self, omega = 1.0):
		errorDataJacobi = []
		x = []
		currentLowerRows = []
		currentUpperRows = []
		d = self.D.diagonal()
		iterationConstant = self.iterationConstant
		# Initial guess is x = (0,0,...0) if not provided as a parameter
		if(self.initSol == []):
			x = np.zeros_like(self.b)
		else:
			x = self.initSol

		for j in range(self.L.shape[0]):
			currentLowerRows.append(self.L.getrow(j))
			currentUpperRows.append(self.U.getrow(j))

		# Iterate constant number of times (TODO: iterate while big error Mx-b)
		for i in range(iterationConstant):
			err = np.subtract(self.M.dot(x), self.b)
			absErr = np.linalg.norm(err) / np.linalg.norm(self.b)
			errorDataJacobi.append(math.log(absErr))

			if(absErr < tol):
				break

			xNew = np.zeros_like(x)
			for j in range(self.L.shape[0]):
				currentLowerRow = currentLowerRows[j]
				currentUpperRow = currentUpperRows[j]

				rowSum = currentLowerRow.dot(x) + currentUpperRow.dot(x) - x[j] * d[j]
				rowSum = 1.0 * (self.b[j] - rowSum) / d[j]
				xNew[j] = x[j] + omega * (rowSum - x[j])

			if np.allclose(x, xNew, rtol=1e-6):
				 break

			x = np.copy(xNew)


		
		err = np.subtract(self.b, self.M.dot(x))
		absErr = np.linalg.norm(err) / np.linalg.norm(self.b)
		errorDataJacobi.append(math.log(absErr))
		return x, absErr, errorDataJacobi, err


	def GaussSeidelIterate(self, omega = 1.0):
		errorDataGaussSeidel = []
		x = []
		d = self.L.diagonal()
		iterationConstant = self.iterationConstant

		currentLowerRows = []
		currentUpperRows =[]

		if(self.initSol == []):
			x = np.zeros_like(self.b)
		else:
			x = self.initSol

		for j in range(self.L.shape[0]):
			currentLowerRows.append(self.L.getrow(j))
			currentUpperRows.append(self.U.getrow(j))

		for i in range(iterationConstant):
			err = np.subtract(self.M.dot(x), self.b)
			absErr = np.linalg.norm(err) / np.linalg.norm(self.b)
			errorDataGaussSeidel.append(math.log(absErr))

			xNew = np.zeros_like(x)
			for j in range(self.L.shape[0]):
				currentLowerRow = currentLowerRows[j]
				currentUpperRow = currentUpperRows[j]

				rowSum = currentLowerRow.dot(xNew) + currentUpperRow.dot(x) 
				rowSum = 1.0 * (self.b[j] - rowSum) / d[j]
				xNew[j] = x[j] + omega * (rowSum - x[j])

			# if np.allclose(x, xNew, rtol=1e-6):
			# 	 break

			if(absErr < tol):
				break

			x = np.copy(xNew)

		return x, absErr, errorDataGaussSeidel, err

	def SSORIterate(self, omega = 1.0, debugOn = False):
		errorDataSSOR = []
		x = []
		d = self.D.diagonal()
		iterationConstant = self.iterationConstant

		currentLowerRows = []
		currentUpperRows = []

		if(self.initSol == []):
			x = np.zeros_like(self.b)
		else:
			x = np.copy(self.initSol)

		for j in range(self.L.shape[0]):
			currentLowerRows.append(self.L.getrow(j))
			currentUpperRows.append(self.U.getrow(j))


		err = np.subtract(self.M.dot(x), self.b)
		absErr = np.linalg.norm(err) / (np.linalg.norm(self.b))
		errorDataSSOR.append(math.log(absErr))


		for k in range(iterationConstant):

			xNew = np.zeros_like(x)

			for i in range(self.L.shape[0]):
				currentLowerRow = currentLowerRows[i]
				currentUpperRow = currentUpperRows[i]

				currSum = currentLowerRow.dot(xNew) + currentUpperRow.dot(x)
				currSum = 1.0 * (self.b[i] - currSum) / d[i]
				xNew[i] = x[i] + omega * (currSum - x[i])

			x = np.copy(xNew)
			if(debugOn and k%10 == 0):
				print("Iteration: ", k)
				print("After top to bottom: ", x)
			xNew = np.zeros_like(x)
			for i in reversed(range(self.L.shape[0])):
				currSum = 0.0
				currentLowerRow = currentLowerRows[i]
				currentUpperRow = currentUpperRows[i]

				currSum = currentLowerRow.dot(x) + currentUpperRow.dot(xNew) - d[i] * x[i]
				currSum = 1.0 * (self.b[i] - currSum) / d[i]
				xNew[i] = x[i] + omega * (currSum - x[i])
			x = np.copy(xNew)
			if(debugOn and k%10 ==0):
				print("After bottom to top: ", x)
				print("______________")
			err = np.subtract(self.M.dot(x), self.b)
			absErr = np.linalg.norm(err) / (np.linalg.norm(self.b))
			errorDataSSOR.append(math.log(absErr))

			if(absErr < tol):
				break

		err = np.subtract(self.b, self.M.dot(x))
		absErr = np.linalg.norm(err) / np.linalg.norm(self.b)
		# errorDataSSOR.append(math.log(absErr))
		return x, absErr, errorDataSSOR, err

	def ConjugateGradientsHS(self):
		M = self.M
		b = self.b

		avoidDivByZeroError = 0.0000000001
		errorDataConjugateGradients = []
		x = np.zeros_like(b, dtype=np.float)
		r = np.subtract(b, M.dot(x))
		d = np.subtract(b, M.dot(x))
		convergence = False

		while(not convergence):
			solutionError = np.subtract(M.dot(x), b)
			absErr = np.linalg.norm(solutionError)
			errorDataConjugateGradients.append(math.log(absErr))
			if(absErr < tol):
				convergence = True
				break

			alpha_numerator = r.dot(r)
			alpha_denominator = d.dot(M.dot(d))
			if(alpha_denominator < avoidDivByZeroError):
				convergence = True
				break
			alpha = 1.0 * alpha_numerator / alpha_denominator

			x = np.add(x, np.multiply(d, alpha))
			r_new = np.subtract(r, np.multiply(M.dot(d), alpha))

			beta_numerator = r_new.dot(r_new)
			beta_denominator = r.dot(r)
			if(beta_denominator < avoidDivByZeroError):
				convergence = True
				break

			beta = 1.0 * beta_numerator / beta_denominator

			d = r_new + np.multiply(d, beta)
			r = r_new

		return x, absErr, errorDataConjugateGradients
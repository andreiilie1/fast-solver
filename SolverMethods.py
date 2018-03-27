import numpy as np
from scipy.sparse import *
from scipy import *
import math

tol = 0.00001
# https://www.ibiblio.org/e-notes/webgl/gpu/mg/poisson_rel.html : eigenvalues for the poisson matrix
class SolverMethods:
	# Iterative methods for solving a linear system

	def __init__(self, iterationConstant, eqDiscretizer, b = [], initSol = [], actualSol = []):
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
		self.actualSol = actualSol

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
			absErr = np.linalg.norm(err) / (1 + np.linalg.norm(self.b))
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
		absErr = np.linalg.norm(err) / (1 + np.linalg.norm(self.b))
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

		flops = 0

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
				flops += max(2 * (currentLowerRow.getnnz() + currentUpperRow.getnnz()) - 1, 0)

				rowSum = 1.0 * (self.b[j] - rowSum) / d[j]
				flops += 2

				xNew[j] = x[j] + omega * (rowSum - x[j])
				flops += 3

			# if np.allclose(x, xNew, rtol=1e-6):
			# 	 break

			if(absErr < tol):
				break

			x = np.copy(xNew)

		print("Flops: ",flops)
		print("Iterations: ", len(errorDataGaussSeidel) - 1)
		return x, absErr, errorDataGaussSeidel, err

	def SSORIterate(self, omega = 1.0, debugOn = False):
		errorDataSSOR = []
		x = []
		d = self.D.diagonal()
		iterationConstant = self.iterationConstant

		flops = 0

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
				flops += max(2 * (currentLowerRow.getnnz() + currentUpperRow.getnnz()) - 1, 0)

				currSum = 1.0 * (self.b[i] - currSum) / d[i]
				flops += 2

				xNew[i] = x[i] + omega * (currSum - x[i])
				flops += 3

			x = np.copy(xNew)

			if(debugOn and k % 10 == 0):
				print("Iteration: ", k)
				print("After top to bottom: ", x)

			xNew = np.zeros_like(x)
			for i in reversed(range(self.L.shape[0])):
				currSum = 0.0
				currentLowerRow = currentLowerRows[i]
				currentUpperRow = currentUpperRows[i]

				currSum = currentLowerRow.dot(x) + currentUpperRow.dot(xNew) - d[i] * x[i]
				flops += 2 * (currentLowerRow.getnnz() + currentUpperRow.getnnz()) + 1

				currSum = 1.0 * (self.b[i] - currSum) / d[i]
				flops += 2

				xNew[i] = x[i] + omega * (currSum - x[i])
				flops += 3

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

		print("Flops: ",flops)
		print("Iterations: ", len(errorDataSSOR) - 1)

		return x, absErr, errorDataSSOR, err

	def ConjugateGradientsHS(self):
		flops = 0
		matrixDots = 0
		vectorAddSub = 0
		vectorDotVector = 0

		M = self.M
		b = self.b
		actualSol = self.actualSol

		avoidDivByZeroError = 0.0000000001
		errorDataConjugateGradients = []
		x = np.zeros_like(b, dtype=np.float)
		r = np.subtract(b, M.dot(x))
		d = np.copy(r)
		matrixDots += 1
		vectorAddSub += 1

		convergence = False
		beta_numerator = r.dot(r)
		vectorDotVector += 1

		while(not convergence):
			solutionError = np.subtract(M.dot(x), b)
			relativeResidualErr = np.linalg.norm(solutionError) / np.linalg.norm(b)

			if(actualSol != []):
				err = np.subtract(actualSol, x)
				absErr = np.linalg.norm(err)
				errorDataConjugateGradients.append(math.log(absErr))
			else:
				errorDataConjugateGradients.append(math.log(relativeResidualErr))

			if(relativeResidualErr < tol):
				convergence = True
				break

			Md = M.dot(d)
			alpha_numerator = beta_numerator
			alpha_denominator = d.dot(Md)
			vectorDotVector += 1
			matrixDots += 1

			if(alpha_denominator < avoidDivByZeroError):
				convergence = True
				break

			alpha = 1.0 * alpha_numerator / alpha_denominator
			flops += 1

			x = np.add(x, np.multiply(d, alpha))
			flops += len(d)
			vectorAddSub += 1

			r_new = np.subtract(r, np.multiply(Md, alpha))
			flops += len(Md)
			vectorAddSub += 1

			beta_numerator = r_new.dot(r_new)
			beta_denominator = alpha_numerator
			vectorDotVector += 1

			if(beta_denominator < avoidDivByZeroError):
				convergence = True
				break

			beta = 1.0 * beta_numerator / beta_denominator
			flops += 1

			d = r_new + np.multiply(d, beta)
			vectorAddSub += 1
			flops += len(d)

			r = r_new

		nonZero = M.nonzero()
		NNZ = len(nonZero[0])
		flops += vectorAddSub * len(x) + vectorDotVector * (2 * len(x) - 1) + matrixDots * (2 * NNZ - len(x))
		print("Iterations: ", len(errorDataConjugateGradients) - 1)
		print("Flops: ", flops)
		return x, relativeResidualErr, errorDataConjugateGradients

	def SteepestDescent(self):
		avoidDivByZeroError = 0.0000000000000000001

		flops = 0
		matrixDots = 0
		vectorAddSub = 0
		vectorDotVector = 0
		actualSol = self.actualSol

		M = self.M
		b = self.b

		x = np.zeros_like(b)
		r = np.subtract(b, M.dot(x))
		matrixDots += 1
		vectorAddSub += 1

		errorDataSteepestDescent = []
		iterationConstant = self.iterationConstant
		for i in range(iterationConstant):
			err = np.subtract(M.dot(x), b)
			divide = np.linalg.norm(b)
			if(actualSol != []):
				err = np.subtract(x, actualSol)
				divide = 1.0
			absErr = np.linalg.norm(err) / divide
			errorDataSteepestDescent.append(math.log(absErr))

			alpha_numerator = r.dot(r)	
			alpha_denominator = r.dot(M.dot(r))
			vectorDotVector += 2
			matrixDots +=1

			if(alpha_denominator < avoidDivByZeroError):
				break

			alpha = alpha_numerator / alpha_denominator
			flops += 1

			x = np.add(x, np.dot(r, alpha))
			flops += len(x)
			vectorAddSub+=1

			r = np.subtract(b, M.dot(x))
			vectorAddSub += 1
			matrixDots += 1

			if(np.linalg.norm(r) / np.linalg.norm(b) < tol):
				break

		NNZ = M.getnnz()
		flops += vectorAddSub * len(x) + vectorDotVector * (2 * len(x) - 1) + matrixDots * (2 * NNZ - len(x))

		err = np.subtract(M.dot(x), b)
		divide = np.linalg.norm(b)
		if(actualSol != []):
			err = np.subtract(x, actualSol)
			divide = 1.0
		absErr = np.linalg.norm(err) / divide
		errorDataSteepestDescent.append(math.log(absErr))
		print("Flops: ",flops)
		print("Iterations: ", len(errorDataSteepestDescent) - 1)
		return x, absErr, errorDataSteepestDescent
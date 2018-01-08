import numpy as np
from scipy.sparse import *
from scipy import *
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import SimpleEquationDiscretizer as sed
import SolverMethods as sm
import FunctionExamples as fe

tol = 0.000001

class MultiGrid:

	def __init__(self, borderFunction, valueFunction, niu1 = 3, niu2 = 3):
		self.borderFunction = borderFunction
		self.valueFunction = valueFunction
		self.niu1 = niu1
		self.niu2 = niu2


	def getCoordinates(self, row, N):
	    return int(row / (N + 1)), row % (N + 1)

	def getRow(self, i, j, N):
	    return(i * (N + 1) + j)


	def restrict(self, r, fineN, coarseN):
		restr = []

		for(i, elem) in enumerate(r):
			(x, y) = self.getCoordinates(i, fineN)

			if(x%2 == 0 and y%2 == 0):
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
				index1 = self.getRow((x - 1) / 2, y / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, y / 2, coarseN)
				interp.append((r[index1] + r[index2]) / 2.0)
			elif(x % 2 == 0 and y % 2 == 1):
				index1 = self.getRow(x / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow(x / 2, (y + 1) / 2, coarseN)
				interp.append((r[index1] + r[index2]) / 2.0)
			else:
				index1 = self.getRow((x - 1) / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, (y - 1) / 2, coarseN)
				index3 = self.getRow((x - 1) / 2, (y + 1) / 2, coarseN)
				index4 = self.getRow((x + 1) / 2, (y + 1) / 2, coarseN)
				interp.append((r[index1] + r[index2] + r[index3] + r[index4]) / 4.0)

		return interp


	def restrictTransposeAction(self, r, fineN, coarseN):
		restr = []

		for i in range((coarseN + 1) * (coarseN + 1)):
			(x, y) = self.getCoordinates(i, coarseN)
			(x, y) = (2 * x, 2 * y)
			newEntry = r[self.getRow(x, y, fineN)]

			for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.5 * r[index]

			for (dX, dY) in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.25 * r[index]

			newEntry = newEntry * 0.25
			restr.append(newEntry)

		return restr


	def vcycle(self, N, f, initSol = []):
		discr = sed.SimpleEquationDiscretizer(N, self.borderFunction, self.valueFunction)

		fSize = len(f)
		if(fSize < 20):
			v = la.solve(discr.M.todense(), f)
			return v

		solver1 = sm.SolverMethods(self.niu1, discr, f, initSol)
		v, _, _, _ = solver1.SSORIterate()

		assert(N % 2 == 0)
		coarseN = N  / 2

		Mv = discr.M.dot(v)
		residual = np.subtract(np.array(f), Mv)
		coarseResidual = self.restrictTransposeAction(residual, N, coarseN)

		coarseV = self.vcycle(coarseN, coarseResidual)
		fineV = self.interpolate(coarseV, N, coarseN)
		v = np.add(v, fineV)

		solver2 = sm.SolverMethods(self.niu2, discr, f, v)
		v2, _, _, _ = solver2.SSORIterate()
		return v2

	def iterateVCycles(self, N, t):
		initSol = []
		vErrors = []
		discr = sed.SimpleEquationDiscretizer(N, self.borderFunction, self.valueFunction)
		f = discr.valueVector2D

		for i in range(t):
			currSol = self.vcycle(N, f, initSol)

			err = np.subtract(discr.M.dot(currSol), discr.valueVector2D)
			absErr = 1.0 * np.linalg.norm(err) / np.linalg.norm(discr.valueVector2D)
			vErrors.append(math.log(absErr))
			if(absErr < tol):
				break

			initSol = currSol

		return currSol, vErrors


class MultiGridAsPreconditioner:

	def __init__(self, borderFunction, valueFunction, maxN, bVector = [], niu1 = 3, niu2 = 3):
		self.borderFunction = borderFunction
		self.valueFunction = valueFunction
		self.niu1 = niu1
		self.niu2 = niu2
		self.maxN = maxN
		self.bVector = bVector
		self.discrLevel = []

		i = 0
		while(maxN >= 2):
			assert(maxN % 2 == 0)
			self.discrLevel.append(sed.SimpleEquationDiscretizer(maxN, self.borderFunction, self.valueFunction))
			i += 1
			maxN /= 2


	def getCoordinates(self, row, N):
	    return int(row / (N + 1)), row % (N + 1)

	def getRow(self, i, j, N):
	    return(i * (N + 1) + j)


	def interpolate(self, r, fineN, coarseN):
		interp = []

		for i in range((fineN + 1) * (fineN + 1)):
			(x, y) = self.getCoordinates(i, fineN)
			if(x % 2 == 0 and y % 2 == 0):
				index = self.getRow(x / 2, y / 2, coarseN)
				interp.append(r[index])
			elif(x % 2 == 1 and y % 2 == 0):
				index1 = self.getRow((x - 1) / 2, y / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, y / 2, coarseN)
				interp.append((r[index1] + r[index2]) / 2.0)
			elif(x % 2 == 0 and y % 2 == 1):
				index1 = self.getRow(x / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow(x / 2, (y + 1) / 2, coarseN)
				interp.append((r[index1] + r[index2]) / 2.0)
			else:
				index1 = self.getRow((x - 1) / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, (y - 1) / 2, coarseN)
				index3 = self.getRow((x - 1) / 2, (y + 1) / 2, coarseN)
				index4 = self.getRow((x + 1) / 2, (y + 1) / 2, coarseN)
				interp.append((r[index1] + r[index2] + r[index3] + r[index4]) / 4.0)

		return interp


	def restrictTransposeAction(self, r, fineN, coarseN):
		restr = []

		for i in range((coarseN + 1) * (coarseN + 1)):
			(x, y) = self.getCoordinates(i, coarseN)
			(x, y) = (2 * x, 2 * y)
			newEntry = r[self.getRow(x, y, fineN)]

			for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.5 * r[index]

			for (dX, dY) in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.25 * r[index]

			newEntry = newEntry * 0.25
			restr.append(newEntry)

		return restr


	def vcycle(self, N, level, f, initSol = []):

		discr = self.discrLevel[level]
		fSize = len(f)
		if(fSize < 20):
			v = la.solve(discr.M.todense(), f)
			return v

		solver1 = sm.SolverMethods(self.niu1, discr, f, initSol)
		v, _, _, _ = solver1.SSORIterate()

		assert(N % 2 == 0)
		coarseN = N  / 2

		Mv = discr.M.dot(v)
		residual = np.subtract(np.array(f), Mv)
		coarseResidual = self.restrictTransposeAction(residual, N, coarseN)

		coarseV = self.vcycle(coarseN, level + 1, coarseResidual)
		fineV = self.interpolate(coarseV, N, coarseN)
		v = np.add(v, fineV)

		solver2 = sm.SolverMethods(self.niu2, discr, f, v)
		v2, _, _, _ = solver2.SSORIterate()
		return v2

	def iterateVCycles(self, N, t):
		initSol = []
		vErrors = []
		discr = sed.SimpleEquationDiscretizer(N, self.borderFunction, self.valueFunction)

		if(self.bVector == []):
			f = discr.valueVector2D
		else:
			f = self.bVector

		for i in range(t):
			currSol = self.vcycle(N, 0, f, initSol)

			err = np.subtract(discr.M.dot(currSol), f)
			absErr = np.linalg.norm(err) / np.linalg.norm(f)
			vErrors.append(math.log(absErr))

			if(absErr < tol):
				break

			initSol = currSol

		return currSol, vErrors


def ConjugateGradientsHS(borderFunction, valueFunction, N):
	discr = sed.SimpleEquationDiscretizer(N, borderFunction, valueFunction)
	M = discr.M
	b = discr.valueVector2D

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

def MultiGridPrecondCG(borderFunction, valueFunction, N):
	avoidDivByZeroError = 0.000000000000000000001
	errorDataMGCG = []

	mg = MultiGridAsPreconditioner(borderFunction, valueFunction, N)
	f = mg.discrLevel[0].valueVector2D
	M = mg.discrLevel[0].M

	x = np.zeros_like(f, dtype = np.float)
	r = np.subtract(f, M.dot(x))

	mg.bVector = r
	rTilda, _ = mg.iterateVCycles(N, 2)
	rTilda = np.array(rTilda)

	p = np.copy(rTilda)

	convergence = False

	while(not convergence):
		solutionError = np.subtract(M.dot(x), f)
		absErr = 1.0 * np.linalg.norm(solutionError) / np.linalg.norm(f)
		errorDataMGCG.append(math.log(absErr))
		print(absErr)

		if(absErr < tol):
			convergence = True
			break

		alpha_numerator = rTilda.dot(r)
		alpha_denominator = p.dot(M.dot(p))

		if(alpha_denominator < avoidDivByZeroError):
			convergence = True
			break

		alpha = 1.0 * alpha_numerator / alpha_denominator

		x = np.add(x, np.multiply(p, alpha))

		newR = np.subtract(r, np.multiply(M.dot(p), alpha))

		mg.bVector = newR
		newR_tilda, _ = mg.iterateVCycles(N, 2)
		newR_tilda = np.array(newR_tilda)

		beta_numerator = newR_tilda.dot(newR)
		beta_denominator = rTilda.dot(r)

		if(beta_denominator < avoidDivByZeroError):
			convergence = True
			break

		beta = 1.0 * beta_numerator / beta_denominator
		p = newR_tilda + np.multiply(p, beta)

		r = newR
		rTilda = newR_tilda

	return x, absErr, errorDataMGCG



def JacobiPrecondCG(borderFunction, valueFunction, N):
	avoidDivByZeroError = 0.000000000000000000001
	errorDataMGCG = []

	mg = sed.SimpleEquationDiscretizer(N, borderFunction, valueFunction)
	solver = sm.SolverMethods(5, mg)
	f = mg.valueVector2D
	M = mg.M

	x = np.zeros_like(f, dtype = np.float)
	r = np.subtract(f, M.dot(x))

	solver.b = r
	rTilda, _ , _, _= solver.JacobiIterate(0.2)
	rTilda = np.array(rTilda)

	p = np.copy(rTilda)

	convergence = False

	while(not convergence):
		solutionError = np.subtract(M.dot(x), f)
		absErr = 1.0 * np.linalg.norm(solutionError) / np.linalg.norm(f)
		errorDataMGCG.append(math.log(absErr))
		print(absErr)

		if(absErr < tol):
			convergence = True
			break

		alpha_numerator = rTilda.dot(r)
		alpha_denominator = p.dot(M.dot(p))

		if(alpha_denominator < avoidDivByZeroError):
			convergence = True
			break

		alpha = 1.0 * alpha_numerator / alpha_denominator

		x = np.add(x, np.multiply(p, alpha))

		newR = np.subtract(r, np.multiply(M.dot(p), alpha))

		solver.b = newR
		newR_tilda, _, _, _ = solver.JacobiIterate(0.2)
		newR_tilda = np.array(newR_tilda)

		beta_numerator = newR_tilda.dot(newR)
		beta_denominator = rTilda.dot(r)

		if(beta_denominator < avoidDivByZeroError):
			convergence = True
			break

		beta = 1.0 * beta_numerator / beta_denominator
		p = newR_tilda + np.multiply(p, beta)

		r = newR
		rTilda = newR_tilda

	return x, absErr, errorDataMGCG



# N = int(input("Enter inverse of coordinates sample rate for the coarser grid\n"))

# sinEquationDiscr = SimpleEquationDiscretizer(N, sinBorderFunction, sinValueFunction)

for N in [4,8,16,32]:
	xSolPrecond, errPrecond, errDataPrecond = MultiGridPrecondCG(fe.sinBorderFunction, fe.sinValueFunction, N)
	plt.plot(errDataPrecond, label=str(N))


# xSol, err, errData = ConjugateGradientsHS(sinBorderFunction, sinValueFunction, N)
# plt.plot(errData, label="Simple CG")

plt.legend(loc='upper right')
plt.show()


# h = 1.0 / N
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# x = y = np.arange(0.0, 1.0 + h, h)
# X, Y = np.meshgrid(x, y)

# Z = np.reshape(xSol, (N+1, N+1))
# ax.plot_wireframe(X, Y, Z)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


# solver = sm.SolverMethods(100, sinEquationDiscr)
# (xFine, absErr, errorDataJacobi, rFine) = solver.JacobiIterate()

# print(errorDataJacobi)
# N = 64
# n = 2
# i = 0
# fig = plt.figure()

# while(n < N):
# 	i = i + 1
# 	n = n * 2
# 	h = 1.0 / n
# 	ax = fig.add_subplot(410 + i, projection='3d')
# 	ax.title.set_text("N = " + str(n))
# 	x = y = np.arange(0.0, 1.0 + h, h)
# 	X, Y = np.meshgrid(x, y)

# 	sinEquationDiscr = SimpleEquationDiscretizer(n, sinBorderFunction, sinValueFunction)
# 	mg = MultiGrid(sinBorderFunction, sinValueFunction)
# 	solMG, vErrors = mg.iterateVCycles(n, 350)
# 	Z = np.reshape(solMG, (n+1, n+1))

# 	ax.plot_wireframe(X, Y, Z)

# 	ax.set_xlabel('X Label')
# 	ax.set_ylabel('Y Label')
# 	ax.set_zlabel('Z Label')
# 	print('last error:', math.exp(vErrors[len(vErrors) - 1]))

	# solMG = mg.vcycle(N, sinEquationDiscr.valueVector2D)
	# print(solMG)
	# print('last error:', math.exp(vErrors[len(vErrors) - 1]))
	# plt.plot(vErrors, label = str(n))
	# plt.legend(loc='upper right')


# while(n < N):
# 	n = n * 2
# 	# eqDiscr = SimpleEquationDiscretizer(n, borderFunction1, valueFunction1)
# 	mg = MultiGrid(sinBorderFunction, sinValueFunction )
# 	solMG, vErrors = mg.iterateVCycles(n, 600)
# 	# solMG = mg.vcycle(N, sinEquationDiscr.valueVector2D)
# 	# print(solMG)
# 	print('last error:', math.exp(vErrors[len(vErrors) - 1]))
# 	plt.plot(vErrors, label = str(n))
# 	plt.legend(loc='upper right')
# 	print(vErrors)

# plt.show()




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
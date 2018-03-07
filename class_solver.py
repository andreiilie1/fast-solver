import numpy as np
from scipy.sparse import *
from scipy import *
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import SimpleEquationDiscretizer as sed
import EquationDiscretizer1D as sed1D
import SolverMethods as sm
import FunctionExamples as fe
import TimeEquationDiscretizer as ted

tol = 0.000001

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


class MultiGrid2D:

	def __init__(self, maxN, borderFunction, valueFunction, niu1 = 4, niu2 = 4):
		self.borderFunction = borderFunction
		self.valueFunction = valueFunction
		self.niu1 = niu1
		self.niu2 = niu2
		self.discrLevel = []
		self.N = maxN

		i = 0
		while(maxN > 2):
			assert(maxN % 2 == 0)
			self.discrLevel.append(sed.SimpleEquationDiscretizer(maxN, self.borderFunction, self.valueFunction))
			i += 1
			maxN /= 2


	def getCoordinates(self, row, N):
		return int(row / (N + 1)), row % (N + 1)

	def getRow(self, i, j, N):
		return(i * (N + 1) + j)

	def restrict(self, r, fineN, coarseN):
		restr = []

		for(i, elem) in enumerate(r):
			(x, y) = self.getCoordinates(i, fineN)

			if(x % 2 == 0 and y % 2 == 0):
				restr.append(elem)

		return restr


	def interpolate(self, r, fineN, coarseN):
		interp = []
		for i in range((fineN + 1) * (fineN + 1)):
			(x, y) = self.getCoordinates(i, fineN)
			
			if(x % 2 == 0 and y % 2 == 0):
				index = self.getRow(x / 2, y / 2, coarseN)
				value = r[index]

			elif(x % 2 == 1 and y % 2 == 0):
				index1 = self.getRow((x - 1) / 2, y / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, y / 2, coarseN)
				value = (r[index1] + r[index2]) / 2.0

			elif(x % 2 == 0 and y % 2 == 1):
				index1 = self.getRow(x / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow(x / 2, (y + 1) / 2, coarseN)
				value = (r[index1] + r[index2]) / 2.0

			else:
				index1 = self.getRow((x - 1) / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, (y - 1) / 2, coarseN)
				index3 = self.getRow((x - 1) / 2, (y + 1) / 2, coarseN)
				index4 = self.getRow((x + 1) / 2, (y + 1) / 2, coarseN)
				value = (r[index1] + r[index2] + r[index3] + r[index4]) / 4.0

			if(x == 0 or y == 0 or x == fineN or y == fineN):
				value = 0

			interp.append(value)

		# plotGraph(fineN, interp)

		return interp

	def restrictTransposeAction(self, r, fineN, coarseN):
		restr = []

		for i in range((coarseN + 1) * (coarseN + 1)):
			(x, y) = self.getCoordinates(i, coarseN)
			(x, y) = (2 * x, 2 * y)
			newEntry = r[self.getRow(x, y, fineN)]

			divideFactor = 1.0

			for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.5 * r[index]
					divideFactor += 0.5

			for (dX, dY) in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.25 * r[index]
					divideFactor += 0.25
			
			newEntry = 1.0 * newEntry / divideFactor
			if(divideFactor < 4.0):
				if(not(x == 0 or y == 0 or x == fineN or y == fineN)):
					print("Error1")

			if(x == 0 or y == 0 or x == fineN or y == fineN):
				newEntry = 0.0

			restr.append(newEntry)

		# plotGraph(coarseN, restr)

		return restr


	def vcycle(self, N, level, f, initSol = [], omega = 1.95):
		discr = self.discrLevel[level]
		fSize = len(f)

		if(level == len(self.discrLevel) - 1):
			v = la.solve(discr.M.todense(), f)
			return v

		else:
			solver1 = sm.SolverMethods(
				iterationConstant = self.niu1, 
				eqDiscretizer = discr, 
				b = f, 
				initSol = initSol,
				)

			# omega = 2.0/(1.0 + math.sin(math.pi/N))
			omega = 1.95

			v, _, _, _ = solver1.SSORIterate(omega)

			coarseN = N  / 2

			Mv = discr.M.dot(v)

			residual = np.subtract(f, Mv)

			coarseResidual = self.restrict(residual, N, coarseN)

			coarseV = self.vcycle(coarseN, level + 1, coarseResidual)

			fineV = self.interpolate(coarseV, N, coarseN)
			w = np.add(v, fineV)

			solver2 = sm.SolverMethods(
				iterationConstant = self.niu2, 
				eqDiscretizer = discr, 
				b = f, 
				initSol = w,
				)

			v2, _, _, _ = solver2.SSORIterate(omega)
			return v2

	def iterateVCycles(self, t):
		initSol = []
		N = self.N

		for i in range((N + 1) * (N + 1)):
			(x, y) = self.getCoordinates(i, N)
			if(x == 0 or y == 0 or x == N or y == N):
				initSol.append(self.borderFunction(1.0 * x / N, 1.0 * y / N))
			else:
				initSol.append(0.0)

		vErrors = []
		discr = self.discrLevel[0]
		f = np.copy(discr.valueVector2D)
		normF = la.norm(f)

		currSol = np.copy(initSol)

		for i in range(t):
			residual = np.subtract(f, discr.M.dot(currSol))
			absErr = 1.0 * la.norm(residual) / (normF + 1)
			vErrors.append(math.log(absErr))

			resSol = self.vcycle(N, 0, residual, np.zeros_like(currSol))
			prevSol = np.copy(currSol)
			currSol = np.add(currSol, resSol)

			if(absErr < tol):
				break
			# if np.allclose(prevSol, currSol, rtol=1e-6):
			# 	 break
			# currSolCopy = np.copy(currSol)
			# nextSol = self.vcycle(N, 0, f, initSol = currSolCopy)
			# currSol = np.copy(nextSol)

		# solX = np.copy(currSol)

		# for omega in [1.97, 1.96, 1.95, 1.94, 1.93]:
		# 	print(omega, ":")
		# 	print()
		# 	currSol = np.copy(solX)
		# 	vErrors = vErrors [:0]
		# 	for i in range(t - 0):
		# 		print(i + 0)
		# 		residual = np.subtract(f, discr.M.dot(currSol))
		# 		absErr = 1.0 * la.norm(residual) / normF
		# 		vErrors.append(math.log(absErr))

		# 		if(absErr < tol):
		# 			break

		# 		resSol = self.vcycle(N, 0, residual, np.zeros_like(currSol), omega)
		# 		currSol = np.add(currSol, resSol)

		# 	plt.plot(vErrors, label = str(omega))
		
		# plt.legend(loc='upper right')
		# plt.show()
		return currSol, vErrors



class MultiGrid:

	def __init__(self, maxN, borderFunction, valueFunction, niu1 = 4, niu2 = 4):
		self.borderFunction = borderFunction
		self.valueFunction = valueFunction
		self.niu1 = niu1
		self.niu2 = niu2
		self.discrLevel = []
		self.N = maxN

		i = 0
		while(maxN > 2):
			assert(maxN % 2 == 0)
			self.discrLevel.append(sed1D.EquationDiscretizer1D(maxN, self.borderFunction, self.valueFunction))
			i += 1
			maxN /= 2
			# print(is_pos_def(self.discrLevel[i-1].M.todense()))
			# print(maxN * 2, np.linalg.eigvals(self.discrLevel[i-1].M.todense()))



	def getCoordinates(self, row, N):
		return int(row / (N + 1)), row % (N + 1)

	def getRow(self, i, j, N):
		return(i * (N + 1) + j)



	def restrict1D(self, r, fineN, coarseN):
		restr = []
		for(i, elem) in enumerate(r):
			if(i%2 == 0):
				restr.append(elem)
		return restr

	def restrict(self, r, fineN, coarseN):
		restr = []

		for(i, elem) in enumerate(r):
			(x, y) = self.getCoordinates(i, fineN)

			if(x % 2 == 0 and y % 2 == 0):
				restr.append(elem)

		return restr


	def interpolate1D(self, r, fineN, coarseN):
		interp = []
		for i in range(fineN + 1):
			if(i%2 == 0):
				interp.append(r[i/2])
			else:
				interp.append((r[(i-1)/2]+r[(i+1)/2])/2.0)
		return interp

	def interpolate(self, r, fineN, coarseN):
		interp = []
		for i in range((fineN + 1) * (fineN + 1)):
			(x, y) = self.getCoordinates(i, fineN)
			
			if(x % 2 == 0 and y % 2 == 0):
				index = self.getRow(x / 2, y / 2, coarseN)
				value = r[index]

			elif(x % 2 == 1 and y % 2 == 0):
				index1 = self.getRow((x - 1) / 2, y / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, y / 2, coarseN)
				value = (r[index1] + r[index2]) / 2.0

			elif(x % 2 == 0 and y % 2 == 1):
				index1 = self.getRow(x / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow(x / 2, (y + 1) / 2, coarseN)
				value = (r[index1] + r[index2]) / 2.0

			else:
				index1 = self.getRow((x - 1) / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, (y - 1) / 2, coarseN)
				index3 = self.getRow((x - 1) / 2, (y + 1) / 2, coarseN)
				index4 = self.getRow((x + 1) / 2, (y + 1) / 2, coarseN)
				value = (r[index1] + r[index2] + r[index3] + r[index4]) / 4.0

			if(x == 0 or y == 0 or x == fineN or y == fineN):
				value = 0

			interp.append(value)

		# plotGraph(fineN, interp)

		return interp

	def restrictTransposeAction(self, r, fineN, coarseN):
		restr = []

		for i in range((coarseN + 1) * (coarseN + 1)):
			(x, y) = self.getCoordinates(i, coarseN)
			(x, y) = (2 * x, 2 * y)
			newEntry = r[self.getRow(x, y, fineN)]

			divideFactor = 1.0

			for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.5 * r[index]
					divideFactor += 0.5

			for (dX, dY) in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.25 * r[index]
					divideFactor += 0.25
			
			newEntry = 1.0 * newEntry / divideFactor
			if(divideFactor < 4.0):
				if(not(x == 0 or y == 0 or x == fineN or y == fineN)):
					print("Error1")

			if(x == 0 or y == 0 or x == fineN or y == fineN):
				newEntry = 0.0

			restr.append(newEntry)

		# plotGraph(coarseN, restr)

		return restr


	def vcycle(self, N, level, f, initSol = [], omega = 1.95):
		discr = self.discrLevel[level]
		fSize = len(f)

		if(level == len(self.discrLevel) - 1):
			v = la.solve(discr.M.todense(), f)
			return v

		else:
			solver1 = sm.SolverMethods(
				iterationConstant = self.niu1, 
				eqDiscretizer = discr, 
				b = f, 
				initSol = initSol,
				)

			# omega = 2.0/(1.0 + math.sin(math.pi/N))
			# omega = 1.95
			v, _, _, _ = solver1.SSORIterate(omega)
			if(not(v[0] == 0)):
				print("ERROR")
			if(not(v[N] == 0)):
				print("ERROR")
			assert(N % 2 == 0)
			coarseN = N  / 2

			Mv = discr.M.dot(v)

			residual = np.subtract(f, Mv)

			coarseResidual = self.restrict1D(residual, N, coarseN)

			if(not(coarseResidual[0] == 0)):
				print("ERROR")
			if(not(coarseResidual[coarseN] == 0)):
				print("ERROR")

			coarseV = self.vcycle(coarseN, level + 1, coarseResidual)

			if(not(coarseV[0] == 0)):
				print("ERROR")
			if(not(coarseV[coarseN] == 0)):
				print("ERROR")

			fineV = self.interpolate1D(coarseV, N, coarseN)
			w = np.add(v, fineV)

			solver2 = sm.SolverMethods(
				iterationConstant = self.niu2, 
				eqDiscretizer = discr, 
				b = f, 
				initSol = w,
				)

			v2, _, _, _ = solver2.SSORIterate(omega)
			return v2

	def iterateVCycles(self, t):
		initSol = []
		N = self.N

		for i in range((N + 1)):
			x = i
			if(x == 0 or x == N):
				initSol.append(self.borderFunction(1.0 * x / N))
			else:
				initSol.append(0.0)

		vErrors = []
		discr = self.discrLevel[0]
		f = np.copy(discr.valueVector2D)
		normF = la.norm(f)

		currSol = np.copy(initSol)

		for i in range(t):
			residual = np.subtract(f, discr.M.dot(currSol))
			absErr = 1.0 * la.norm(residual) / (normF + 1)
			vErrors.append(math.log(absErr))

			resSol = self.vcycle(N, 0, residual, np.zeros_like(currSol))
			prevSol = np.copy(currSol)
			currSol = np.add(currSol, resSol)

			if(absErr < tol):
				break
			# if np.allclose(prevSol, currSol, rtol=1e-6):
			# 	 break
			# currSolCopy = np.copy(currSol)
			# nextSol = self.vcycle(N, 0, f, initSol = currSolCopy)
			# currSol = np.copy(nextSol)
		print(vErrors)

		# solX = np.copy(currSol)

		# for omega in [1.97, 1.96, 1.95, 1.94, 1.93]:
		# 	print(omega, ":")
		# 	print()
		# 	currSol = np.copy(solX)
		# 	vErrors = vErrors [:0]
		# 	for i in range(t - 0):
		# 		print(i + 0)
		# 		residual = np.subtract(f, discr.M.dot(currSol))
		# 		absErr = 1.0 * la.norm(residual) / normF
		# 		vErrors.append(math.log(absErr))

		# 		if(absErr < tol):
		# 			break

		# 		resSol = self.vcycle(N, 0, residual, np.zeros_like(currSol), omega)
		# 		currSol = np.add(currSol, resSol)

		# 	plt.plot(vErrors, label = str(omega))
		
		# plt.legend(loc='upper right')
		# plt.show()
		return currSol, vErrors


class MultiGridAsPreconditioner:

	def __init__(self, borderFunction, valueFunction, maxN, bVector = [], niu1 = 2 , niu2 = 2):
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


	def restrict(self, r, fineN, coarseN):
		restr = []

		for(i, elem) in enumerate(r):
			(x, y) = self.getCoordinates(i, fineN)

			if(x % 2 == 0 and y % 2 == 0):
				restr.append(elem)

		return restr

	def interpolate(self, r, fineN, coarseN):
		interp = []
		for i in range((fineN + 1) * (fineN + 1)):
			(x, y) = self.getCoordinates(i, fineN)
			
			if(x % 2 == 0 and y % 2 == 0):
				index = self.getRow(x / 2, y / 2, coarseN)
				value = r[index]

			elif(x % 2 == 1 and y % 2 == 0):
				index1 = self.getRow((x - 1) / 2, y / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, y / 2, coarseN)
				value = (r[index1] + r[index2]) / 2.0

			elif(x % 2 == 0 and y % 2 == 1):
				index1 = self.getRow(x / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow(x / 2, (y + 1) / 2, coarseN)
				value = (r[index1] + r[index2]) / 2.0

			else:
				index1 = self.getRow((x - 1) / 2, (y - 1) / 2, coarseN)
				index2 = self.getRow((x + 1) / 2, (y - 1) / 2, coarseN)
				index3 = self.getRow((x - 1) / 2, (y + 1) / 2, coarseN)
				index4 = self.getRow((x + 1) / 2, (y + 1) / 2, coarseN)
				value = (r[index1] + r[index2] + r[index3] + r[index4]) / 4.0

			if(x == 0 or y == 0 or x == fineN or y == fineN):
				value = 0

			interp.append(value)

		return interp

	def restrictTransposeAction(self, r, fineN, coarseN):
		restr = []

		for i in range((coarseN + 1) * (coarseN + 1)):
			(x, y) = self.getCoordinates(i, coarseN)
			(x, y) = (2 * x, 2 * y)
			newEntry = r[self.getRow(x, y, fineN)]

			divideFactor = 1.0

			for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.5 * r[index]
					divideFactor += 0.5

			for (dX, dY) in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
				newX = x + dX
				newY = y + dY
				if(0 <= newX and newX <= fineN and 0 <= newY and newY <= fineN):
					index = self.getRow(newX, newY, fineN)
					newEntry += 0.25 * r[index]
					divideFactor += 0.25
			
			newEntry = 1.0 * newEntry / divideFactor
			if(divideFactor < 4.0):
				if(not(x == 0 or y == 0 or x == fineN or y == fineN)):
					print("Error1")

			if(x == 0 or y == 0 or x == fineN or y == fineN):
				newEntry = 0.0

			restr.append(newEntry)

		return restr


	def vcycle(self, N, level, f, initSol = []):

		discr = self.discrLevel[level]
		fSize = len(f)
		if(fSize < 20):
			v = la.solve(discr.M.todense(), f)
			return v

		omega = 1.95

		solver1 = sm.SolverMethods(self.niu1, discr, f, initSol)
		v, _, _, _ = solver1.SSORIterate(omega)

		assert(N % 2 == 0)
		coarseN = N  / 2

		Mv = discr.M.dot(v)
		residual = np.subtract(np.array(f), Mv)
		coarseResidual = self.restrictTransposeAction(residual, N, coarseN)

		coarseV = self.vcycle(coarseN, level + 1, coarseResidual)
		fineV = self.interpolate(coarseV, N, coarseN)
		v = np.add(v, fineV)

		solver2 = sm.SolverMethods(self.niu2, discr, f, v)
		v2, _, _, _ = solver2.SSORIterate(omega)
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


def MultiGridPrecondCG(borderFunction, valueFunction, N):
	avoidDivByZeroError = 0.000000000000000000001
	errorDataMGCG = []

	mg = MultiGridAsPreconditioner(borderFunction, valueFunction, N)
	f = mg.discrLevel[0].valueVector2D
	M = mg.discrLevel[0].M

	x = np.zeros_like(f, dtype = np.float)
	r = np.subtract(f, M.dot(x))

	mg.bVector = r
	rTilda, _ = mg.iterateVCycles(N, 1)
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
		newR_tilda, _ = mg.iterateVCycles(N, 1)
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


#Idea : compute finer solutions as we advance in the timestep
# i.e. 100 iterations for t=0, 120 for t = 1, 140 for t = 2, etc.
def solveHeatEquationForAllTimeSteps(discr):
	solHeat = discr.initialHeatTimeSolution()
	valueVector = discr.computeVectorAtTimestep(1, solHeat)
	solver = sm.SolverMethods(1000, discr, valueVector)
	sol =[solHeat]

	for k in range(1, discr.T + 1):
		t = k * discr.dT
		valueVector = discr.computeVectorAtTimestep(k, solHeat)
		solver.b = valueVector
		(solHeat, err, _) = solver.ConjugateGradientsHS()
		sol.append(solHeat)
		print(err)

	return sol

# ______________________________________________________________
# DEBUG TOOLS



def testHeatEquationSolver():
	discrTest = ted.TimeEquationDiscretizer(32,32,fe.heatSinBorderFunction, fe.heatRhsFunction, fe.heatInitialFunction)
	sol = solveHeatEquationForAllTimeSteps(discrTest)
	xSol = sol[16]
	return xSol

def testMGCG():
	for N in [4,8,16,32]:
		xSolPrecond, errPrecond, errDataPrecond = MultiGridPrecondCG(fe.sinBorderFunction, fe.sinValueFunction, N)
		plt.plot(errDataPrecond, label=str(N))
		plt.legend(loc='upper right')

	plt.show()

def plotGraph(N, valuesVector):
	h = 1.0 / N
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	x = y = np.arange(0.0, 1.0 + h, h)
	X, Y = np.meshgrid(x, y)

	Z = np.reshape(valuesVector, (N+1, N+1))
	ax.plot_wireframe(X, Y, Z)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()


def testMG2D():
	N = 128
	n = 4
	i = 0
	while(n < N):
		n = n * 2
		mg = MultiGrid2D(n, fe.sin2BorderFunction, fe.sin2ValueFunction)
		solMG, vErrors = mg.iterateVCycles(1000)
		plt.plot(vErrors, label = str(n))
		plt.legend(loc='upper right')
		print(vErrors)

	plt.show()

def testMG1D():
	N = 128
	n = 2
	i = 0
	while(n < N):
		n = n * 2
		mg = MultiGrid(n, fe.sin1DBorderFunction2, fe.sin1DValueFunction2)
		solMG, vErrors = mg.iterateVCycles(1000)
		plt.plot(vErrors, label = str(n))
		plt.legend(loc='upper right')
		print(vErrors)

	plt.show()


def testDifferentParamIterations1D():
	N = int(input("Enter inverse of coordinates sample rate for the coarser grid\n"))

	
	# for omega in [1.97, 1.98, 2.0/(1.0 + math.sin(math.pi / N))]:
	optSOROmega = 2.0/(1.0 + math.sin(math.pi / N))
	for omega in [optSOROmega, 1.974]:
		print(omega)
		sinEquationDiscr = sed1D.EquationDiscretizer1D(N, fe.sin1DBorderFunction, fe.sin1DValueFunction)

		initSol = []

		# for i in range((N + 1) * (N + 1)):
		# 	(x, y) = sinEquationDiscr.getCoordinates(i)
		# 	if(x == 0 or y == 0 or x == N or y == N):
		# 		initSol.append(sinEquationDiscr.borderFunction(1.0 * x / N, 1.0 * y / N))
		# 	else:
		# 		initSol.append(0.0)

		for i in range((N+1)):
			if(i == 0 or i == N):
				initSol.append(sinEquationDiscr.borderFunction(1.0 * i/ N))
			else:
				initSol.append(0)

		solver = sm.SolverMethods(700, sinEquationDiscr, initSol = initSol)
		(xFine, absErr, errorDataJacobi, rFine) = solver.SSORIterate(omega, debugOn = False)
		plt.plot(errorDataJacobi, label=str(omega))
		print(len(errorDataJacobi))
	plt.legend(loc = "upper right", prop={'size':'15'})
	plt.show()

def testDifferentParamIterations():
	N = int(input("Enter inverse of coordinates sample rate for the coarser grid\n"))

	
	# for omega in [1.97, 1.98, 2.0/(1.0 + math.sin(math.pi / N))]:
	optSOROmega = 2.0/(1.0 + math.sin(math.pi / N))
	for omega in [optSOROmega, 1.852]:
		print(omega)
		sinEquationDiscr = sed.SimpleEquationDiscretizer(N, fe.sinBorderFunction, fe.sinValueFunction)

		initSol = []

		for i in range((N + 1) * (N + 1)):
			(x, y) = sinEquationDiscr.getCoordinates(i)
			if(x == 0 or y == 0 or x == N or y == N):
				initSol.append(sinEquationDiscr.borderFunction(1.0 * x / N, 1.0 * y / N))
			else:
				initSol.append(0.0)

		# for i in range((N+1)):
		# 	if(i == 0 or i == N):
		# 		initSol.append(sinEquationDiscr.borderFunction(1.0 * i/ N))
		# 	else:
		# 		initSol.append(0)

		solver = sm.SolverMethods(300, sinEquationDiscr, initSol = initSol)
		(xFine, absErr, errorDataJacobi, rFine) = solver.SSORIterate(omega)
		plt.plot(errorDataJacobi, label=str(omega))
		print(len(errorDataJacobi))
	plt.legend(loc = "upper right", prop={'size':'15'})
	plt.show()

lineColor = ["red", "green", "blue", "brown", "black", "pink", "gray"]

def testJacobi():
	N = 128
	n = 2
	index = 0

	while(n < N):
		print(n, ':')
		n = 2 * n
		sinEquationDiscr = sed.SimpleEquationDiscretizer(n, fe.sinBorderFunction, fe.sinValueFunction)

		initSol = []

		for i in range((n + 1) * (n + 1)):
			(x, y) = sinEquationDiscr.getCoordinates(i)
			if(x == 0 or y == 0 or x == n or y == n):
				initSol.append(sinEquationDiscr.borderFunction(1.0 * x / n, 1.0 * y / n))
			else:
				initSol.append(0.0)

		solver = sm.SolverMethods(2000, sinEquationDiscr, initSol = initSol)
		# wOpt = 2.0/(1.0 + math.sin(math.pi / n))
		# (x, absErr, errorData, r) = solver.SSORIterate(wOpt)
		(x, absErr, errorData, r) = solver.JacobiIterate()
		# print(np.linalg.eigvals(sinEquationDiscr.M.todense()))
		plt.plot(errorData, label = str(n)+", Jacobi", color=lineColor[index])
		index +=1
	# plt.legend(loc = "upper right", prop={'size':'16'})
	# plt.show()


def testGaussSeidel():
	N = 64
	n = 2
	index = 0

	while(n < N):
		n = 2 * n
		print(n, ':')
		sinEquationDiscr = sed.SimpleEquationDiscretizer(n, fe.sinBorderFunction, fe.sinValueFunction)

		initSol = []

		for i in range((n + 1) * (n + 1)):
			(x, y) = sinEquationDiscr.getCoordinates(i)
			if(x == 0 or y == 0 or x == n or y == n):
				initSol.append(sinEquationDiscr.borderFunction(1.0 * x / n, 1.0 * y / n))
			else:
				initSol.append(0.0)

		solver = sm.SolverMethods(2000, sinEquationDiscr, initSol = initSol)
		wOpt = 2.0/(1.0 + math.sin(math.pi / n))
		# (x, absErr, errorData, r) = solver.SSORIterate(wOpt)
		(x, absErr, errorData, r) = solver.SSORIterate(wOpt)
		# print(np.linalg.eigvals(sinEquationDiscr.M.todense()))
		plt.plot(errorData, label = str(n))
		index += 1

	plt.legend(loc = "upper right", prop={'size':'15'})
	plt.show()


def printDiscretization():
	f1=open('./testfile', 'w+')

	eqDiscr = sed.SimpleEquationDiscretizer(3, fe.sinBorderFunction, fe.sinValueFunction)
	f1.write(str(eqDiscr.M.todense()))

	f1.close()

# printDiscretization()
# testMG2D()
# testMG1D()
# testJacobi()
# testGaussSeidel()

testDifferentParamIterations()
testDifferentParamIterations1D()
# testMGCG()


# xSol, err, errData = ConjugateGradientsHS(sinBorderFunction, sinValueFunction, N)
# plt.plot(errData, label="Simple CG")

# plt.legend(loc='upper right')
# plt.show()
# plotGraph(32, testHeatEquationSolver())


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
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
import MGMethods as MG



wSSOR2D = {}
wSSOR2D[8] = 1.503
wSSOR2D[16] = 1.720
wSSOR2D[32] = 1.852
wSSOR2D[64] = 1.923
wSSOR2D[128] = 1.961


def testHeatEquationSolver():
	discrTest = ted.TimeEquationDiscretizer(32,32,fe.heatSinBorderFunction, fe.heatRhsFunction, fe.heatInitialFunction)
	sol = MG.solveHeatEquationForAllTimeSteps(discrTest)
	xSol = sol[16]
	return xSol

def testMGCG():
	for N in [4,8,16,32, 64, 128]:
		xSolPrecond, errPrecond, errDataPrecond = MG.MultiGridPrecondCG(fe.sinBorderFunction, fe.sinValueFunction, N)
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
	N = 64
	n = 4
	i = 0
	while(n < N):
		n = n * 2
		flops = 0
		mg = MG.MultiGrid2D(n, fe.sin2BorderFunction, fe.sin2ValueFunction)
		print(flops)
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
		mg = MG.MultiGrid(n, fe.sin1DBorderFunction2, fe.sin1DValueFunction2)
		solMG, vErrors = mg.iterateVCycles(1000)
		plt.plot(vErrors, label = str(n))
		plt.legend(loc='upper right')
		print(vErrors)

	plt.show()


def testDifferentParamIterations1D():
	N = int(input("Enter inverse of coordinates sample rate for the coarser grid\n"))

	
	# for omega in [1.97, 1.98, 2.0/(1.0 + math.sin(math.pi / N))]:
	optSOROmega = 2.0/(1.0 + math.sin(math.pi / N))
	for omega in [optSOROmega]:
		print(omega)
		sinEquationDiscr = sed1D.EquationDiscretizer1D(N, fe.sin1DBorderFunction, fe.sin1DValueFunction)

		initSol = []

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
	for omega in [optSOROmega, 1.503]:
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

def testConjugateGradient():
	N = 128
	n = 4
	index = 0
	print("CONJUGATE GRADIENT TEST")
	while(n < N):
		print(2 * n, ':')
		n = 2 * n
		sinEquationDiscr = sed.SimpleEquationDiscretizer(n, fe.sinBorderFunction, fe.sinValueFunction)

		initSol = []
		actualSol = fe.actualSinSolution(n)

		for i in range((n + 1) * (n + 1)):
			(x, y) = sinEquationDiscr.getCoordinates(i)
			if(x == 0 or y == 0 or x == n or y == n):
				initSol.append(sinEquationDiscr.borderFunction(1.0 * x / n, 1.0 * y / n))
			else:
				initSol.append(0.0)

		solver = sm.SolverMethods(2000, sinEquationDiscr, initSol = initSol, actualSol = actualSol)
		# wOpt = 2.0/(1.0 + math.sin(math.pi / n))
		# (x, absErr, errorData, r) = solver.SSORIterate(wOpt)
		(x, absErr, errorData) = solver.ConjugateGradientsHS()
		# print(np.linalg.eigvals(sinEquationDiscr.M.todense()))
		plt.plot(errorData, label = "N = " + str(n))
		index = index + 1
		
	plt.legend(loc = "upper right", prop={'size':'16'})
	plt.show()

def testSteepestDescent():
	N = 128
	n = 4
	index = 0
	print("STEEPEST DESCENT TEST")
	while(n < N):
		print(2*n, ':')
		n = 2 * n
		sinEquationDiscr = sed.SimpleEquationDiscretizer(n, fe.sinBorderFunction, fe.sinValueFunction)

		initSol = []

		for i in range((n + 1) * (n + 1)):
			(x, y) = sinEquationDiscr.getCoordinates(i)
			if(x == 0 or y == 0 or x == n or y == n):
				initSol.append(sinEquationDiscr.borderFunction(1.0 * x / n, 1.0 * y / n))
			else:
				initSol.append(0.0)

		actualSol = fe.actualSinSolution(n)
		solver = sm.SolverMethods(200000, sinEquationDiscr, initSol = initSol, actualSol = [])

		(x, absErr, errorData) = solver.SteepestDescent()
		plt.plot(errorData, label = "N = " + str(n))
		index = index + 1

	plt.legend(loc = "upper right", prop={'size':'16'})
	plt.show()

def testJacobiSmoothing1D():
	N = 32
	h = 1.0 / N
	x = np.arange(0.0, 1.0 + h, h)
	k = 3.0
	initSol = np.sin(math.pi * 8 * x) + np.sin(math.pi * 5 * x) + np.sin(math.pi * 3 *x)
	plt.plot(x,initSol)
	# plt.show()
	sinEquationDiscr = sed1D.EquationDiscretizer1D(N, fe.zero1D, fe.zero1D)
	solver = sm.SolverMethods(20, sinEquationDiscr, initSol = initSol)
	(y, absErr, errorData, r) = solver.JacobiIterate(1.0)
	plt.plot(x,y)
	plt.show()
	print(len(errorData))


def testJacobi():
	N = 128
	n = 2
	index = 0

	while(n < N):
		print(2 * n, ':')
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
	plt.legend(loc = "upper right", prop={'size':'16'})
	plt.show()


def testGaussSeidel():
	N = 128
	n = 4
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
		(x, absErr, errorData, r) = solver.GaussSeidelIterate(wOpt)
		# print(np.linalg.eigvals(sinEquationDiscr.M.todense()))
		plt.plot(errorData, label = str(n) +", SOR", color=lineColor[index])
		index += 1
	plt.show()

def testSSOR(maxN):
	N = maxN
	n = 4
	index = 0
	print("SSOR TEST")
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
		# wOpt = 1.0
		wOpt = wSSOR2D[n]
		(x, absErr, errorData, r) = solver.SSORIterate(wOpt)
		# (x, absErr, errorData, r) = solver.SSORIterate(wSSOR2D[n])
		# doubleSize = list(range(0, 2 * len(errorData), 2))
		# plt.plot(doubleSize, errorData, label = str(n)+", SSOR", color=lineColor[index], linestyle='dashed')
		plt.plot(errorData, label = str(n) + ", SSOR")
		index += 1

	plt.legend(loc = "upper right", prop={'size':'15'})
	plt.show()

def printDiscretization():
	f1=open('./testfile', 'w+')

	eqDiscr = sed.SimpleEquationDiscretizer(3, fe.sinBorderFunction, fe.sinValueFunction)
	f1.write(str(eqDiscr.M.todense()))

	f1.close()

def plotExactSol1D():
	plt.figure(1)
	n = 2
	N = 32
	t = np.arange(0.0, 1.0, 0.01)
	k = 3.0
	s = np.sin(k*np.pi*t)
	index = 0
	while(n < N):
		n = 2 * n
		h = 1.0 / n
		print(n, ':')
		x  = np.arange(0.0, 1.0 + h, h)
		sinEquationDiscr = sed1D.EquationDiscretizer1D(n, fe.sin1DBorderFunction2, fe.sin1DValueFunction2)

		initSol = []

		for i in range((n+1)):
			if(i == 0 or i == n):
				initSol.append(sinEquationDiscr.borderFunction(1.0 * i/ n))
			else:
				initSol.append(0)
		M = sinEquationDiscr.M.todense()
		v = sinEquationDiscr.valueVector2D
		exactSol = np.linalg.solve(M, v)
		index = index +1
		plt.subplot('22'+str(index))
		plt.plot(x,exactSol, 'bo')
		if(index == 1):
			plt.plot(x,exactSol, label = "Linear interpolation of $\mathbf{u}^{4}$")
		if(index == 2):
			plt.plot(x,exactSol, label = "Linear interpolation of $\mathbf{u}^{8}$")
		if(index == 3):
			plt.plot(x,exactSol, label = "Linear interpolation of $\mathbf{u}^{16}$")
		if(index == 4):
			plt.plot(x,exactSol, label = "Linear interpolation of $\mathbf{u}^{32}$")
		plt.plot(t, s, label = "Exact continuous solution: $u$")
		plt.legend(loc = "lower right", prop={'size':'12'})
		plt.title('N = '+str(n))

	plt.show()

testMGCG()
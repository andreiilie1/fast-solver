# testTools.py

#  ________________________________________________________________________________
# |USED FOR CARRYING TESTS ON ALL THE STRUCTURES AND METHODS DEFINED IN THE PROJECT|
# |________________________________________________________________________________|

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

# Default line colors used in some testing utilities
lineColor = ["red", "green", "blue", "brown", "black", "pink", "gray"]

# Empirically discovered approximation of the optimal SSOR parameter for the class of problems
# used in our tests
wSSOR2D = {}
wSSOR2D[8] = 1.503
wSSOR2D[16] = 1.720
wSSOR2D[32] = 1.852
wSSOR2D[64] = 1.923
wSSOR2D[128] = 1.961


# Heat Equation solver testing utility
def testHeatEquationSolver():
	discrTest = ted.TimeEquationDiscretizer(32, 32, fe.heatSinBorderFunction, fe.heatRhsFunction, fe.heatInitialFunction)
	sol = MG.solveHeatEquationForAllTimeSteps(discrTest)
	xSol = sol[16]
	return xSol

# Multigrid Preconditioned Conjugate Gradient solver testing utility
def testMGCG():
	# Niu1 and niu2 are the number of pre and post smoothing steps
	niu1 = 1
	niu2 = 1

	# The paramter of the smoothing iteration used in MG as a preconditioner
	omega = 1.92

	print("Testing MGCG, niu1 = niu2 = ", niu1, ", omega = ", omega)

	for N in [4, 8, 16, 32, 64, 128, 256]:
		print(N)
		xSolPrecond, errPrecond, errDataPrecond, flops = MG.MultiGridPrecondCG(
															fe.sin2BorderFunction, 
															fe.sin2ValueFunction, 
															N, 
															niu1= niu1, 
															niu2= niu2, 
															omega = omega,
														 )
		plt.plot(errDataPrecond, label=str(N))

		print("FLOPS:", flops)

	plt.legend(loc='upper right')
	plt.title("Testing MGCG, niu1 = niu2 = " + str(niu1) + ", omega = " + str(omega))
	plt.show()

# PlotGraph is a helper function which interpolates a 3D graph of a function, given values at the grid knots
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

# Multigrid solver testing utility for 2D problem instances
def testMG2D():
	N = 256
	n = 4

	i = 0

	niu1 = 1
	niu2 = 1

	omega = 1.5
	while(n < N):
		n = n * 2
		# Select omega = 2.0 / (1.0 + math.sin(math.pi / n)) to test the optimal SSOR parameter as a smoothing parameter
		print("Testing MG2D, niu1, niu2 = ", niu1," ", niu2,", omega = ", omega)
		print(n, ':')

		mg = MG.MultiGrid2D(n, fe.sin2BorderFunction, fe.sin2ValueFunction, omega = omega, niu1 = niu1, niu2 = niu2)
		solMG, vErrors, flops = mg.iterateVCycles(1000)

		print("Flops: ",flops)

		plt.plot(vErrors, label = str(n))
		
	plt.title("Testing MG2D, niu1, niu2 = " + str(niu1) + " "+str(niu2) + ", omega = " + str(omega))
	plt.legend(loc='upper right', prop={'size':'16'})
	plt.show()

# Multigrid Preconditioned Conjugate Gradient solver testing utility for 2D problem instances for different smoothing parameters
def testMG2DvarW():
	N = 128
	listOmega = [1.89, 1.9, 1.91, 1.92, 1.93, 1.94, 1.95, 1.96]

	for omega in listOmega:
		mg = MG.MultiGrid2D(N, fe.sin2BorderFunction, fe.sin2ValueFunction, omega = omega, niu1 = 2, niu2 = 2)
		solMG, vErrors, flops = mg.iterateVCycles(1000)

		print(N, ":", flops)
		plt.plot(vErrors, label = str(omega))

	plt.legend(loc='upper right', prop={'size':'16'})
	plt.show()

# Multigrid solver testing utility for 1D problem instances
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
	plt.show()

# Multigrid Conjugate Gradient solver testing utility for 1D problem instances for different smoothing parameters
def testDifferentParamIterations1D():
	N = int(input("Enter inverse of coordinates sample rate for the coarser grid\n"))

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

# SSOR solver testing utility for 2D problem instances for different smoothing parameters
def testDifferentParamIterations():
	N = int(input("Enter inverse of coordinates sample rate for the coarser grid\n"))

	
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

		solver = sm.SolverMethods(300, sinEquationDiscr, initSol = initSol)
		(xFine, absErr, errorDataJacobi, rFine) = solver.SSORIterate(omega)
		plt.plot(errorDataJacobi, label=str(omega))

	plt.legend(loc = "upper right", prop={'size':'15'})
	plt.show()

# Conjugate Gradient testing utility
def testConjugateGradient():
	N = 128
	n = 4
	index = 0
	print("Testing Conjugate Gradient")

	while(n < N):
		n = 2 * n
		print(n)
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

		(x, absErr, errorData) = solver.ConjugateGradientsHS()
		plt.plot(errorData, label = "N = " + str(n))
		index = index + 1
		
	plt.legend(loc = "upper right", prop={'size':'16'})
	plt.show()

# Steepest Descent testing utility
def testSteepestDescent():
	N = 128
	n = 4
	index = 0
	print("Testing Steepest Descent")

	while(n < N):
		n = 2 * n
		print(n)
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

# Jacobi solver testing utility on a 1D model problem
def testJacobiSmoothing1D():
	N = 32
	h = 1.0 / N
	x = np.arange(0.0, 1.0 + h, h)
	k = 3.0
	initSol = np.sin(math.pi * 8 * x) + np.sin(math.pi * 5 * x) + np.sin(math.pi * 3 *x)

	plt.plot(x,initSol)

	sinEquationDiscr = sed1D.EquationDiscretizer1D(N, fe.zero1D, fe.zero1D)
	solver = sm.SolverMethods(20, sinEquationDiscr, initSol = initSol)
	(y, absErr, errorData, r) = solver.JacobiIterate(1.0)

	plt.plot(x,y)
	plt.show()

# Jacobi solver testing utility
def testJacobi():
	N = 128
	n = 4
	index = 0
	print("Testing Jacobi")
	while(n < N):
		n = 2 * n
		print(n)
		sinEquationDiscr = sed.SimpleEquationDiscretizer(n, fe.sinBorderFunction, fe.sinValueFunction)

		initSol = []

		for i in range((n + 1) * (n + 1)):
			(x, y) = sinEquationDiscr.getCoordinates(i)
			if(x == 0 or y == 0 or x == n or y == n):
				initSol.append(sinEquationDiscr.borderFunction(1.0 * x / n, 1.0 * y / n))
			else:
				initSol.append(0.0)

		solver = sm.SolverMethods(2000, sinEquationDiscr, initSol = initSol)
		(x, absErr, errorData, r) = solver.JacobiIterate()
		plt.plot(errorData, label = str(n)+", Jacobi")
		index +=1

	plt.legend(loc = "upper right", prop={'size':'16'})
	plt.show()

# Gauss Seidel solver testing utility
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
		(x, absErr, errorData, r) = solver.GaussSeidelIterate(wOpt)
		plt.plot(errorData, label = str(n) +", SOR", color=lineColor[index])
		index += 1

	plt.show()

# SSOR solver testing utility
def testSSOR():
	N = 256
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
		wOpt = wSSOR2D[n]

		(x, absErr, errorData, r) = solver.SSORIterate(wOpt)

		plt.plot(errorData, label = str(n) + ", SSOR")
		index += 1

	plt.legend(loc = "upper right", prop={'size':'15'})
	plt.show()

# Helper function which prints the resulting discretization matrix in dense format
# Note: only use this for small values, otherwise making a dense matrix from our
# sparse format becomes too expensive
def printDiscretization():
	f1=open('./testfile', 'w+')

	eqDiscr = sed.SimpleEquationDiscretizer(3, fe.sinBorderFunction, fe.sinValueFunction)
	f1.write(str(eqDiscr.M.todense()))

	f1.close()

# Helper function to plot the exact solution of a 1D model problem
def plotExactSol1D():
	plt.figure(1)
	n = 2
	N = 32
	t = np.arange(0.0, 1.0, 0.01)
	k = 3.0
	s = np.sin(k * np.pi * t)
	index = 0
	while(n < N):
		n = 2 * n
		h = 1.0 / n
		print(n, ':')
		x  = np.arange(0.0, 1.0 + h, h)
		sinEquationDiscr = sed1D.EquationDiscretizer1D(n, fe.sin1DBorderFunction2, fe.sin1DValueFunction2)

		initSol = []

		for i in range(n + 1):
			if(i == 0 or i == n):
				initSol.append(sinEquationDiscr.borderFunction(1.0 * i / n))
			else:
				initSol.append(0)
		M = sinEquationDiscr.M.todense()
		v = sinEquationDiscr.valueVector2D
		exactSol = np.linalg.solve(M, v)
		index = index + 1
		plt.subplot('22'+str(index))
		plt.plot(x,exactSol, 'bo')
		if(index == 1):
			plt.plot(x,exactSol, label = "Linear interpolation of $\mathbf{u}^{4}$")

		elif(index == 2):
			plt.plot(x,exactSol, label = "Linear interpolation of $\mathbf{u}^{8}$")

		elif(index == 3):
			plt.plot(x,exactSol, label = "Linear interpolation of $\mathbf{u}^{16}$")

		elif(index == 4):
			plt.plot(x,exactSol, label = "Linear interpolation of $\mathbf{u}^{32}$")

		plt.plot(t, s, label = "Exact continuous solution: $u$")
	
	plt.legend(loc = "lower right", prop={'size':'12'})
	plt.title('N = '+str(n))
	plt.show()

# Backward Euler method for solving the Heat Equation (which includes the time variable)
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
		plotGraph(discr.N, solHeat)

	return sol

# Heat Equation solver testing utility
def testHeatEquationSolver():
	discr = ted.TimeEquationDiscretizer(
		N = 32, 
		T = 6, 
		borderTimeFunction = fe.heatSinBorderFunction,
		rhsHeatEquationFunction = fe.heatRhsFunction,
		initialHeatTimeFunction = fe.heatInitialFunction,
	)

	solveHeatEquationForAllTimeSteps(discr)
# ___________________________________
# USED FOR AUXILLIARY PLOTS IN THE THESIS

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math

tol = 0.00001

A = np.array([[2.0, 1.0], [1.0,3.0]], dtype = np.float)
b = np.array([5.0,5.0], dtype = np.float)

def f(x, y):
	return (1 * x ** 2 +  1.5 * y ** 2 + 1 * x*y - 5 * x - 5 * y)

ax = plt.axes(projection='3d')
# plt.axis('off')
def plotF():
	x = np.linspace(0.8, 2.5, 20)
	y = np.linspace(0.8, 2, 20)

	X, Y = np.meshgrid(x, y)
	Z = f(X, Y)
	fig = plt.figure()

	ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
	                cmap='viridis', edgecolor='none')

	# ax.plot([0,1],[0,1],[0,1])
	# plt.show()

Xs = []
def SteepestDescent(M, b, iterationConstant = 100, plotFirstIterations = False):
	# ax = plt.axes()
	plt.xlim(-2, 3)
	plt.ylim(-2, 2)
	avoidDivByZeroError = 0.0000000000000000001
	x = np.ones_like(b)
	r = np.subtract(b, M.dot(x))
	errorDataSteepestDescent = []
	iterationConstant = iterationConstant
	for i in range(iterationConstant):
		Xs.append(x)

		err = np.subtract(M.dot(x), b)
		absErr = np.linalg.norm(err) / np.linalg.norm(b)
		errorDataSteepestDescent.append(math.log(absErr))

		alpha_numerator = r.dot(r)	
		alpha_denominator = r.dot(M.dot(r))
		if(alpha_denominator < avoidDivByZeroError):
			break
		alpha = alpha_numerator / alpha_denominator

		xOld = np.copy(x)
		x = np.add(x, np.dot(r, alpha))

		# Used for plotting first iterations of Steepest descent
		fSize = 26
		strCoord = "("+format(xOld[0], ".2f")+", "+ format(xOld[1], ".2f")+", "+format(f(xOld[0], xOld[1]), ".3f")+")"
		if(i>=0):
			ax.plot([xOld[0], x[0]], [xOld[1], x[1]], [f(xOld[0], xOld[1]), f(x[0], x[1])], color="k", linewidth = 3)
		if(plotFirstIterations):
			if(i == 0 ):
				ax.plot([xOld[0]],[xOld[1]],[f(xOld[0], xOld[1])], markerfacecolor='white', markeredgecolor='white', marker='o', markersize=4, alpha=1)
				ax.text(xOld[0],xOld[1] +0.1,f(xOld[0], xOld[1]),"$\mathbf{x}_0$" + strCoord, fontsize = fSize, color = "white")
			if(i == 1 ):
				ax.plot([xOld[0]],[xOld[1]],[f(xOld[0], xOld[1])], markerfacecolor='white', markeredgecolor='white', marker='o', markersize=4, alpha=1)
				ax.text(xOld[0],xOld[1],f(xOld[0], xOld[1]),"$\mathbf{x}_1$" + strCoord, fontsize = fSize, color = "white")
			if(i == 2 ):
				ax.plot([xOld[0]],[xOld[1]],[f(xOld[0], xOld[1])], markerfacecolor='white', markeredgecolor='white', marker='o', markersize=4, alpha=1)
				ax.text(xOld[0]- 0.1,xOld[1]- 0.1,f(xOld[0], xOld[1]),"$\mathbf{x}_2$" + strCoord, fontsize = fSize, color = "white")
			if(i == 3):
				ax.plot([xOld[0]],[xOld[1]],[f(xOld[0], xOld[1])], markerfacecolor='white', markeredgecolor='white', marker='o', markersize=4, alpha=1)
				ax.text(xOld[0],xOld[1]+0.01,f(xOld[0], xOld[1]),"$\mathbf{x}_3$" + strCoord, fontsize = fSize, color = "white")


		r = np.subtract(b, M.dot(x))
		if(np.linalg.norm(r) < tol):
			break

	err = np.subtract(M.dot(x), b)
	Xs.append(x)
	absErr = np.linalg.norm(err) / np.linalg.norm(b)
	errorDataSteepestDescent.append(math.log(absErr))

	#Plot solution for model problem
	ax.plot([2.0],[1.0],[f(2.0, 1.0)], markerfacecolor='red', markeredgecolor='red', marker='o', markersize=5, alpha=1)
	xSol = 2.0
	ySol = 1.0
	strCoord = "("+format(xSol, ".2f")+", "+ format(ySol, ".2f")+", "+format(f(xSol, ySol), ".3f")+")"
	ax.text(2.0, 0.95, f(2.0, 1.0), "$\mathbf{x}_{sol}$"+strCoord, fontsize = fSize, color = "red")
	plt.show()

	return x, absErr, errorDataSteepestDescent


def ConjugateGradientsHS(M, b):
	avoidDivByZeroError = 0.0000000001
	errorDataConjugateGradients = []
	x = np.zeros_like(b, dtype=np.float)
	r = np.subtract(b, M.dot(x))
	d = np.subtract(b, M.dot(x))
	convergence = False

	while(not convergence):
		solutionError = np.subtract(M.dot(x), b)
		absErr = np.linalg.norm(solutionError)
		try:
			errorDataConjugateGradients.append(math.log(absErr))
		except:
			convergence = True

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

def ConjugateGradients_Golub(M, b):
	errorDataConjugateGradients = []
	tol = 0.000001
	k = 0
	x = np.zeros_like(b)
	r = np.subtract(b, A.dot(x))
	ro_c = r.dot(r)
	delta = tol * np.linalg.norm(b)
	while math.sqrt(ro_c) > delta:
		err = np.subtract(M.dot(x), b)
		absErr = np.linalg.norm(err)
		errorDataConjugateGradients.append(absErr)
		k = k + 1
		if(k == 1):
			p = r
		else:
			tau = ro_c / ro_minus
			p = np.add(r, np.multiply(p, tau))
		w = A.dot(p)
		miu_nominator = ro_c
		miu_denominator = w.dot(p)
		miu = miu_nominator / miu_denominator
		x = np.add(x, np.multiply(p, miu))
		r = np.subtract(r, np.multiply(w, miu))
		ro_minus = ro_c
		ro_c = r.dot(r)

	err = np.subtract(M.dot(x), b)
	absErr = np.linalg.norm(err)
	errorDataConjugateGradients.append(absErr)
	return x, absErr, errorDataConjugateGradients


def plotDiscretizedSine1D():
	N = 16
	h = 1.0 / N
	r = 1.0 / 100.0
	xCont = np.arange(0.0, 1.0 + r, r)
	xDiscr = np.arange(0.0, 1.0 + h, h)

	xCont2 = np.arange(0.0, 1.0 + r, r)
	xDiscr2 = np.arange(0.0, 1.0 + h, h)


	k = 5.0
	contFunction = np.sin(math.pi * k * xCont)
	discrFunction = np.sin(math.pi * k * xDiscr)

	contFunction2 = np.sin(math.pi * 8.0 * xCont2)
	discrFunction2 = np.sin(math.pi * 8.0 * xDiscr2)
	plt.subplot(211)
	plt.plot(xCont, contFunction)
	plt.plot(xCont, contFunction2)

	plt.subplot(212)
	plt.plot(xDiscr, discrFunction, linestyle='dashed') 
	plt.plot(xDiscr, discrFunction, 'ko')

	plt.plot(xDiscr, discrFunction2, linestyle='dashed') 
	plt.plot(xDiscr, discrFunction2, 'ko')
	plt.show()

def plotSineModes():
	N = 16
	h = 1.0 / N
	xDiscr = np.arange(0.0, 1.0 + h, h)

	discrFunction1 = np.sin(math.pi * 1.0  * xDiscr)
	discrFunction2 = np.sin(math.pi * 4.0 * xDiscr)
	discrFunction3 = np.sin(math.pi * 10.0 * xDiscr)
	discrFunction4 = np.sin(math.pi * 14.0 * xDiscr)
	discrFunction5 = np.sin(math.pi * 16.0 * xDiscr)

	plt.subplot(511)
	plt.title("$\sin{(\pi x)}$")
	plt.plot(xDiscr, discrFunction1, linestyle='dashed') 
	plt.plot(xDiscr, discrFunction1, 'ko')

	plt.subplot(512)
	plt.title("$\sin{(4 \pi x)}$")
	plt.plot(xDiscr, discrFunction2, linestyle='dashed') 
	plt.plot(xDiscr, discrFunction2, 'ko')

	plt.subplot(513)
	plt.title("$\sin{(10 \pi x)}$")
	plt.plot(xDiscr, discrFunction3, linestyle='dashed') 
	plt.plot(xDiscr, discrFunction3, 'ko')

	plt.subplot(514)
	plt.title("$\sin{(14 \pi x)}$")
	plt.plot(xDiscr, discrFunction4, linestyle='dashed') 
	plt.plot(xDiscr, discrFunction4, 'ko')

	plt.subplot(515)
	plt.title("$\sin{(31 \pi x)}$")
	plt.plot(xDiscr, discrFunction5, linestyle='dashed') 
	plt.plot(xDiscr, discrFunction5, 'ko')

	plt.show()


def plotProjectedSine():
	N1 = 16
	h1 = 1.0 / N1
	xDiscr1 = np.arange(0.0, 1.0 + h1, h1)

	N2 = 8
	h2 = 1.0 / N2
	xDiscr2 = np.arange(0.0, 1.0 + h2, h2)
	xDiscr22 = np.arange(0.0, 2.0 + h2, h2)

	discrFunction1 = np.sin(math.pi * 6.0 * xDiscr1)
	discrFunction2 = np.sin(math.pi * 6.0 * xDiscr2)
	discrFunction22 = np.sin(math.pi * 6.0 * xDiscr22)

	plt.subplot(311)
	plt.plot(xDiscr1, discrFunction1, linestyle='dashed') 
	plt.plot(xDiscr1, discrFunction1, 'ko')

	plt.subplot(312)
	plt.plot(xDiscr2, discrFunction2, linestyle='dashed') 
	plt.plot(xDiscr2, discrFunction2, 'ko')

	plt.subplot(313)
	plt.plot(xDiscr22, discrFunction22, linestyle='dashed') 
	plt.plot(xDiscr22, discrFunction22, 'ko')
	plt.show()

plotProjectedSine()
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
def SteepestDescent(M, b, iterationConstant = 100):
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
		print(x)
		err = np.subtract(M.dot(x), b)
		absErr = np.linalg.norm(err) / np.linalg.norm(b)
		errorDataSteepestDescent.append(math.log(absErr))
		alpha_numerator = r.dot(r)	
		alpha_denominator = r.dot(M.dot(r))
		if(alpha_denominator < avoidDivByZeroError):
			print('exit here divzero')
			break
		alpha = alpha_numerator / alpha_denominator
		xOld = np.copy(x)
		x = np.add(x, np.dot(r, alpha))
		# print(xOld)
		# print(x)
		# print('______')
		fSize = 26
		strCoord = "("+format(xOld[0], ".2f")+", "+ format(xOld[1], ".2f")+", "+format(f(xOld[0], xOld[1]), ".3f")+")"
		if(i>=0):
			ax.plot([xOld[0], x[0]], [xOld[1], x[1]], [f(xOld[0], xOld[1]), f(x[0], x[1])], color="k", linewidth = 3)
		if(i>= 0):
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
			# if(i == 4):
			# 	ax.plot([xOld[0]],[xOld[1]],[f(xOld[0], xOld[1])], markerfacecolor='white', markeredgecolor='white', marker='o', markersize=4, alpha=1)
			# 	ax.text(xOld[0],xOld[1],f(xOld[0], xOld[1]),"$\mathbf{x}_4$", fontsize = fSize, color = "white")
			# if(i == 5):
			# 	ax.plot([xOld[0]],[xOld[1]],[f(xOld[0], xOld[1])], markerfacecolor='white', markeredgecolor='white', marker='o', markersize=4, alpha=1)
			# 	ax.text(xOld[0],xOld[1],f(xOld[0], xOld[1]),"$\mathbf{x}_5$", fontsize = fSize, color = "white")
		r = np.subtract(b, M.dot(x))
		if(np.linalg.norm(r) < tol):
			break
	err = np.subtract(M.dot(x), b)
	Xs.append(x)
	absErr = np.linalg.norm(err) / np.linalg.norm(b)
	errorDataSteepestDescent.append(math.log(absErr))
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


def conjugate_grad(A, b, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    n = len(b)
    if not x:
        x = np.zeros(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in xrange(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            print 'Itr:', i
            break
        p = beta * p - r
	return x

plotF()
x, absErr, errData= SteepestDescent(A, b)
print(len(errData))

# # print(x)

# x = SteepestDescent(A, b)
# print(x)

print(Xs)
n = len(Xs)
# ax = plt.axes()
# for i in range(1,n):
# 	ax.plot([Xs[i-1][0], Xs[i][0]],[Xs[i-1][1], Xs[i][1]], color="k", linewidth = 3)
# plt.show()
print(n)


# Steepest Descent on sinx sin y model problem number of steps (start at N=2)
# 23, 37, 124, 460, 1640, 5710, 19560, 65030

# flops:
# (2, ':')
# 1858
# (4, ':')
# 11938
# (8, ':')
# 174098
# (16, ':')
# 2743858
# (32, ':')
# 40484818
# (64, ':')
# 574008898
# (128, ':')
# 7937609298
# (256, ':')
# 106048973058



# SSOR flops (starting an N = 8)

# (8, ':')
# ('flops: ', 32508)

# (16, ':')
# ('flops: ', 244398)

# (32, ':')
# ('flops: ', 1747308)

# (64, ':')
# ('flops: ', 12407892)

# (128, ':')
# ('flops: ', 85576050)


# SOR flops 

# (8, ':')
# 16440

# (16, ':')
# 129732

# (32, ':')
# 992154

# (64, ':')
# 7640730

# (128, ':')
# 59699844

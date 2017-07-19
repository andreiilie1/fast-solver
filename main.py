import numpy as np
from scipy.sparse import *
from scipy import *

#TODO: function JacobiIteration(D,R,b) to solve Mx=b

# We construct matrix M to approximate the solution of a differential equation
# We'll get the equation Mx = nablaValueVector and try to solve it by different methods
# The scope is to get a fast solver for a class of differential equations

def JacobiIterate(D,R,b,N):
	x = []
	d = D.diagonal()
	print("Diagonal: ",d)
	# Initial guess is x = (0,0,...0)
	for i in range(N):
		x.append(0)
	# print("Current guess: ",x)
	# Iterate for 50 times
	for i in range(1000):
		y = R.dot(x)
		r = np.subtract(b,y)
		x = [r_i / d_i for r_i, d_i in zip(r, d)]
		# print("Current guess: ",x)
	return x

# Value of the border function on values x,y
def borderFunction(x,y):
	value = 1
	return value

# Nabla value of the differential equation at points x, y
def nablaFunction(x,y):
	value = 0
	return value * h * h

# NablaValueVector from Mx = nablaValueVector
nablaValueVector = []

# Check if a(i,j) is on border
def isOnBorder(i, j):
	if(i == 0 or j == 0 or i == N - 1 or j == N - 1):
		return True
	else:
		return False

# Get the coordinates of the variable around which the row-th row is created
def getCoordinates(row):
	return int(row / N), row % N

# Get the row of a(i, j)'s equation
def getRow(i, j):
	return(i * N + j)

# Compute the elements of row-th row in (rowList, colList, dataList)
def computeRow(row):
	(x, y) = getCoordinates(row)
	if(isOnBorder(x, y)):
		rowList.append(row)
		colList.append(row)
		dataList.append(1)
		# The value of the border on point x/N, y/N is known, so append the equation variable = value to the system
		nablaValueVector.append(borderFunction(x / N, y / N))

		# Add entry to Diagonal component of the matrix
		rowListDiagonal.append(row)
		colListDiagonal.append(row)
		dataListDiagonal.append(1)
	else:
		value = - nablaFunction(x / N, y / N)
		rowList.append(row)
		colList.append(row)
		dataList.append(4)

		# Add entry to Diagonal component of the matrix
		rowListDiagonal.append(row)
		colListDiagonal.append(row)
		dataListDiagonal.append(4)

		for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
			if(not(isOnBorder(x + dX, y + dY))):
				rowList.append(row)
				colList.append(getRow(x + dX, y + dY))
				dataList.append(-1)

				rowListRemainder.append(row)
				colListRemainder.append(getRow(x + dX, y + dY))
				dataListRemainder.append(-1)
			else:
				localValue = borderFunction((x + dX) / N, (y + dY)/N)
				value += localValue
		nablaValueVector.append(value)


N = int(input("Enter inverse of sample rate\n"))
h = 1.0 / N

rowList = []
colList = []
dataList = []

rowListDiagonal = []
colListDiagonal = []
dataListDiagonal = []

rowListRemainder = []
colListRemainder = []
dataListRemainder = []

# Process the entries of each row to create matrices M, D, R
for currentRow in range(N * N):
	computeRow(currentRow)

# Instantiate sparse matrix M according to data kept in (rowList, colList, dataList)
M = csr_matrix((np.array(dataList), (np.array(rowList), np.array(colList))), shape = (N * N, N * N))
D = csr_matrix((np.array(dataListDiagonal), (np.array(rowListDiagonal), np.array(colListDiagonal))), shape = (N * N, N * N))
R = csr_matrix((np.array(dataListRemainder), (np.array(rowListRemainder), np.array(colListRemainder))), shape = (N * N, N * N))

solution = JacobiIterate(D, R, nablaValueVector, N * N)

# Next lines are for debugging purpose

print(solution)

# print(M.toarray())
# print(D.toarray())
# print(R.toarray())
#
# print(nablaValueVector)
# print(M.data())
# print(*rowList, sep=' ')
# print(*colList, sep=' ')

# The development plan is as follows:
# 1) Gauss-Seidel, 2) Jacobi, 3) Steepest Descent, 4) Gauss Elimination
# 5) Encapsulate the whole state in a class to which we can pass as paramters
#    the functions for the border values and for the nabla values so we are able
#    to instantiate it for different problems and test the efficiency easier

# Sample output for N = 4 (we're showing the matrix and the nabla vector)


# Enter inverse of sample rate
# 4
# [[ 1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  4 -1  0  0 -1  0  0  0  0  0  0]
#  [ 0  0  0  0  0 -1  4  0  0  0 -1  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0 -1  0  0  0  4 -1  0  0  0  0  0]
#  [ 0  0  0  0  0  0 -1  0  0 -1  4  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1]]
# 0.0625 0.0625 0.0625 0.0625 0.0625 0.125 0.125 0.0625 0.0625 0.125 0.125 0.0625 0.0625 0.0625 0.0625 0.0625

import numpy as np
from scipy.sparse import *
from scipy import *

# We construct matrix M to approximate the solution of a differential equation
# We'll get the equation Mx = nablaValueVector and try to solve it by different methods
# The scope is to get a fast solver for a class of differential equations

# Value of the border function on values x,y
def borderFunction(x,y):
	value = 1
	return value * h * h

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
	return row / N, row % N

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
	else:
		rowList.append(row)
		colList.append(row)
		dataList.append(4)
		for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
			if(not (isOnBorder(x + dX, y + dY))):
				rowList.append(row)
				colList.append(getRow(x + dX, y + dY))
				dataList.append(-1)


N = int(input("Enter inverse of sample rate\n"))
h = 1 / N

rowList = []
colList = []
dataList = []

# Process the entries of each row to create matrix M
for currentRow in range(N * N):
	computeRow(currentRow)

# Instantiate sparse matrix M according to data kept in (rowList, colList, dataList)
M = csr_matrix((np.array(dataList), (np.array(rowList), np.array(colList))), shape = (N * N, N * N))
print(M.toarray())

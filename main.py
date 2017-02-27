import numpy as np
from scipy.sparse import *
from scipy import *

# Current version assumes the value on the border is 1
# Curreny version assumes the nabla operator of f is 0 everywhere

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
	else:
		rowList.append(row)
		colList.append(row)
		dataList.append(4)
		for (dX, dY) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
			rowList.append(row)
			colList.append(getRow(x + dX, y + dY))
			dataList.append(-1)


N = input("Enter inverse of sample rate\n")

rowList = []
colList = []
dataList = []

# Process the entries of each row to create matrix M
for currentRow in range(N * N):
	computeRow(currentRow)

# Instantiate sparse matrix M according to data kept in (rowList, colList, dataList)
M = csr_matrix((np.array(dataList), (np.array(rowList), np.array(colList))), shape = (N * N, N * N))
print(M.toarray())

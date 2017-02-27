import numpy as np
from scipy.sparse import *
from scipy import *

# Current version assumes the value on the border is 1
# Curreny version assumes the nabla operator of f is 0 everywhere

def isOnBorder(i, j):
	if(i == 0 or j == 0 or i == N - 1 or j == N - 1):
		return True
	else:
		return False

def getCoordinates(row):
	return row / N, row % N

def computeRow(Matrix, row):
	(x, y) = getCoordinates(row)
	if(isOnBorder(x, y)):
		row.append(row)
		col.append(row)
		data.append(1)
	else:
		row.append(row)
		col.append(row)
		data.append(4)
		


N = input("Enter inverse of sample rate\n")

row = []
col = []
data = []
M = csr_matrix((np.array(data), (np.array(row), np.array(col))), shape = (3, 3))

for currentRow in range(N):
	computeRow(currentRow)
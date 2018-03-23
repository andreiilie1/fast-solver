import math


# Function examples for a simple equation discretizer

# Value of the border function on values x,y
def sinBorderFunction(x, y):
	# Assert (x,y) is on border
	value = 1.0 * math.sin(x) * math.sin(y)
	return value


# RHS value of the differential equation at points x, y
def sinValueFunction(x, y):
	value = - 2.0 * math.sin(x) * math.sin(y)
	return value


def borderFunction1(x, y):
	value = 1.0 * (x * x * x + y * y * y + x + y + 1.0)
	return value

def laplaceValueFunction1(x, y):
	value = 6.0 * x + 6.0 * y
	return value

def sin2BorderFunction(x, y):
	# Assert (x,y) is on border
	k = 2.0
	j = 5.0
	value = 1.0 * math.sin(math.pi * k * x) * math.sin(math.pi * j * y)
	return value


# RHS value of the differential equation at points x, y
def sin2ValueFunction(x, y):
	k = 2.0
	j = 5.0
	value = - 1.0 * math.sin(math.pi * k * x) * math.sin(math.pi * j * y) * math.pi * math.pi * (k * k + j * j)
	return value


# Function examples for a time equation discretizer (heat equation)

def heatSinBorderFunction(x, y, t):
	value = 1.0 * math.sin(x) * math.sin(y)
	return value

def heatRhsFunction(x, y, t):
	value = -2.0 * math.sin(x) * math.sin(y)
	return value

def heatInitialFunction(x, y, t):
	# Assert t == 0
	value = math.sin(x) * math.sin(y)
	return value



def sin1DValueFunction(x):
	value = - math.sin(x)
	return value

def sin1DBorderFunction(x):
	value = math.sin(x)
	return value


def sin1DValueFunction2(x):
	k = 3.0
	value = - math.sin(math.pi * k * x) * k * k * math.pi * math.pi
	return value

def sin1DBorderFunction2(x):
	k = 3.0
	value = math.sin(math.pi * k * x)
	return value


def zero1D(x):
	return 0
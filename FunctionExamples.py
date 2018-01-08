import math

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



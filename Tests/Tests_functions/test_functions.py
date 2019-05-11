# -*- coding: utf-8 -*-

from math import pi, cos
import numpy as np

class sinuses():
	def __init__(self, a, b):
		self.parameters = a
		self.coef = b
		self.min = self.calculate_function(b[0]/2, b[1]/2)
		self.min_x = b[0]/2
		self.min_y = b[1]/2
		s = a[0][1] + a[0][0]
		for i in range(1, len(a)):
			s += (i+1) * (a[i][0] + a[i][1]) * a[i][2]**i
		self.L = s
	def calculate_function(self, x, y):
		a = self.parameters
		b = self.coef
		z = -a[0][0] * np.sin(x * pi / b[0]) - a[0][1] * np.sin(y * pi / b[1])
		for i in range(1, len(a)):
			z +=  (-a[i][0] * np.sin(x * pi / b[0]) - a[i][1] * np.sin(y * pi / b[1]) + a[i][0] + a[i][1] + a[i][2])**(i+1)
		return z
	def lipschitz_function(self, Q):
		return self.L

	def lipschitz_gradient(self, Q):
		return self.L

	def der_x(self, x, y):
		a = self.parameters
		b = self.coef
		der = -a[0][0]
		for i in range(1, len(a)):
			der += (-(i+1) * a[i][0] * (-a[i][0] * np.sin(x * pi / b[0]) -
			a[i][1] * np.sin(y * pi / b[1]) + a[i][0] + a[i][1] + a[i][2])**i)
		return der * cos(x * pi / b[0]) * pi / b[0]
	
	def der_y(self, x, y):
		a = self.parameters
		b = self.coef
		der = -a[0][1]
		for i in range(1, len(a)):
			der += (-(i+1) * a[i][1] * (-a[i][0] * np.sin(x * pi / b[0]) -
			a[i][1] * np.sin(y * pi/b[1]) + a[i][0] + a[i][1] + a[i][2])**i)
		return der * cos(y * pi / b[1]) * pi / b[1]
	
	def gradient(self, x, y):
		return [self.der_x(x,y), self.der_y(x,y)]
	
	def get_est(self, x, num):
		return -1
	
#Quadratic function
class quadratic_function():
	def __init__(self, list_of_paremeters):
		self.A = list_of_paremeters
		p = self.A
		self.solution = np.linalg.solve(2 * np.array([[p[0]**2 + p[2], p[0] * p[1]],
												  [p[0] * p[1], p[1]**2]]),
							np.array([-p[3], -p[4]]))
		self.min = self.calculate_function(self.solution[0], self.solution[1])

	def lipschitz_function(self, Q):
		p = self.A
		L = 3 * max(abs(p[0]), abs(p[1]), p[2]) *  max(abs(Q[1]), abs(Q[3]))
		return L

	def lipschitz_gradient(self, Q):
		param = self.A
		M = (2 * param[0]**2 + 4 * abs(param[0] * param[1]) + 2 * param[1] ** 2 + 2 * param[2]) * 10
		return M

	def calculate_function(self, x, y):
		p = self.A
		f = (p[0] * x + p[1] * y)**2 + p[2] * x**2
		f += p[3] * x + p[4] * y + p[5]
		return f
	
	def der_x(self, x, y):
		p = self.A
		der = 2 * p[0] * (p[0] * x + p[1] * y) + 2 * p[2] * x + p[3]
		return der
	
	def der_y(self, x, y):
		p = self.A
		der = 2 * p[1] * (p[0] * x + p[1] * y) + p[4]
		return der

	def gradient(self, x, y):
		return np.array([self.der_x(x,y), self.der_y(x,y)])
	
	def get_sol_hor(self, segm, y):
		p = self.A
		if  (p[0]**2 + p[2]) != 0:
			x_0 = - (2 * p[0] * p[1] * y + p[3]) / (2 * (p[0]**2 + p[2]))
		else:
			x_0 = segm[0] - 1
		if segm[0] <= x_0 <= segm[1]:
			return x_0
		elif self.calculate_function(segm[0], y) < self.calculate_function(segm[1], y):
			return segm[0]
		else:
			return segm[1]

	def get_sol_vert(self, segm, x):
		p = self.A
		if  p[1] != 0:
			y_0 = - (2 * p[0] * p[1] * x + p[4]) / (2 * p[1]**2)
		else:
			y_0 = segm[0] - 1
		if segm[0] <= y_0 <= segm[1]:
			return y_0
		elif self.calculate_function(x, segm[0]) < self.calculate_function(x, segm[1]):
			return segm[0]
		else:
			return segm[1]

#Quadratic function
class LSM_exp():
	def __init__(self, a, b, n):
		self.x = np.random.uniform(-1, 1, n)
		self.y = a * np.exp(b * x) + np.random.normal(size = n)
		self.n = n
		self.min = 0

	def lipschitz_function(self, Q):
		return np.inf

	def lipschitz_gradient(self, Q):
		return np.inf

	def calculate_function(self, a, b):
		f = 1 / self.n * np.linalg.norm(a * np.exp(b * self.x) - self.y)**2
		return f
	
	def der_x(self, a, b):
		val = np.exp(b * self.x)
		der = 2 / self.n * val.dot(a * val - self.y)
		return der
	
	def der_y(self, x, y):
		val = np.exp(b * self.x)
		der = 2 / self.n * a * (x  * val).dot(a * val - self.y)
		return der

	def gradient(self, x, y):
		return np.array([self.der_x(x,y), self.der_y(x,y)])

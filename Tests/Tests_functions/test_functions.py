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
import scipy
from scipy import optimize
class quadratic_function():
	def __init__(self, list_of_parameters):
		self.A = list_of_parameters
		p = self.A
		self.solution = np.linalg.solve(2 * np.array([[p[0]**2 + p[2], p[0] * p[1]],
												  [p[0] * p[1], p[1]**2]]),
							np.array([-p[3], -p[4]]))
		self.min = self.calculate_function(self.solution[0], self.solution[1])
		self.L = None
		self.M = None
	def lipschitz_function(self, Q):
		if self.L is None:
			self.L = -scipy.optimize.minimize(lambda x: -np.linalg.norm(self.gradient(x[0], x[1])), np.array([Q[0], Q[2]]), bounds = [(Q[0], Q[1]), (Q[2], Q[3])])['fun']
		return self.L

	def lipschitz_gradient(self, Q):
		if self.M is None:
			p = self.A
			A = [[p[0]**2+p[2]**2, p[0]*p[1]], [p[0]*p[1], p[1]**2]]
			A = 2 * np.array(A)
			w, _ = np.linalg.eig(A)
			self.M = w.max()
		return self.M

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


# LOG-SUM-EXP
import scipy
from scipy import optimize

class LogSumExp():
	def __init__(self, list_of_parameters, R1 = 1, R2 = 2):
		self.a = list_of_parameters
		a = self.a
		# self.f = lambda x: np.log(1 + sum([np.exp(i*x[ind]) for ind, i in enumerate(a)]))
		self.f = lambda x: np.linalg.norm(x)
		self.g1 = lambda x: np.linalg.norm(x)**2 - self.R2**2
		self.g2 = lambda x: -np.linalg.norm(x)**2 + self.R1**2
		self.phi = lambda l1, l2: lambda x: self.f(x) + l1 * self.g1(x) + l2 * self.g2(x)
		self.R1 = R1
		self.R2 = R2
		self.L = None
		self.M = None
		self.x_cur = None

	def lipschitz_function(self, Q):
		if self.L is None:
			L = scipy.optimize.minimize(lambda x: -np.linalg.norm(self.gradient(x[0], x[1])), 
				np.array([Q[0], Q[2]]),
				bounds = [(Q[0], Q[1]),
				(Q[2], Q[3])])
			self.L = -L['fun']
			print(self.L)
		return self.L

	def lipschitz_gradient(self, Q):
		if self.M is None:
			self.M = 10000
			# There should be code
			print('There is not code')
		return self.M

	def calculate_function(self, l1, l2):
		a = self.a
		phi = self.phi
		x_cur = scipy.optimize.minimize(phi(l1, l2), 
			np.zeros(self.a.shape)).x
		#M = (a*x_cur).max()
		#x = x_cur - M*np.ones(x_cur.shape)
		return phi(l1, l2)(x_cur)
	
	def der_x(self, l1, l2):
		phi = self.phi
		x_cur = scipy.optimize.minimize(phi(l1, l2), 
			np.zeros(self.a.shape)).x
		return -self.g1(x_cur)
	
	def der_y(self, l1, l2):
		phi = self.phi
		x_cur = scipy.optimize.minimize(phi(l1, l2), 
			np.zeros(self.a.shape)).x
		return -self.g2(x_cur)

	def gradient(self, x, y):
		return np.array([self.der_x(x,y), self.der_y(x,y)])

	def sol_prime(self, l1 = None, l2 = None):
		phi = self.phi
		x_cur = scipy.optimize.minimize(phi(l1, l2), 
			np.zeros(self.a.shape)).x
		return f(x_cur)

	def get_square(self):
		x = np.zeros(len(self.a))
		x[0] = self.R1 + (self.R2-self.R1)/10
		gamma = min(-self.g1(x), -self.g2(x))
		q = self.f(x) / gamma
		self.Q = [0, q, 0, q]
		return self.Q

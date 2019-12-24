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

# LOG-SUM-EXP
import time
class LogSumExp():
	def __init__(self, list_of_parameters, c = None, R1 = 2, R2 = 3, C = 1):
		self.a = list_of_parameters
		a = self.a
		n = a.shape[0]
		self.C = C
		#v = np.random.uniform(1,10, (1,))[0]
		v = 1
		self.v = v
		self.f = lambda x: np.log(1 + np.exp(v*x).sum()) + np.linalg.norm(x)**2*self.C
		#self.f = lambda x: np.log(1 + sum([np.exp(i*x[ind]) for ind, i in enumerate(a)]))
		# self.f = lambda x: np.linalg.norm(x)
		if c is None:
			c = np.ones(self.a.shape)
			c/=np.linalg.norm(c)
		n = c.shape[0]
		b1 = np.random.uniform(-1,1,(n,))
		self.b1 = b1
		self.g1mu, self.g1L, self.g1M = 0, np.linalg.norm(b1), 0
		self.g1 = lambda x: - self.R1**2 + b1.dot(x)

		b2 = np.random.uniform(-1,1,(n,))
		self.b2 = b2
		self.g2mu, self.g2L, self.g2M = 0, np.linalg.norm(b2), 0
		self.g2 = lambda x: - self.R2**2 + b2.dot(x)
		B = np.vstack((b1, b2))
		l = np.linalg.eig(B.dot(B.T))[0]
		self.lmin, self.lmax = l.min(), l.max()
		self.c = c
		self.phi = lambda l1, l2: lambda x: -(self.f(x) + l1 * self.g1(x) + l2 * self.g2(x))
		self.R1 = R1
		self.R2 = R2
		self.L = None
		self.M = None
		self.x_cur = None
		self.values = dict()
		self.Q = self.get_square()
		self.R0 = self.get_R0()
		print('R0', self.R0)
		self.fL = None
		self.fmu  = 2*self.C
		self.get_lipschitz_constants()
		cond_num = (4*self.fL/self.fmu * self.lmax/self.lmin)
		print('cond_num', cond_num)
		print('cond_num_sqrt', np.sqrt(cond_num))

	def get_R0(self):
		l_max = self.Q[1]
		n = self.a.shape[0]
		R0 = np.sqrt(2*np.log(n+1))/self.C
		return R0

	def get_lipschitz_constants(self):
		self.fL = self.v + 2 * self.C
		print('L_f',self.fL)

	def f_der(self, x):
		m = lambda x: (self.v*x).max()
		grad = lambda x: np.exp(self.v*x-m(x)) / (1/np.exp(m(x))+np.exp(self.v*x-m(x)).sum())
		return grad(x)

	def g1_der(self, x):
		g = np.zeros(x.shape)
		g = self.b1
		return g

	def g2_der(self, x):
		g = np.zeros(x.shape)
		g = self.b2
		return g

	def lipschitz_function(self, Q):
		if self.L is None:
			norm = lambda x: np.linalg.norm(x)
			self.L = (norm(self.b1) + norm(self.b2)) * self.R0
		return self.L

	def lipschitz_gradient(self, Q):
		if self.M is None:
			self.M = (self.g1L + self.g2L)**2 / self.C
		return self.M

	def calculate_function(self, l1, l2):
		a = self.a
		phi = self.phi
		if (l1,l2) in self.values:
			x_cur = self.values[(l1,l2)]
		else:
			s = time.time()
			x_cur = scipy.optimize.minimize(lambda x:-phi(l1, l2)(x), 
				np.zeros(self.a.shape), method = 'CG').x
			self.values[(l1,l2)] = x_cur
		return phi(l1, l2)(x_cur)

	def GD(self, l1, l2, L1, der, warm = None):
		f = lambda x: self.phi(l1,l2)(x)
		grad = lambda x: self.f_der(x) + l1*self.g1_der(x) + l2*self.g2_der(x)
		L = self.fL +l1*self.g1M + l2*self.g2M
		mu = (self.fmu+ l1*self.g1mu+l2*self.g2mu)
		M = L/mu
		q = (M-1)/(M+1)
		R = self.R0
		alpha = 1/(L+mu)
		x = np.zeros(self.a.shape)
		if not warm is None:
			x, R = warm
		#R *= L/2
		x, x_prev= x - 1/L * grad(x), x
		#R *= 1/5
		N = 1
		while abs(der(x)) < R:
			#R *= min(q, (N+4)/(N+5))
			R *= q
			x = x - alpha *grad(x)
			N += 1
		return x

	def der_x(self, l1, l2, warm = None):
		x_cur = self.GD(l1, l2, self.g1L, self.g1, warm)
		return -self.g1(x_cur)
	
	def der_y(self, l1, l2, warm = None):
		x_cur = self.GD(l1, l2, self.g2L, self.g2, warm)
		return -self.g2(x_cur)

	def gradient(self, l1, l2):
		phi = self.phi
		if (l1,l2) in self.values:
			x_cur = self.values[(l1,l2)]
		else:
			x_cur = scipy.optimize.minimize(lambda x:-phi(l1, l2)(x), 
				np.zeros(self.a.shape), method = 'CG').x
			self.values[(l1,l2)] = x_cur
		return np.array([-self.g1(x_cur), -self.g2(x_cur)])

	def sol_prime(self, l1 = None, l2 = None):
		phi = self.phi
		if (l1,l2) in self.values:
			x_cur = self.values[(l1,l2)]
		else:
			x_cur = scipy.optimize.minimize(lambda x:-phi(l1, l2)(x), 
				np.zeros(self.a.shape)).x
			self.values[(l1,l2)] = x_cur
		return f(x_cur)

	def get_square(self):
		x = np.zeros((self.a.shape))
		gamma = min(-self.g1(x), -self.g2(x))
		q = self.f(x) / gamma
		self.Q = [0, q, 0, q]
		return self.Q


# -*- coding: utf-8 -*-

import math
import numpy as np

class solver_segment:
	def __init__(self, f, Q, eps):
		self.f = f
		self.Q = Q.copy()
		self.size = Q[1] - Q[0]
		self.eps = eps
		self.solve = self.gss
		self.type_stop = 'true_grad'
		self.value = 0
		self.axis = 'x'
		self.segm = [Q[0], Q[1]]
		self.est = None
		self.f_L = self.f.lipschitz_function(self.Q)
		self.f_M = self.f.lipschitz_gradient(self.Q)


	def init_help_function(self, stop_func = 'true_grad', solve_segm = 'gss'):
		self.type_stop = stop_func
		if solve_segm == 'gss':
			self.solve = self.gss
		if solve_segm == 'grad_desc':
			self.solve = self.grad_descent_segment

	def TG(self, a, b):
		if self.est is None:
			f = self.f
			arg = self.value
			if self.axis == 'x':
				self.est = abs(f.der_y(f.get_sol_hor(self.segm, arg), arg)) / self.f_M
			if self.axis == 'y':
				self.est = abs(f.der_x(arg, f.get_sol_vert(self.segm, arg))) / self.f_M
		cond = ((b - a) / 2 <= self.est)
		if cond:
			self.est = None
		return cond

	def CE(self, a, b):
		if self.est is None:
			M, R, L, eps = self.f_M, self.size, self.f_L, self.eps
			self.est = eps / (2 * M * R * math.sqrt(5) * (math.log((2 * L * R * math.sqrt(2)) / eps, 2)))
		return ((b - a) / 2 <= self.est)

	def CG_CE(self, a, b):
		is_inter = (a != self.segm[0] and b != self.segm[1])
		if is_inter:
			if self.axis == 'y':
				grad = np.linalg.norm(self.f.gradient(self.value, (b+a) / 2))
			if self.axis == 'x':
				grad = np.linalg.norm(self.f.gradient((b+a) / 2, self.value))
			cond_TG = ((b - a) / 2 <= grad/self.f_M)
		else:
			cond_TG = False
		if self.est is None:
			M, R, L, eps = self.f_M, self.size, self.f_L, self.eps
			self.est = eps / (2 * M * R * math.sqrt(5) * (math.log((2 * L * R * math.sqrt(2)) / eps, 2)))
		cond_CE = ((b - a) / 2 <= self.est)
		return cond_CE or cond_TG

	def always_true(self, a, b):
		return True

	def stop(self):
		if self.type_stop == 'true_grad':
			return self.TG
		if self.type_stop == 'const_est':
			return self.CE
		if self.type_stop == 'cur_grad':
			return self.CG_CE
		if self.type_stop == 'true':
			return self.always_true

	def grad_descent_segment(self):
		segm = self.segm
		if self.axis == 'x':
			deriv = lambda x: self.f.der_x(x, self.value)
			x_opt = self.f.get_sol_hor(self.segm, self.value)
		else:
			deriv = lambda y: self.f.der_y(self.value, y)
			x_opt = self.f.get_sol_vert(self.segm, self.value)
		N = 0
		x, alpha_0 = (segm[0] + segm[1]) / 2, (segm[0] + segm[1]) / 4
		mystop = stop()
		while not mystop(x, x_opt) and N < 200:
			x = x - alpha_0 / math.sqrt(N + 1) * deriv(x)
			x = min(max(x, segm[0]), segm[1])
			N += 1
		if N >= 200:
			N = -1
		return x
	
	def gss(self):
		if self.axis == 'x':
			f = lambda x: self.f.calculate_function(x, self.value)
		if self.axis == 'y':
			f = lambda y: self.f.calculate_function(self.value, y)

		a, b = self.segm
		gr = (math.sqrt(5) + 1) / 2
		c = b - (b - a) / gr
		d = a + (b - a) / gr 
		f_c, f_d = f(c), f(d)
		N = 0
		mystop = self.stop()
		while not mystop(a, b):
			if f(c) < f(d):
				b = d
				d, f_d = c, f_c
				c = b - (b- a) / gr
				f_c = f(c)
			else:
				a = c
				c, f_c = d, f_d
				d = a + (b - a) / gr
				f_d = f(d)
			N+=1
			if N >= 200:
				return (b+a)/2
		return (b + a) / 2

class main_solver(solver_segment):
	def add_cond(self, x, y):
		return False
		eps = self.eps
		if np.linalg.norm(self.f.gradient(x, y)) <= eps / (self.size * math.sqrt(2)):
			return True
		return False

	def halving_square(self):
		eps = self.eps
		Q = self.Q.copy()
		N = 0
		minimum = self.f.min
		x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
		if self.add_cond(x_0, y_0):
			return ((x_0, y_0), N)
		f_opt = self.f.calculate_function(x_0, y_0)
		while True:
			y_0 = (Q[2] + Q[3]) / 2
			self.axis, self.value, self.segm = 'x', y_0, [Q[0], Q[1]]
			x_0 = self.solve()
			if self.add_cond(x_0, y_0):
				return ((x_0, y_0), N)
			der = self.f.der_y(x_0, y_0)
			if der == 0 and self.f.der_x(x_0, y_0) == 0:
				return ((x_0, (Q[2] + Q[3]) / 2), N)
			if der > 0:
				Q[2], Q[3] = Q[2],  y_0
			else:
				Q[3], Q[2] = Q[3],  y_0
			
			x_0 = (Q[0] + Q[1]) / 2
			self.axis, self.value, self.segm = 'y', x_0, [Q[2], Q[3]]
			y_0 = self.solve()
			if self.add_cond(x_0, y_0):
				return ((x_0, y_0), N)
			der = self.f.der_x(x_0, y_0)
			if der == 0 and self.f.der_y(x_0, y_0) == 0:
				return ((x_0, (Q[2] + Q[3]) / 2), N)
			if der > 0:
				Q[0], Q[1] = Q[0],  x_0
			else:
				Q[1], Q[0] = Q[1],  x_0

			N += 1
			
			x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
			f_opt = self.f.calculate_function(x_0, y_0) 
			if N >= 100 or abs(f_opt - minimum) < eps:
				if N >= 100:
					N = -1
				return ((x_0, y_0), N)

def gradient_descent(f, Q, grad, eps, step, minimum):
	N = 0
	x = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
	x_prev = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
	while (abs(f(x[0], x[1]) - f(x_prev[0], x_prev[1])) > eps and N < 1000) or (N == 0):
		der = grad(x[0], x[1])
		x[0], x_prev[0] = min(max(x[0] - step / math.sqrt(N+1) * der[0], Q[0]), Q[1]), x[0]
		x[1], x_prev[1] = min(max(x[1] - step / math.sqrt(N+1) * der[1], Q[2]), Q[3]), x[1]
		N += 1
	if N >= 1000:
		N = -1
	return (x, N, abs(f(x[0], x[1]) - f(x_prev[0], x_prev[1])))

def ellipsoid(f, Q, x_0=None, eps=None):
	n = 2
	x = np.array([(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]) if x_0 is None else x_0
	eps = 5e-10 if eps is None else eps
	rho = (Q[1] - Q[0]) / 2
	H = np.identity(n)
	q = n * (n - 1) ** (-(n-1) / (2*n)) * (n + 1) ** (-(n+1) / (2*n))
	k = 0
	while abs(f.calculate_function(x[0], x[1]) - f.min) > eps and k < 10000:
		gamma = (rho / (n+1)) * (n / np.sqrt(n ** 2 - 1)) ** k
		_df = f.gradient(x[0], x[1])
		x = x - gamma * H @ _df
		H -= (2 / (n + 1)) * ((H @ _df @ _df.T * H) / ((H @ _df).T @ _df))
		k += 1
	if k >= 10000:
		k = -1
	return (x, k)

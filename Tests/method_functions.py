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


	def init_help_function(self, stop_func = 'cur_grad', solve_segm = 'gss'):
		self.type_stop = stop_func
		if solve_segm == 'gss':
			self.solve = self.gss
		if solve_segm == 'grad_desc':
			self.solve = self.grad_descent_segment

	def TrueGrad(self, a, b):
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

	def ConstEst(self, a, b):
		if self.est is None:
			M, R, L, eps = self.f_M, self.size, self.f_L, self.eps
			self.est = eps / (2 * M * R * math.sqrt(5) * (math.log((2 * L * R * math.sqrt(2)) / eps, 2)))
		return ((b - a) / 2 <= self.est)

	def CurGrad(self, a, b):
		if self.axis == 'y':
			grad = abs(self.f.der_x(self.value, (b+a) / 2))
		if self.axis == 'x':
			grad = abs(self.f.der_y((b+a) / 2, self.value))
		cond_TG = ((b - a) / 2 <= grad/self.f_M)
		return cond_TG

	def AlwaysTrue(self, a, b):
		return True

	def stop(self):
		if self.type_stop == 'true_grad':
			return self.TrueGrad
		if self.type_stop == 'const_est':
			return self.ConstEst
		if self.type_stop == 'cur_grad':
			return self.CurGrad
		if self.type_stop == 'true':
			return self.AlwaysTrue

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
		results = [(x_0, y_0)]
		if self.add_cond(x_0, y_0):
			return ((x_0, y_0), N, results)
		f_opt = self.f.calculate_function(x_0, y_0)
		while True:
			y_0 = (Q[2] + Q[3]) / 2
			self.axis, self.value, self.segm = 'x', y_0, [Q[0], Q[1]]
			x_0 = self.solve()
			if self.add_cond(x_0, y_0):
				return ((x_0, y_0), N, results)
			der = self.f.der_y(x_0, y_0)
			if der == 0 and self.f.der_x(x_0, y_0) == 0:
				return ((x_0, (Q[2] + Q[3]) / 2), N, results)
			if der > 0:
				Q[2], Q[3] = Q[2],  y_0
			else:
				Q[3], Q[2] = Q[3],  y_0
			
			x_0 = (Q[0] + Q[1]) / 2
			self.axis, self.value, self.segm = 'y', x_0, [Q[2], Q[3]]
			y_0 = self.solve()
			if self.add_cond(x_0, y_0):
				return ((x_0, y_0), N, results)
			der = self.f.der_x(x_0, y_0)
			if der == 0 and self.f.der_y(x_0, y_0) == 0:
				return ((x_0, (Q[2] + Q[3]) / 2), N, results)
			if der > 0:
				Q[0], Q[1] = Q[0],  x_0
			else:
				Q[1], Q[0] = Q[1],  x_0

			N += 1
			x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
			results.append((x_0, y_0))
			f_opt = self.f.calculate_function(x_0, y_0) 
			if N >= 100 or abs(f_opt - minimum) < eps:
				if N >= 100:
					N = -1
				return ((x_0, y_0), N,results)

def gradient_descent(f, Q, grad, L, eps, minimum):
	N = 0
	x = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
	results = [x.copy()]
	x_prev = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
	while (abs(f(x[0], x[1]) - minimum) > eps and N < 100) or (N == 0):
		der = grad(x[0], x[1])
		x[0], x_prev[0] = min(max(x[0] - 1. / L * der[0], Q[0]), Q[1]), x[0]
		x[1], x_prev[1] = min(max(x[1] - 1. / L * der[1], Q[2]), Q[3]), x[1]
		N += 1
		results.append(x.copy())
	if N >= 100:
		N = -1
	return (x, N, results)

def ellipsoid(f, Q, x_0=None, eps=None):
	n = 2
	x = np.array([(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]) if x_0 is None else x_0
	eps = 5e-3 if eps is None else eps
	rho = (Q[1] - Q[0]) * np.sqrt(2) / 2
	H = np.identity(n)
	q = n * (n - 1) ** (-(n-1) / (2*n)) * (n + 1) ** (-(n+1) / (2*n))
	domain = np.array([[Q[0], Q[1]], [Q[2], Q[3]]])
	k = 0
	results = [x]
	while abs(f.calculate_function(x[0], x[1]) - f.min) > eps and k < 100:
		gamma = (rho / (n+1)) * (n / np.sqrt(n ** 2 - 1)) ** k
		d = (n / np.sqrt(n ** 2 - 1)) ** k
		_df = f.gradient(x[0], x[1])
		_df = _df / (np.sqrt(abs(_df@H@_df)))
		x = np.clip(x - gamma * H @ _df, *domain.T)
		H = H - (2 / (n + 1)) * (H @ np.outer(_df, _df) @ H)
		k += 1
		results.append(x)
	if k >= 100:
		k = -1
	return (x, k, results)

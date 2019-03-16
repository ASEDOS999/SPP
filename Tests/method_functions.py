# -*- coding: utf-8 -*-

import math
import numpy as np

class solver_segment:
	def __init__(self, f, Q, eps):
		self.f = f
		self.Q = Q.copy()
		self.size = Q[1] - Q[0]
		self.eps = eps
		self.stop = self.const_est
		self.solve = self.gss
		self.value = 0
		self.axis = 'x'
		self.segm = []

	def init_help_function(self, stop_func = 'const_est', solve_segm = 'gss'):
		if stop_func == 'big_grad':
			self.stop = self.big_grad
		if stop_func == 'stop_ineq':
			self.stop = self.stop_ineq
		if stop_func == 'true':
			self.stop = self.always_true
		if stop_func == 'const_est':
			self.stop = self.const_est
		
		if solve_segm == 'gss':
			self.solve = self.gss
		if solve_segm == 'grad_desc':
			self.solve = self.grad_descent_segment

	def big_grad(self, a, b):
		arg = self.value
		f = self.f
		L = f.lipschitz_gradient(self.Q)
		if self.axis == 'x':
			return b - a <= abs(f.der_y(f.get_sol_hor(self.segm, arg), arg)) / L
		if self.axis == 'y':
			return b - a <= abs(f.der_x(arg, f.get_sol_vert(self.segm, arg))) / L
	
	def stop_ineq(self, a, b):
		if self.axis == 'x':
			sol = self.f.get_sol_hor(self.segm, self.value)
			val_grad = np.linalg.norm(self.f.gradient(sol, self.value))
		if self.axis == 'y':
			sol = self.f.get_sol_vert(self.segm, self.value)
			val_grad = np.linalg.norm(self.f.gradient(self.value, sol))
		is_inter = sol != self.segm[0] and sol != self.segm[1]
		if is_inter:
			L = self.f.lipshitz_gradient(self.Q)
			return b - a <= grad / L
		return big_grad(a, b)
	
	def always_true(self, a, b):
		return True
	
	def const_est(self, a, b):
		L = self.f.lipschitz_function(self.Q)
		M = self.f.lipschitz_gradient(self.Q)
		R = self.size
		eps = self.eps
		est = eps / (2 * M * R * (math.sqrt(2) + math.sqrt(5)) * ( - math.log(eps / (L * R * math.sqrt(2)), 2)))
		return b - a <= 0.0001 * est

	def grad_descent_segment(self):
		segm = self.segm
		if self.axis == 'x':
			deriv = self.f.der_x
			x_opt = f.get_sol_hor(self.segm)
		else:
			deriv = self.f.der_y
			x_opt = f.get_sol_vert(self.segm)
		N = 0
		x, alpha_0 = (segm[0] + segm[1]) / 2, (segm[0] + segm[1]) / 4
		if delta == 0:
			delta = 0.01 * alpha_0
		if delta < 0:
			return x
		while not self.stop(x, x_opt) and N < 1000:
			x = x - alpha_0 / math.sqrt(N + 1) * deriv(x)
			x = min(max(x, segm[0]), segm[1])
			N += 1
		if N >= 1000:
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
		while not self.stop(a, b):
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
		return (b + a) / 2

class main_solver(solver_segment):
	def add_cond(self, x, y):
		eps = self.eps
		if np.linalg.norm(self.f.gradient(x, y)) <= eps / (self.size * math.sqrt(2)):
			return True
	
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
				return (x_0, y_0)
			der = self.f.der_y(x_0, y_0)
			if der == 0 and self.f.der_x(x_0, y_0) == 0:
				return ((x_0, (Q[2] + Q[3]) / 2), N)
			if der > 0:
				Q[2], Q[3] = Q[2],  y_0
			else:
				Q[3], Q[2] = Q[3],  y_0
			
			x_0 = (Q[2] + Q[3]) / 2
			self.axis, self.value, self.segm = 'y', x_0, [Q[2], Q[3]]
			y_0 = self.solve()
			if self.add_cond(x_0, y_0):
				return (x_0, y_0)
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
				if N >= 1000:
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

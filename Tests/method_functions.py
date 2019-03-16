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
		self.type_stop = 'big_grad'
		self.value = 0
		self.axis = 'x'
		self.segm = [Q[0], Q[1]]
		self.est = 0
		self.f_L = self.f.lipschitz_function(self.Q)
		self.f_M = self.f.lipschitz_gradient(self.Q)


	def init_help_function(self, stop_func = 'const_est', solve_segm = 'gss'):
		self.type_stop = stop_func
		
		if solve_segm == 'gss':
			self.solve = self.gss
		if solve_segm == 'grad_desc':
			self.solve = self.grad_descent_segment

	def get_est(self):
		L, M = self.f_L, self.f_M
		R = self.size
		eps = self.eps
		arg = self.value
		f = self.f
		if self.type_stop == 'big_grad':
			if self.axis == 'x':
				self.est = abs(f.der_y(f.get_sol_hor(self.segm, arg), arg)) / M
			if self.axis == 'y':
				self.est = abs(f.der_x(arg, f.get_sol_vert(self.segm, arg))) / M
		if self.type_stop == 'const_est':
			self.type_stop = 'const_est_est'
			self.est = eps / (2 * M * R * (math.sqrt(2) + math.sqrt(5)) * ( - math.log(eps / (L * R * math.sqrt(2)), 2)))
		if self.type_stop == 'stop_ineq':
			if self.axis == 'x':
				sol = self.f.get_sol_hor(self.segm, self.value)
				is_inter = sol != self.segm[0] and sol != self.segm[1]
				if is_inter:
					self.est = -1
				else:
					self.est = abs(f.der_y(sol, arg)) / M
			if self.axis == 'y':
				sol = self.f.get_sol_vert(self.segm, self.value)
				is_inter = sol != self.segm[0] and sol != self.segm[1]
				if is_inter:
					self.est = -1
				else:
					self.est = abs(f.der_y(arg, sol)) / M
		if self.type_stop == 'true':
			self.est = self.size

	def stop(self, a, b):
		if self.type_stop == 'true':
			return True

		if self.type_stop == 'stop_ineq':
			if self.est == -1:
				if self.axis == 'y':
					grad = np.linalg.norm(self.f.gradient(self.value, (b+a) / 2))
				if self.axis == 'x':
					grad = np.linalg.norm(self.f.gradient((b+a) / 2, self.value))
				return b - a <= grad/self.f_M
			else:
				return b - a <= self.est

		if self.type_stop == 'const_est_est' or self.type_stop == 'big_grad':
			return b - a <= self.est

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
		self.get_est()
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
		self.get_est()
		while not self.stop(a, b):
		#while b - a >= self.est:
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
		self.get_est()
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

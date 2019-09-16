# -*- coding: utf-8 -*-

import math
import numpy as np
import time
def get_cond(**kwargs):
	f = kwargs['f']
	if kwargs.__contains__('time') and kwargs['time']:
		def _(list_time, T = kwargs['time_max'], **kwargs):
			list_time.append(time.time())
			return (list_time[-1]-list_time[0] > T)
		args = (_, [time.time()])
		return args
	if kwargs.__contains__('eps'):
		def _(list_time, **kwargs):
			x, N = kwargs['x'], kwargs['N']
			list_time.append(time.time())
			if kwargs.__contains__('minimum') and not kwargs['minimum'] is None:
				return not ((abs(f(x[0], x[1]) - kwargs['minimum']) > kwargs['eps'] and N < 100) or N == 0)
			else:
				return kwargs['size_Q'] < kwargs['eps']
		args = (_, [time.time()])
		return args

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
		self.f_L = self.f.L
		self.f_M = self.f.M


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
		if eps is None:
			return False
		if np.linalg.norm(self.f.gradient(x, y)) <= eps / (self.size * math.sqrt(2)):
			return True
		return False

	def halving_square(self, **kwargs):
		if kwargs.__contains__('eps'):
			self.eps = kwargs['eps']
			del kwargs['eps']
		eps = self.eps
		m = self.f.min if hasattr(self.f, 'min') else None
		cond, args = get_cond(f = self.f.calculate_function, eps = eps, minimum = m, **kwargs)
		Q = self.Q.copy()
		N = 0
		x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
		results = [(x_0, y_0)]
		if self.add_cond(x_0, y_0):
			return ((x_0, y_0), N, results, args)
		f_opt = self.f.calculate_function(x_0, y_0)
		while True:
			y_0 = (Q[2] + Q[3]) / 2
			self.axis, self.value, self.segm = 'x', y_0, [Q[0], Q[1]]
			x_0 = self.solve()
			if self.add_cond(x_0, y_0):
				return ((x_0, y_0), N, results,args)
			der = self.f.der_y(x_0, y_0)
			if der > 0:
				Q[2], Q[3] = Q[2],  y_0
			else:
				Q[3], Q[2] = Q[3],  y_0
			
			x_0 = (Q[0] + Q[1]) / 2
			self.axis, self.value, self.segm = 'y', x_0, [Q[2], Q[3]]
			y_0 = self.solve()
			if self.add_cond(x_0, y_0):
				return ((x_0, y_0), N, results, args)
			der = self.f.der_x(x_0, y_0)
			if der > 0:
				Q[0], Q[1] = Q[0],  x_0
			else:
				Q[1], Q[0] = Q[1],  x_0

			N += 1
			x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
			results.append((x_0, y_0))
			f_opt = self.f.calculate_function(x_0, y_0) 
			if cond(args, x = (x_0, y_0), N = N, minimum = m, eps = eps, size_Q = Q[1]-Q[0]):
				if N >= 100:
					N = -1
				return (x_0, y_0), N,results, args

def gradient_descent(f, Q, L, **kwargs):
	N = 0
	cond, args = get_cond(f=f, **kwargs)
	x = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
	results = [x.copy()]
	grad = lambda x: f.gradient(x[0], x[1])
	x_prev = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
	while True:
		der = grad(x)
		x[0], x_prev[0] = min(max(x[0] - 1. / L * der[0], Q[0]), Q[1]), x[0]
		x[1], x_prev[1] = min(max(x[1] - 1. / L * der[1], Q[2]), Q[3]), x[1]
		N += 1
		results.append(x.copy())
		if cond(args, x=x, N=N, size_Q = Q[1]-Q[0]):
			if kwargs.__contains__('res'):
				kwargs['res'][0][kwargs['res'][1]] = (x, N, results, args, f.values)
			return (x, N, results, args)

def ellipsoid(f, Q, eps = None, **kwargs):
	n = 2
	cond, args = get_cond(f=f, **kwargs)
	x = np.array([(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]) if not kwargs.__contains__('x_0') else kwargs['x_0']
	eps = 5e-3 if eps is None else eps
	rho = (Q[1] - Q[0]) * np.sqrt(2) / 2
	H = np.identity(n)
	q = n * (n - 1) ** (-(n-1) / (2*n)) * (n + 1) ** (-(n+1) / (2*n))
	domain = np.array([[Q[0], Q[1]], [Q[2], Q[3]]])
	k = 0
	results = [x]
	while True:
		gamma = (rho / (n+1)) * (n / np.sqrt(n ** 2 - 1)) ** k
		d = (n / np.sqrt(n ** 2 - 1)) ** k
		_df = f.gradient(x[0], x[1])
		_df = _df / (np.sqrt(abs(_df@H@_df)))
		x = np.clip(x - gamma * H @ _df, *domain.T)
		H = H - (2 / (n + 1)) * (H @ np.outer(_df, _df) @ H)
		k += 1
		results.append(x)
		if cond(args, x=x, N=k, size_Q = Q[1]-Q[0]):
			if kwargs.__contains__('res'):
				kwargs['res'][0][kwargs['res'][1]] = (x, k, results, args, f.values)
			return (x,k,results,args)



class halving_square:
	def __init__(self, f, Q, eps):
		self.f = f
		self.Q = Q.copy()
		self.size = Q[1] - Q[0]
		self.eps = 0.5
		self.solve = self.gss
		self.type_stop = 'true_grad'
		self.value = 0
		self.axis = 'x'
		self.segm = [Q[0], Q[1]]
		self.est = None
		self.f_L = self.f.L
		self.f_M = self.f.M
		self.solve = self.gss
		self.type_stop = 'gss'
		self.stop = self.CurGrad
	def estimate_grad(self, a, b):
		if self.axis == 'y':
			l1, l2 = self.value, (b+a)/2
			g = self.f.g1
			L_gk = self.f.g1L
		if self.axis == 'x':
			l1, l2 = (b+a)/2, self.value
			g = self.f.g2
			L_gk = self.f.g2L
		delta = (b-a)/2

		a = self.f.a
		grad = lambda x: self.f.f_der(x) + self.f.g1_der(x)*l1 + self.f.g2_der(x)*l2
		L = self.f.fL+ l1* self.f.g1L + l2 * self.f.g2L
		R = self.f.R0
		mu = 2*(1+l1+l2)
		M = L/ mu
		q = (np.sqrt(M)-1)/(np.sqrt(M)+1)
		alpha = 4/(np.sqrt(L)+np.sqrt(mu))**2
		beta = q**2
		x = np.zeros(a.shape)
		x, x_prev = x - alpha*grad(x), x
		while L_gk*R/self.f_M > abs(delta-abs(g(x))/self.f_M):
			x, x_prev = x - alpha * grad(x) + beta*(x-x_prev), x
			R *= q
		return delta-abs(g(x))/self.f_M<=0
	def CurGrad(self, a, b):
		return self.estimate_grad(a,b)
	def ConstEst(self, a, b):
		if self.est is None:
			M, R, L, eps = self.f_M, self.size, self.f_L, self.eps
			self.est = eps / (2 * M * R * math.sqrt(5) * (math.log((2 * L * R * math.sqrt(2)) / eps, 2)))
		return ((b - a) / 2 <= self.est)

	def heavy_ball(self, lambda1, lambda2, x0, R0, eps):
		a = self.f.a
		grad = lambda x: self.f.f_der(x) + self.f.g1_der(x)*lambda1 + self.f.g2_der(x)*lambda2
		L = self.f.fL+lambda1 * self.f.g1L + self.value * self.f.g2L
		mu = 2 * (1+lambda1 + lambda2)
		M = L/mu
		q = (np.sqrt(M)-1)/(np.sqrt(M)+1)
		R = R0
		alpha = 4/(np.sqrt(L)+np.sqrt(mu))**2
		beta = q**2
		x_prev = x0
		x = x_prev - alpha*grad(x_prev)
		while L * R >= eps:
			R *= q
			x,x_prev = x - alpha*grad(x)+beta*(x-x_prev), x
		return self.f.phi(lambda1, lambda2)(x), x, R
	def get_delta(self, lambda1, lambda2, x1 = None, x2 = None, R = None):
		R1, R2 = self.f.R0, self.f.R0
		if x1 is None:
			x1 = np.zeros(self.f.a.shape)
		elif not R is None:
			R1 = R
		if x2 is None:
			x2 = np.zeros(self.f.a.shape)
		elif not R is None:
			R2 = R
		delta = self.f_L * abs(lambda1-lambda2)
		f1, x1, R1 = self.heavy_ball(lambda1, lambda2, x1, R1, delta)
		f2, x2, R2 = self.heavy_ball(lambda1, lambda2, x2, R2, delta)
		return f1-f2, x1, x2, max(R1,R2)
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
		mystop = self.stop
		x1,x2,R = None, None, None
		while not mystop(a, b):
			delta, x1, x2, R = self.get_delta(c,d, x1, x2, R) 
			if delta  < 0:
				b = d
				d, f_d = c, f_c
				x1, x2 = None, x1
				c = b - (b- a) / gr
			else:
				a = c
				c, f_c = d, f_d
				x1,x2 = x2, None
				d = a + (b - a) / gr
			N+=1
			if N >= 200:
				return (b+a)/2
		return (b + a) / 2

	def add_cond(self, x, y):
		eps = self.eps
		if eps is None:
			return False
		if np.linalg.norm(self.f.gradient(x, y)) <= eps / (self.size * math.sqrt(2)):
			return True
		return False

	def halving_square(self, **kwargs):
		if kwargs.__contains__('eps'):
			self.eps = kwargs['eps']
			del kwargs['eps']
		eps = self.eps
		m = self.f.min if hasattr(self.f, 'min') else None
		cond, args = get_cond(f = self.f.calculate_function, eps = eps, minimum = m, **kwargs)
		Q = self.Q.copy()
		N = 0
		x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
		results = [(x_0, y_0)]
		if self.add_cond(x_0, y_0):
			return ((x_0, y_0), N, results, args)
		f_opt = self.f.calculate_function(x_0, y_0)
		while True:
			y_0 = (Q[2] + Q[3]) / 2
			self.axis, self.value, self.segm = 'x', y_0, [Q[0], Q[1]]
			x_0 = self.solve()
			if self.add_cond(x_0, y_0):
				return ((x_0, y_0), N, results,args)
			der = self.f.der_y(x_0, y_0, False)
			if der > 0:
				Q[2], Q[3] = Q[2],  y_0
			else:
				Q[3], Q[2] = Q[3],  y_0
			
			x_0 = (Q[0] + Q[1]) / 2
			self.axis, self.value, self.segm = 'y', x_0, [Q[2], Q[3]]
			y_0 = self.solve()
			if self.add_cond(x_0, y_0):
				return ((x_0, y_0), N, results, args)
			der = self.f.der_x(x_0, y_0, False)
			if der > 0:
				Q[0], Q[1] = Q[0],  x_0
			else:
				Q[1], Q[0] = Q[1],  x_0

			N += 1
			x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
			results.append((x_0, y_0))
			f_opt = self.f.calculate_function(x_0, y_0) 
			if cond(args, x = (x_0, y_0), N = N, minimum = m, eps = eps, size_Q = Q[1]-Q[0]):
				if N >= 100:
					N = -1
				if kwargs.__contains__('res'):
					kwargs['res'][0][kwargs['res'][1]] = ((x_0, y_0), N, results, args, self.f.values)
				return (x_0, y_0), N,results, args

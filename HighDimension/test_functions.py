#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
class QuadraticFunction:
	def __init__(self, n = None, C = 0, mu = 0.1, L_max = 1, L_min = 0.1, way = 'random'):
		if n is None:
			n = 100
		if way == 'random':
			A = np.random.uniform(-1, 1, (n,n))
			A = A.T.dot(A)+ np.eye(n)*C
		if way == 'control':
			a = np.random.uniform(L_min, L_max, (n,))
			a[np.argmax(a)] = L_max
			a[np.argmin(a)] = L_min
			Q, _ = np.linalg.qr(np.random.uniform(-1, 1, (n,n)))
			L = np.diag(np.array(a))
			A = Q@L@Q.T
		self.A_ = A
		self.eig_value = np.linalg.eig(A)[0]
		self.eig_value.sort()
		b = np.random.uniform(-10, 10, (n,))

		self.A = A
		self.b = b
		self.n = n
		self.L_full_grad = self.eig_value[-1]
		self.mu_full = self.eig_value[0]
		self.L, self.mu, self.grad = [], [], []
		self.grad = [lambda x:self.get_grad(x)[ind] for ind in range(self.n)]
		print(self.L_full_grad/self.mu_full, self.mu_full)
		self.get_params()
		
	def get_params(self):
		for ind in range(self.n):
			A = np.delete(np.delete(self.A, ind, 0), ind, 1)
			eig = np.linalg.eig(A)[0]
			eig.sort()
			#self.mu.append(self.mu_full)
			#self.L.append((self.L_full_grad, self.L_full_grad))
			self.mu.append(eig[0])
			self.L.append((2*np.linalg.norm(self.A[ind,:]), 2 * eig[-1])) 

	def func_value(self, x):
		#return np.linalg.norm(self.A_.dot(x) - self.b)**2
		return x.T@self.A@x - 2*self.b.T@x
	
	def get_grad(self, x, without = None, only_ind = None):
		g = list()
		if without is None and only_ind is None:
			return 2 *self.A.dot(x) - 2 * self.b
		if not only_ind is None:
			for ind in only_ind:
				g.append(2*self.A[ind,:].dot(x) - 2*self.b[ind])
			return np.array(g)
		if without is None:
			without = []
		indexes = [i for i,_ in enumerate(x) if not i in without]
		for ind in indexes:
			g.append(2*self.A[ind,:].dot(x) - 2*self.b[ind])
		return np.array(g)

	def get_square(self):
		lim = 2 * np.linalg.norm(self.b)/self.mu_full
		return [[-lim, lim] for i in range(self.n)]

	def get_start_point(self, ind, new_Q):
		b = np.delete(self.b+self.A[ind,:], ind)
		A = np.delete(np.delete(self.A, ind, 0), ind, 1)
		eig = np.linalg.eig(A)[0]
		eig.sort()
		lmin = eig[0]
		lim = 2 * np.linalg.norm(b)/lmin
		s = 0
		for i in new_Q:
			if type(i)==type(list()):
				s += (i[0]-i[1])**2
		return np.zeros((self.n-1,)), lim
		
class LogSumExp:
	def __init__(self, d_primal = 100, d_dual = 2, C = 1):
		self.d_primal, self.d_dual = d_primal, d_dual
		self.C = C
		v = 1
		self.v = v
		self.f = lambda x: np.log(1 + np.exp(v*x).sum()) + np.linalg.norm(x)**2*self.C
		self.b = []
		self.g_mu, self.g_L, self.g_M = [], [], []
		self.g, self.g_der = [], []
		for i in range(d_dual):
			b = np.random.uniform(-1,1,(n,))
			self.b.append(b)
			self.g_mu.append(0)
			self.g_L.append(np.linalg.norm(b1))
			self.g_M.append(0)
			self.g.append(lambda x: self.b.dot(x) - 1)
			self.g_der.append(b)

		B = np.vstack(tuple(self.b))
		l = np.linalg.eig(B.dot(B.T))[0]
		self.lmin, self.lmax = l.min(), l.max()
		
		g = lambda x: np.array([i(x) for i in self.g])
		self.phi = lambda lambda_: lambda x: -(self.f(x) + lambda_.dot(g(x))
		
		self.L = None
		self.M = None

		self.values = dict()
		self.Q = self.get_square()
		
		self.R0 = self.get_R0()
		print('R0', self.R0)
		
		self.fL = self.v + 2 * self.C
		self.fmu  = 2*self.C
		
		cond_num = (4*self.fL/self.fmu * self.lmax/self.lmin)
		print('cond_num', cond_num)


	def get_R0(self):
		l_max = self.Q[0][1]
		R0 = np.sqrt(2*np.log(self.d_primal+1))/self.C
		return R0

	def f_der(self, x):
		m = lambda x: (self.v*x).max()
		grad = lambda x: np.exp(self.v*x-m(x)) / (1/np.exp(m(x))+np.exp(self.v*x-m(x)).sum())
		return grad(x)

	def lipschitz_function(self, Q):
		if self.L is None:
			norm = lambda x: np.linalg.norm(x)
			self.L = sum([norm(b) for b in self.b]) * self.R0
		return self.L

	def lipschitz_gradient(self, Q):
		if self.M is None:
			self.M = sum(self.g_L)**2 / self.C
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

	def get_x(self, lambda_, cond, warm = None):
		f = lambda x: self.phi(l1,l2)(x)
		g_grad = lambda x: [lambda[ind] * i for ind,i in enumerate(self.g_der)]
		grad = lambda x: self.f_der(x) + sum(g_grad(x))
		L = self.fL + lambda_.dot(np.array(self.g_L))
		mu = self.fmu + lambda_.dot(np.array(self.g_mu))
		M = L/mu
		q = (M-1)/(M+1)
		R = self.R0
		alpha = 1/(L+mu)
		x = np.zeros(self.a.shape)
		if lambda_ in self.values:
			x, R = self.values[tuple(lambda_)]
		R *= L/2
		x, x_prev= x - 1/L * grad(x), x
		R *= 1/5
		N = 1
		while not cond(x,R):
			R *= min(q, (N+4)/(N+5))
			x = x - alpha *grad(x)
			N += 1
		self.values[tuple(lambda_)] = (x,R)
		return x


	def get_square(self):
		x = np.zeros(self.d_primal)
		gamma = min([-i(x) for i in self.g])
		q = self.f(x) / gamma
		self.Q = [[0,q] for i in range(d_dual)]
		return self.Q

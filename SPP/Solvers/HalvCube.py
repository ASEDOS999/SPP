#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

class Dichotomy:
	def __init__(self, history = {}, key = "Dichotomy"):
		self.R = 0
		self.L, self.M = np.infty, np.infty
		self.mu = 0
		self.n = np.infty
		self.R = np.infty
		self.hist = []
		self.history = history
		self.key = key
		self.indexes = {}
		
	def help_task(self, f, Q, ind):
		Q = list(Q)
		Q_new = Q[:ind] + Q[ind+1:]
		return Q_new

	def get_new_eps(self, eps):
			return self.mu * eps**2 / (128 * self.L**2 * self.n * self.R)
		
	def cond(self, x, Q, eps, ind):
		Q_ = np.array(self.Q)
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		est = lambda g: self.Est(eps, g, R)
		def stop_cond(y, R, x, ind, est = None, L = self.L, f = self.f, L_yy = self.f.L_yy):
			lipschitz_estimate = L_yy * R
			g = f.grad_y(x, y)[ind]
			if g == 0:
				return True
			est = est(g)
			if lipschitz_estimate / L <= est:
				return True
			s = min(min(x - Q_[:, 0]), min(Q_[:, 1] - x))
			if s >= np.sqrt(R / (2 * L)):
				return np.sqrt(L * R / 2) <= est
			return False
		return lambda y, R: stop_cond(y, R, x, ind, est = est)
	
	def fix(self, fixed_value, ind, indexes):
		new_indexes = indexes.copy()
		keys = list(new_indexes.keys())
		keys.sort()
		u = 0
		while u<len(keys) and keys[u] <= ind:
			ind += 1
			u += 1
		new_indexes[ind] = fixed_value
		def reconstruct(x, indexes = new_indexes):
			n = len(x) + len(indexes)
			new_x = np.zeros((n,))
			for ind,i in enumerate(new_x):
				if ind in indexes:
					new_x[ind] = indexes[ind]
				else:
					new_x[ind] = None
			x = list(x)
			for ind, i in enumerate(new_x):
				if np.isnan(i):
					new_x[ind] = x.pop(0)
			return new_x
		return lambda x: reconstruct(x), new_indexes, ind
		
	def Est1(self, g):
		est = abs(g)/self.L
		return est
	
	def Est2(self, eps, g, R):
		est = (eps - R * abs(g))/(self.M + self.L*R)
		return est
	
	def Est(self, eps, g, R):
		return max(self.Est1(g), self.Est2(eps, g, R))/2
	
	def Halving(self, f, Q, eps, indexes = {}, time_max = None):
		if self.L == np.infty:
			# There was not initialization
			self.f = f
			self.L = f.L_xx
			self.M = f.M_x
			self.n = len(Q)
			self.Q = Q.copy()
			self.mu = f.mu_x
			Q_ = np.array(Q)
			self.R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
			self.history[self.key] = [((Q_[:,0] + Q_[:,1])/2, time.time())]
		Q_ = np.array(Q)
		if len(Q) == 0:
			# This set includes only one point
			return [], 0
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		while True:
			for ind, i in enumerate(Q):
				# Fix coordinat with index 'ind'
				Q_new = self.help_task(f, Q, ind)
				# The new function of reconstructing x
				reconstruct, new_indexes, true_ind = self.fix(sum(i)/2, ind, indexes.copy())
				
				# Solution of the new problem through this method
				x, Delta  = self.Halving(f, Q_new, self.get_new_eps(eps), new_indexes)

				# Calculate delta-subgradient
				grad = f.get_delta_grad(reconstruct(x),
									 self.cond(reconstruct(x), Q, eps, true_ind))
				g = grad[true_ind]
				x_ = list(x)
				x = x_[:ind] + [sum(i)/2] + x_[ind:]
				x = np.array(x)				
				if g == 0:
					return x, 0
				
				# Try stop condition at point x
				Est = self.Est(eps, g, R)
				if len(Q) < self.n:
					# It is not main problem
					if Delta <= Est:
						return x, R
				if len(Q) == self.n:
					# It is initial square
					# Update History
					self.history[self.key].append((x, time.time()))
					# Try condition
					if self.M * R <= eps:
						return x, R
					if not time_max is None:
						if self.history[self.key][-1][1] - self.history[self.key][0][1] >time_max:
							return x, R
				
				# Choice of multidimensional rectangle
				c = sum(i)/2
				if g > 0:
					Q[ind] = [i[0], c]
				else:
					Q[ind] = [c, i[1]]
				# Update estimation of distance to point solution
				Q_ = np.array(Q)
				R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		return x, R

class Dichotomy_exact:
	def __init__(self, history = {}, key = "Dichotomy"):
		self.R = 0
		self.L, self.M = np.infty, np.infty
		self.mu = 0
		self.n = np.infty
		self.R = np.infty
		self.hist = []
		self.history = history
		self.key = key
		self.indexes = {}
		
	def help_task(self, Q, ind):
		Q = list(Q)
		Q_new = Q[:ind] + Q[ind+1:]
		return Q_new

	def get_new_eps(self, eps):
			return self.mu * eps**2 / (128 * self.L**2 * self.n * self.R)
		
	def cond(self, x, Q, eps, ind):
		Q_ = np.array(self.Q)
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		est = lambda g: self.Est(eps, g, R)
		def stop_cond(y, R, x, ind, est = None, L = self.L, f = self.f, L_yy = self.f.L_yy):
			lipschitz_estimate = L_yy * R
			g = f.grad_y(x, y)[ind]
			if g == 0:
				return True
			est = est(g)
			if lipschitz_estimate / L <= est:
				return True
			s = min(min(x - Q_[:, 0]), min(Q_[:, 1] - x))
			if s >= np.sqrt(R / (2 * L)):
				return np.sqrt(L * R / 2) <= est
			return False
		return lambda y, R: stop_cond(y, R, x, ind, est = est)
	
	def fix(self, fixed_value, ind, indexes):
		new_indexes = indexes.copy()
		keys = list(new_indexes.keys())
		keys.sort()
		u = 0
		while u<len(keys) and keys[u] <= ind:
			ind += 1
			u += 1
		new_indexes[ind] = fixed_value
		def reconstruct(x, indexes = new_indexes):
			n = len(x) + len(indexes)
			new_x = np.zeros((n,))
			for ind,i in enumerate(new_x):
				if ind in indexes:
					new_x[ind] = indexes[ind]
				else:
					new_x[ind] = None
			x = list(x)
			for ind, i in enumerate(new_x):
				if np.isnan(i):
					new_x[ind] = x.pop(0)
			return new_x
		return lambda x: reconstruct(x), new_indexes, ind
		
	def Est1(self, g):
		est = abs(g)/self.L
		return est
	
	def Est2(self, eps, g, R):
		est = (eps - R * abs(g))/(self.M + self.L*R)
		return est
	
	def Est(self, eps, g, R):
		return max(self.Est1(g), self.Est2(eps, g, R))/2
	
	def Halving(self, func, 
					   grad,
					   L, mu, eps,
					   start_point = None,
					   cond = None,
					   Q = None,
					   indexes = {}):
		if self.L == np.infty:
			# There was not initialization
			self.L = L
			x, R = start_point
			Q = np.hstack([x + R * np.ones(x.shape), x - R * np.ones(x.shape)])
			self.n = len(Q)
			self.Q = Q.copy()
			self.mu = mu
			Q_ = np.array(Q)
			self.R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
			self.history[self.key] = [((Q_[:,0] + Q_[:,1])/2, time.time())]
		Q_ = np.array(Q)
		if len(Q) == 0:
			# This set includes only one point
			return [], 0
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		while True:
			for ind, i in enumerate(Q):
				# Fix coordinat with index 'ind'
				Q_new = self.help_task(Q, ind)
				# The new function of reconstructing x
				reconstruct, new_indexes, true_ind = self.fix(sum(i)/2, ind, indexes.copy())
				
				# Solution of the new problem through this method
				x, Delta  = self.Halving(func, grad, L, mu, self.get_new_eps(eps), Q = Q_new, indexes = new_indexes)

				# Calculate delta-subgradient
				grad = grad(reconstruct(x))
				g = grad[true_ind]
				x_ = list(x)
				x = x_[:ind] + [sum(i)/2] + x_[ind:]
				x = np.array(x)				
				if g == 0:
					return x, 0
				
				# Try stop condition at point x
				Est = self.Est(eps, g, R)
				if len(Q) < self.n:
					# It is not main problem
					if Delta <= Est:
						return x, R
				if len(Q) == self.n:
					# It is initial square
					# Update History
					self.history[self.key].append((x, time.time()))
					# Try condition
					if cond(x, np.sqrt(Est/mu)):
						return x, R
				
				# Choice of multidimensional rectangle
				c = sum(i)/2
				if g > 0:
					Q[ind] = [i[0], c]
				else:
					Q[ind] = [c, i[1]]
				# Update estimation of distance to point solution
				Q_ = np.array(Q)
				R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		return x, R
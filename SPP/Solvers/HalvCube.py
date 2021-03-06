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
		
	def cond(self, x, Q, eps, ind, delta = np.infty):
		Q_ = np.array(self.Q)
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		est_ = lambda g: self.Est(eps, g, R)
		def stop_cond(y, R = None, f_est = None):
			if f_est is None:
				f_est = self.f.M_y * R
			if f_est >= delta:
				return False
			lipschitz_estimate = self.f.L_xy * R
			g = self.f.grad_x(x, y)[ind]
			est = est_(g)
			#print("HC", self.f.L_xy, self.L, lipschitz_estimate / self.L, g, est)
			if lipschitz_estimate / self.L <= est:
				return True
			s = min(min(x - Q_[:, 0]), min(Q_[:, 1] - x))
			if s >= np.sqrt(R / (2 * self.L)):
				return np.sqrt(self.L * R / 2) <= est
			return False
		return stop_cond
	
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
	
	def Halving(self, f, Q, eps, indexes = {}, time_max = None, stop_cond = lambda *args: False, eps_R = None, out_ind = None):
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
			self.time_max = time_max
		if eps_R is None:
			eps_R =  (eps)
		Q_ = np.array(Q)
		if len(Q) == 0 and self.n != 0:
			return [], 0, False
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		while True:
			for ind, i in enumerate(Q):
				# Fix coordinat with index 'ind'
				Q_new = self.help_task(f, Q, ind)
				# The new function of reconstructing x
				reconstruct, new_indexes, true_ind = self.fix(sum(i)/2, ind, indexes.copy())
				
				# Solution of the new problem through this method
				if len(Q) == self.n:
					new_eps = eps
				else:
					new_eps = self.get_new_eps(eps)
				x, Delta, stop  = self.Halving(f, Q_new, new_eps, new_indexes, out_ind = ind)
				if stop and len(Q)<self.n:
					x_ = list(x)
					x_.insert(ind, sum(i)/2)
					x = np.array(x_)
					return x, R, False
					
				# Calculate inexact subgradient
				x_reconstructed = reconstruct(x)
				if R <= eps_R and len(Q) == self.n:
					cond_grad = self.cond(x_reconstructed, Q, eps, true_ind, delta = eps)
				else:
					cond_grad = self.cond(x_reconstructed, Q, eps, true_ind)
				grad, y = f.get_delta_grad(x_reconstructed, cond_grad)
				
				g = grad[true_ind]
				#print(g)
				# Choice of multidimensional rectangle
				c = sum(i)/2
				if g >= 0:
					Q[ind] = [i[0], c]
				else:
					Q[ind] = [c, i[1]]
				x_ = list(x)
				x_.insert(ind, sum(i)/2)
				x = np.array(x_)
				
				
				# Update estimation of distance to point solution
				Q_ = np.array(Q)
				R = np.linalg.norm(Q_[:,0] - Q_[:,1])		
				
				# Not initial problem
				if len(Q) < self.n:
					g = grad[out_ind]
					if R <= self.Est2(eps, g, R):
						return x, R, True
					if R <= self.Est1(g):
						return x, R, False
					if time.time()-self.history[self.key][0][1]> self.time_max:
						return x, R, False
				
				# Initial problem
				if len(Q) == self.n:
					# Update History
					Q_ = np.array(Q)
					#x = (Q_[:, 0] + Q_[:, 1])/2
					self.history[self.key].append(((x,y), time.time()))
					# Try condition
					if not time_max is None:
						if self.history[self.key][-1][1] - self.history[self.key][0][1] >time_max:
							return x, R
					#print(R, self.f.M_x, eps/4)
					#if (stop_cond(x, y) or f.M_x *R <= eps/4) and R < eps_R:
					if stop_cond(x, y) and R < eps_R:
						return x, R
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
					   get_grad,
					   L, mu, eps,
					   start_point = None,
					   cond = None,
					   Q = None,
					   indexes = {},
					   out_ind = None):
		if self.L == np.infty:
			# There was not initialization
			self.L = L
			x, R = start_point
			Q = np.vstack([x - R * np.ones(x.shape), x + R * np.ones(x.shape)]).T
			self.n = len(Q)
			self.Q = Q.copy()
			self.mu = mu
			Q_ = np.array(Q)
			self.R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		Q_ = np.array(Q)
		if len(Q) == 0:
			# This set includes only one point
			return [], 0, False
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		while True:
			for ind, i in enumerate(Q):
				# Fix coordinat with index 'ind'
				Q_new = self.help_task(Q, ind)
				# The new function of reconstructing x
				reconstruct, new_indexes, true_ind = self.fix(sum(i)/2, ind, indexes.copy())
				
				# Solution of the new problem through this method
				if len(Q) == self.n:
					new_eps = eps
				else:
					new_eps = self.get_new_eps(eps)
				x, Delta, stop  = self.Halving(func, get_grad, L, mu, new_eps, Q = Q_new, indexes = new_indexes, out_ind = ind)
				if stop:
					x_ = list(x)
					x_.insert(ind, sum(i)/2)
					x = np.array(x_)
					if len(Q) < self.n:
						return x, R, False
					else:
						return x, R
				# Calculate delta-subgradient
				grad = get_grad(reconstruct(x))
				g = grad[true_ind]
				x_ = list(x)
				x = x_[:ind] + [sum(i)/2] + x_[ind:]
				x = np.array(x)				
				if g == 0:
					if len(Q) < self.n:
						return x, 0, False
					else:
						return x, 0

				# Choice of multidimensional rectangle
				c = sum(i)/2
				if g > 0:
					Q[ind] = [i[0], c]
				else:
					Q[ind] = [c, i[1]]
				# Update estimation of distance to point solution
				Q_ = np.array(Q)
				if R <= 1e-15:
					if len(Q) < self.n:
						return x, R, False
					else:
						return x, R
				R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
				# Try stop condition at point x
				if len(Q) < self.n:
					# It is not main problem
					g = grad[out_ind]
					Est = self.Est(eps, g, R)
					if R<= self.Est2(eps,g,R):
						return x, R, True
					if R <= Est:
						return x, R, False
				if len(Q) == self.n:
					# Try condition
					if cond(x, R = R):
						return x, R
		return x, R
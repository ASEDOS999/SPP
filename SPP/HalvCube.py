#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time

class Dichotomy:
	def __init__(self, f, Q):
		self.R = 0
		self.L, self.M = np.infty, np.infty
		self.mu = 0
		self.n = np.infty
		self.R = np.infty
		self.hist = []
		
	def help_task(self, f, Q, ind):
		Q_new = Q[:ind] + Q[ind+1:]
		f_new = fix(f, ind, sum(Q[i])/2)
		return f_new, Q_new

	def get_new_eps(self, eps):
			return self.mu * eps**2 / (128 * self.L**2 * self.n * self.R)
		
	def cond(self, x, Q, eps):
		def stop_cond(y, eps, R, f, x, Q, L_yy = f.L_yy, L = self.L):
			c1 = L_yy * R <= abs(f.grad_y(x, y)[ind])
			if c1:
				return c1
			s = 0
			for ind, i in enumerate(Q):
				s += (min(x[ind] - i[0], i[1] - x[ind]))**2
			if s >= np.sqrt(eps / (2 * L)):
				return np.sqrt(L * eps / 2) <= abs(f.grad_y(x, y)[ind])
			return False
		return lambda y, eps, R: stop_cond(y, eps, R, self.f, x, Q)
	
	def fix(self, ind, n):
		def reconstruct(x, ind, n):
			# ...
		return lambda x: reconstuct(x, ind, n)
		
	def Halving(self, f, Q, eps, reconstruct = lambda x: x):
		if self.L == np.infty:
			self.L = f.L_xx
			self.M = f.M_xx
			self.n = len(Q)
			self.mu = f.mu_xx
			Q_ = np.array(Q)
			self.R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		Q_ = np.array(Q)
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		while (self.L * R > eps):
			for ind, i in enumerate(Q):
				f_new, Q_new = self.help_task(f, Q, ind)
				new_reconstruct = self.fix(ind, x.len(x))
				reconstrunct = lambda x: new_reconstuct(reconstruct(x))
				x, Delta  = self.Halving(f, Q_new, self.get_new_eps(eps), reconstruct)
				x_ = list(x)
				x = x[:ind] + [sum(i)/2] + x[ind:]
				g = f.get_delta_grad(reconstruct(x), self.cond(x, Q, eps))[ind]				
				if g = 0:
					return x, R*2
				c = sum(i)/2
				if g > 0:
					Q[ind] = [i[0], c]
				else:
					Q[ind] = [c, i[1]]
				R = R + (-(i[1]-i[0])**2 + (Q[ind][1]-Q[ind][0])**2) / 2
		return x, R
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
		
	def help_task(self, f, Q, ind):
		Q_new = Q[:ind] + Q[ind+1:]
		return Q_new

	def get_new_eps(self, eps):
			return self.mu * eps**2 / (128 * self.L**2 * self.n * self.R)
		
	def cond(self, x, Q, eps, ind):
		Q_ = np.array(Q)
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		est = lambda g: self.Est(eps, g, R)
		def stop_cond(y, R, x, ind, est = None, L = self.L, f = self.f, L_yy = self.f.L_yy):
			lipschitz_estimate = L_yy * R
			g = f.grad_y(x, y)[ind]
			est = est(g)
			if lipschitz_estimate / L <= est:
				return True
			s = min(min(x[ind] - Q[:, 0]), min(Q[:, 1] - x[ind]))
			if s >= np.sqrt(R / (2 * L)):
				return np.sqrt(L * R / 2) <= est
			return False
		return lambda y, R: stop_cond(y, R, x, ind, est = est)
	
	def fix(self, fixed_value, ind, n):
		def reconstruct(x, fixed_value, ind):
			x = list(x)
			x = x[:ind] + [fixed_value] + x[ind:]
			return np.array(x)
		return lambda x: reconstruct(x, fixed_value, ind, n)
		
	def Est1(self, g):
		return abs(g)/self.L
	
	def Est2(self, eps, g, R):
		return (eps - R * abs(g))/(self.M + self.L*R)
	
	def Est(self, eps, g, R):
		return max(self.Est1(g), self.Est2(eps, g, R))/2
	
	def Halving(self, f, Q, eps, reconstruct = lambda x: x):
		if self.L == np.infty:
			# There was not initialization
			self.L = f.L_xx
			self.M = f.M_xx
			self.n = len(Q)
			self.mu = f.mu_x
			Q_ = np.array(Q)
			self.R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
			self.history[self.key] = [((Q_[:,0] + Q_[:,1])/2, time.time())]
		Q_ = np.array(Q)
		if len(Q) == 0:
			# This set includes only one point
			return [], 0
		R = np.linalg.norm(Q_[:,0] - Q_[:,1])/2
		while (self.L * R > eps):
			for ind, i in enumerate(Q):
				# Fix coordinat with index 'ind'
				Q_new = self.help_task(f, Q, ind)
				
				# The new function of reconstructing x
				new_reconstruct = self.fix(sum(i)/2, ind)
				new_reconstruct = lambda x: reconstruct(new_reconstruct(x))
				
				# Solution of the new problem through this method
				x, Delta  = self.Halving(f, Q_new, self.get_new_eps(eps), new_reconstruct)
				
				# Reconstuct accoding to current cube
				x_ = list(x)
				x = x_[:ind] + [sum(i)/2] + x_[ind:]
				x = np.array(x)
				
				# Calculate delta-subgradient
				g = f.get_delta_grad(reconstruct(x),
									 self.cond(reconstruct(x), Q, eps, ind))[ind]				
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
				
				# Choice of multidimensional rectangle
				c = sum(i)/2
				if g > 0:
					Q[ind] = [i[0], c]
				else:
					Q[ind] = [c, i[1]]
					
				# Update estimation of distance to point solution
				R = R + (-(i[1]-i[0])**2 + (Q[ind][1]-Q[ind][0])**2) / 2
		return x, R
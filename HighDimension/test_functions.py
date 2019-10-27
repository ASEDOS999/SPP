#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
class QuadraticFunction:
	def __init__(self, n = None):
		if n is None:
			n = 100
		A = np.random.uniform(-1, 1, (n,n))
		self.A_ = A
		A = A.T.dot(A)
		self.eig_value = np.linalg.eig(A)[0]
		self.eig_value.sort()
		b = np.random.uniform(-1, 1, (n,))

		self.A = A
		self.b = b
		self.n = n
		self.L_full_grad = self.eig_value[-1]
		self.mu_full = self.eig_value[0]
		self.L, self.mu, self.grad = [], [], []
		self.get_params()
		
	def get_params(self):
		self.grad = [lambda x:(2*self.A[ind,:].dot(x) - 2*self.b[ind]) for ind in range(self.n)]
		for ind in range(self.n):
			A = np.delete(np.delete(self.A, ind, 0), ind, 1)
			eig = np.linalg.eig(A)[0]
			eig.sort()
			self.mu.append(eig[0])
			self.L.append((self.L_full_grad, eig[-1])) 

	def func_value(self, x):
		return np.linalg.norm(self.A_.dot(x) - self.b)**2
	
	def get_grad(self, x, without = None):
		g = 2*self.A.dot(x) - 2 * self.b
		if without is None:
			return g
		else:
			return np.delete(g, without)
	
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
		return np.zeros((self.n-1,)), lim
		
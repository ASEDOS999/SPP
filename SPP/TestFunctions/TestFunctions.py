#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:04:56 2020

@author: elias
"""

import numpy as np

# There is implementation of parent classes for to implement functions
# S(x,y) = r(x) + F(x,y) - h(y)
# G(x) = max_y S(x,y)

class ConvFunc_OneArg:
	# The parent class for convex function of one argument
	# The functions r and f will be implemented through this class 
	def __init__(self):
		self.L, self.M = 0,0
		self.mu = 0
		
	def get_value(self, x):
		return 0
	
	def grad(self, x):
		return np.zeros(x.shape)
	
class ConvConcFunc:
	# Convex-concave function
	# The fynction F(X,y) will be implemented through this class
	def __init__(self):
		self.L_xx, self.L_yy = 0,0
		self.L_yx, self.L_xy = 0,0
		self.mu_y, self.mu_x = 0, 0
		self.M_x, self.M_y = 0,0
		
	def get_value(self, x, y):
		return 0

	def grad_y(self, x, y):
		return np.zeros(y.shape)
	
	def grad_x(self, x, y):
		return np.zeros(x.shape)

class TestFunction:
	# It is class for to implement function S through functions r, F, h
	def __init__(self, r = None, F = ConvConcFunc(), h = ConvFunc_OneArg, solver = None, get_start_point = None):
		# Arguments 'r' and 'h' are object of 'ConvFunc_OneArg' class
		# Argument 'F' is object of 'ConvConcFunc' class
		# S(x,y) = r(x) + F(x, y) - h(y)
		
		if r is None:
			r = ConvFunc_OneArg()
		if F is None:
			F = ConvConcFunc()
		if h is None:
			h = ConvFunc_OneArg()
		self.r = r
		self.F = F
		self.h = h
		
		# Solver for the internal problem
		# It should be callable object 
		# solver(func, grad, L, mu, start_point, cond)
		# 'func' is minimized function
		# 'gtad  is gradient of this function
		# 'L' and 'mu' are constants of Lipshitz and strong convexity
		# 'start_point' is a tuple of start point and distance to solution (x, R)
		# 'cond' is the stop-condition for solver
		self.solver = solver
		
		# Function for to get start point and estimation of distance between it
		# and the point-solution of the internal problem with fixed x
		self.get_start_point = get_start_point
		
		# Lipshitz constants for gradient according to notation in the article
		self.L_xx = r.L + F.L_xx + (F.L_xy)**2 / (F.mu_y+h.mu)
		self.L_yy = h.L + F.L_yy
		
		# Lipschitz constants for function
		self.M_x = r.M + F.M_x
		self.M_y = h.M + F.M_y
		
		# Constants of strong convexity (concavity) on x (y)
		self.mu_x = r.mu
		self.mu_y = h.mu
		
	def get_value(self, x, y):
		return self.r.get_value(x) + self.F.get_value(x, y) - self.h.get_value(y)
	def grad_y(self, x, y):
		# Gradient with respect to y for function S
		return self.F.grad_y(x, y) - self.h.grad(y)
	
	def grad_x(self, x, y):
		# Gradient with respect to y for function S
		return self.r.grad(x) + self.F.grad_x(x, y)
		
	def get_delta_grad(self, x, cond = None):
		# G(x) = max_y S(x,y)
		# The method should return some delta-subgradient of function G
		start_point = self.get_start_point(x)
		y, eps = self.solver(func = lambda y: -(self.F.get_value(x, y) - self.h.get_value(y)), 
					   grad = lambda y: -self.grad_y(x, y),
					   L = self.L_yy, mu = self.mu_y,
					   start_point = start_point,
					   cond = cond)
		return self.grad_x(x, y)
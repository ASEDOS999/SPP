#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:04:56 2020

@author: elias
"""

import numpy as np

class ConvFunc_OneArg:
	def __init__(self):
		self.L, self.M = np.infty, np.infty
		self.mu = 0
		
	def get_value(self, x):
		return 0
	
	def grad(self, x):
		return np.zeros(x.shape)
	
class ConvConcFunc:
	def __init__(self):
		self.L_xx, self.L_yy = np.infty, np.infty
		self.L_yx, self.L_xy = np.infty, np.infty
		self.mu_y, self.mu_x = 0, 0
		self.M_x, self.M_y = np.infty, np.infty
		
	def get_value(self, x, y):
		return 0

	def grad_y(self, x, y):
		return np.zeros(y.shape)
	
	def grad_x(self, x, y):
		return np.zeros(x.shape)

class TestFunction:
	def __init__(self, r, F, h, solver, get_start_point):
		# S(x,y) = r(x) + F(x, y) - h(y)
		self.r = r
		self.F = F
		self.h = h
		
		# Solver for the internal problem
		self.solver = solver
		
		# Function for to get start point and estimation of distance between it
		# and the point-solution of the internal problem with fixed x
		self.get_start_point = get_start_point
		
		# Lipshitz constants for gradient
		self.L_xx = r.L + F.L_xx + (F.L_xy)**2 / F.mu_y
		self.L_yy = h.L + F.L_yy
		
		# Lipschitz constant for function
		self.M_x = r.M + F.M_x
		
		# History of calculated delta-gradients
		self.history_x, self.history = [], []
		
	def grad_y(self, x, y):
		return self.F.grad_y(x, y) - self.h.grad(y)
	
	def grad_x(self, x, y):
		return self.r.grad(x) + self.F.grad_x(x, y)

	def update_history(self, x, y, eps, max_len = np.infty):
		self.history_x.append(x)
		self.history.append((y, eps))
		if len(self.history_x) > max_len:
			self.history = self.history[-max_len:]
		
	def get_delta_grad(self, x, cond = None):
		if x in self.history_x:
			start_point = self.history[self.history_x.index(x)]
		else:
			start_point = self.get_start_point(x)
		y, eps = self.solver(func = lambda y: -(self.F.get_value(x, y) - self.h.get_value(y)), 
					   grad = lambda y: -self.grad(x, y),
					   L = self.L_yy,
					   start_point = start_point,
					   cond = cond)
		self.update_history(x, y, eps)
		return self.grad_x(x, y)
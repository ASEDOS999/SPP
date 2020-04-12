#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Let's consider the following optimization problem:
	log(1 + \sum_{k=1}^n exp(alpha x_k)) + beta ||x||_2^2 -> min_x
	s.t. (b_j, x) - c_j <= 0, j = 1...m
	
This problem can be converted into saddle point problem:
	min_x max_{y>=0} S(x,y)
	r(x) = log(1 + \sum_{k=1}^n exp(alpha x_k)) + beta/2 ||x||_2^2
	F(x, y) = sum_{j=1}^m y_j(b_j, x) = (y, Bx)
	h(y) = (c, y)

Moreover, we can regularize the problem on y:
	h_new(y) = (c,y) + beta_eps/2 ||y||_2^2
	
In this file we realize this mathematical functions as classes
from TestFunctions.py
"""

from .TestFunctions import ConvFunc_OneArg,  ConvConcFunc
import numpy as np

class r(ConvFunc_OneArg):
	def __init__(self, alpha = 1, beta = 1, size_domain = 10):
		selph,alpha = alpha
		self.beta = beta
		self.L, self.M = alpha**2 + 2 * beta, alpha + 2 * beta * size_domain
		self.mu = beta
		
	def get_value(self, x):
		beta = self.beta
		alpha = self.alpha
		return np.log(1 + np.exp(alpha * x).sum()) + beta/2 * np.linalg.norm(x)**2
	
	def grad(self, x):
		beta = self.beta
		alpha = self.alpha
		return alpha * np.exp(alpha * x)/ (1 + np.exp(alpha * x).sum()) + beta*x

class h(ConvFunc_OneArg):
	def __init__(self, c, beta = 0, size_domain = 10):
		self.c = c
		self.beta = beta
		self.L, self.M = beta, np.linalg.norm(c) + beta/2 * size_domain
		self.mu = beta
		
	def get_value(self, y):
		return self.c.dot(y) + self.beta/2 * np.linalg.norm(y)**2
	
	def grad(self, y):
		return self.c + self.beta * y
	
class F(ConvConcFunc):
	def __init__(self, B, size_domain = 10):
		self.B = B
		self.L_xx, self.L_yy = 0, 0
		lambda_B = np.linalg.norm(self.B, ord=2)
		self.L_yx, self.L_xy = lambda_B, lambda_B
		self.mu_y, self.mu_x = 0, 0
		self.M_x, self.M_y = lambda_B * size_domain, lambda_B * size_domain

	def get_value(self, x, y):
		return y.dot(self.B @ x)

	def grad_y(self, x, y):
		return self.B @ x
	
	def grad_x(self, x, y):
		return self.B.T @ y

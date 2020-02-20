#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from TestFunctions import ConvFunc_OneArg,  ConvConcFunc
from TestFunctions import TestFunction
import numpy as np


class r(ConvFunc_OneArg):
	def __init__(self):
		self.L, self.M = 2, np.infty
		self.mu = 1
	def get_L(self):
		return self.L
		
	def get_value(self, x):
		return np.linalg.norm(x)**2
	
	def grad(self, x):
		return 2 * x

class h(ConvFunc_OneArg):
	def __init__(self):
		self.L, self.M = 2, np.infty
		self.mu = 1
		
	def get_value(self, y):
		return np.linalg.norm(y)**2
	
	def grad(self, y):
		return 2 * y
	
class F(ConvConcFunc):
	def __init__(self, alpha = 1):
		self.L_xx, self.L_yy = 0, 0
		self.L_yx, self.L_xy = 2, 2
		self.mu_y, self.mu_x = 1, 1
		self.M_x = 2
		self.alpha = alpha
	def get_value(self, x, y):
		return 2 * self.alpha * x.dot(y)

	def grad_y(self, x, y):
		return 2 * self.alpha * x
	
	def grad_x(self, x, y):
		return 2 * self.alpha * y

TrivialFunc = TestFunction(r(), F(), h(), None)
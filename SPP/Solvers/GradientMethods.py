#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

def cond_for_fgm(f, eps, R):
	L = f.L_xx
	if f.mu_x == 0:
		k = np.sqrt(4 * L * R/ eps)
		z = 1/3 * k+ 2.4
	else:
		k = min(np.sqrt(4 * L * R/ eps), 2 * np.sqrt(f.L_xx/f.mu_x) * np.log(L * R/ eps))
		z = min(1/3 * k+ 2.4, 1 + np.sqrt(f.L_xx/f.mu_x))
	def cond(y, R):
		return f.L_yy * R <= eps / (2 * z)
	return cond

def FGM_external(f, start_x, R, Q, eps = 0.001, history = {}, key = "FGM", time_max = None, stop_cond = lambda *args: False):
	# It is realization for FGM method to optimize Saddle-Point problem
	# on external variables
	N = 0
	cond = cond_for_fgm(f, eps, R)
	domain = np.array(Q)
	x = start_x
	history[key] = [(x.copy(), time.time())]
	grad = lambda x: f.get_delta_grad(x, cond)
	L = f.L_xx
	mu = f.mu_x
	alpha, A = 1, 1
	u, v = 2*L, 2*x
	while True:
		der, y_ = grad(x)
		history[key].append(((x.copy(),y_), time.time()))
		est = 2 * min(4 * L * R/N**2, L * R * np.exp(-N/2 * np.sqrt(mu/L)))
		if est <= eps:
			return x, N
		if not time_max is None:
			if history[key][-1][1] - history[key][0][1] >time_max:
				return x, R
		if stop_cond(x, y_):
			return x, R
		y = x - 1/L * der
		y = np.clip(y, *domain.T)

		u += mu * alpha
		v += alpha * (mu * x - der)
		z = v/u
		z = np.clip(z, *domain.T)
		tau = alpha/A

		x = tau * z + (1 - tau) * y
		alpha = (L+mu*A)/(2*L) * (1 + np.sqrt(abs(1 + (4*L*A)/(L+mu*A))) )
		A += alpha

		N += 1

def FGM_internal(func, 
					   grad,
					   L, mu,
					   start_point,
					   cond):
	# It is realization for FGM method to optimize Saddle-Point problem
	# on external variables
	N = 0
	x, R = start_point
	alpha, A = 1, 1
	u, v = 2*L, 2*x
	while True:
		der = np.array(grad(x))
		np.array(der)

		y = x - 1/L * der

		u += mu * alpha
		v += alpha * (mu * x - der)
		z = v/u
		tau = alpha/A
		x = tau * z + (1 - tau) * y

		alpha = (L+mu*A)/(2*L) * (1 + np.sqrt(abs(1 + (4*L*A)/(L+mu*A))) )
		A += alpha
		N += 1
		est = min(4 * L * R/N**2, L * R * np.exp(-N/2 * np.sqrt(mu/L)))
		if cond(x, np.sqrt(est/mu)):
			return (x, np.sqrt(est/mu))


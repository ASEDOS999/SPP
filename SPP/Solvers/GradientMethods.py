#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

def cond_for_fgm(f, eps, R):
	L = f.L_xx
	k = min(np.sqrt(4 * L * R/ eps), 2 * np.sqrt(f.L_xx/f.mu_x) * np.log(L * R/ eps))
	z = min(1/3 * k+ 2.4, 1 + np.sqrt(f.L_xx/f.mu_x))
	return lambda y, R : f.L * R <= eps / (2 * z)

def FGM_external(f, start_x, R, Q, eps = 0.001, history = {}, key = "FGM"):
	# It is realization for FGM method to optimize Saddle-Point problem
	# on external variables
	N = 0
	cond = cond_for_fgm(f, eps)
	domain = np.array(Q)
	x = start_x
	results = [(x.copy(), time.time())]
	grad = lambda x: f.get_delta_grad(x, cond)
	L = f.L_xx
	mu = f.mu_x
	alpha, A = 1, 1
	u, v = 2*L, np.zeros(x.shape)
	while True:
		der = np.array(grad(x))
		np.array(der)

		y = x - 1/L * der
		y = np.clip(y, *domain.T)

		u += mu * alpha
		v += alpha * (mu * x - der)
		z = v/u
		z = np.clip(z, *domain.T)
		tau = alpha/A
		x = tau * z + (1 - tau) * y

		alpha = (L+mu*A)/(2*L) * (1 + np.sqrt(abs(1 + (4*L*A)/(L+mu*A))) )

		N += 1
		results.append((x.copy(), time.time()))
		est = 2 * min(4 * L * R/N**2, L * R * np.exp(-N/2 * np.sqrt(mu/L)))
		if est <= eps:
			history[key] = results
			return (x, N, results)

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
	u, v = 2*L, np.zeros(x.shape)
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

		N += 1
		est = min(4 * L * R/N**2, L * R * np.exp(-N/2 * np.sqrt(mu/L)))
		if cond(y, np.sqrt(est/mu)):
			return (x, np.sqrt(est/mu))


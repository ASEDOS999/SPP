#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def cond_for_ellipsoids(f, eps, R):
	return lambda y, R : f.L_yy * R <= eps / 2

def delta_ellipsoid(f, Q, eps = 0.001, history = {}, key = "Ellipsoids"):
	n = len(Q)
	Q = np.array(Q)
	x = (Q[:, 0] + Q[:, 1]) / 2
	R = np.linalg.norm((Q[:, 0] - Q[:, 1]))/ 2
	cond = cond_for_ellipsoids(f, eps, R)
	H = R**2 * np.identity(n)
	domain = np.array(Q)
	grad = lambda x: f.get_delta_grad(x, cond)
	N = 0
	results = [x]
	while True:
		_df = grad(x)
		_df = _df / (np.sqrt(abs(_df@H@_df)))
		x = x - 1/(n+1) * H @ _df
		H = n**2/(n**2 - 1)*(H - (2 / (n + 1)) * (H @ np.outer(_df, _df) @ H))
		N += 1
		results.append((np.clip(x, *domain.T)))
		x = (np.clip(x, *domain.T))
		est = f.L_xx * R * np.exp(- N / (2 * n**2))
		if est <= eps:
			history[key] = results
			return x, N
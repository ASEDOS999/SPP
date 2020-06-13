#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

def cond_for_ellipsoids(f, eps, R):
	return lambda y, R : f.M_y * R <= eps / 2

def get_w(Q, x):
	w = []
	for ind, i in enumerate(x):
		if i > Q[ind, 1]:
			w.append(1)
		elif i < Q[ind, 0]:
			w.append(-1)
		else:
			w.append(0)
	return np.array(w)

def delta_ellipsoid(f, Q, eps = 0.001, history = {}, key = "Ellipsoids", time_max = None, stop_cond = lambda *args: False):
	n = len(Q)
	Q = np.array(Q)
	x = (Q[:, 0] + Q[:, 1]) / 2
	R = np.linalg.norm((Q[:, 0] - Q[:, 1]))/ 2
	cond = cond_for_ellipsoids(f, eps, R)
	H = R**2 * np.identity(n)
	domain = np.array(Q)
	grad = lambda x: f.get_delta_grad(x, cond)
	N = 0
	history[key] = []
	while True:
		if (np.clip(x, *domain.T) == x).any():
			_df, y = grad(x)
			history[key].append(((np.clip(x, *domain.T),y), time.time()))
		else:
			_df = get_w(Q, x)
		_df = _df / (np.sqrt(abs(_df@H@_df)))
		x = x - 1/(n+1) * H @ _df
		H = n**2/(n**2 - 1)*(H - (2 / (n + 1)) * (H @ np.outer(_df, _df) @ H))
		N += 1
		#x = (np.clip(x, *domain.T))
		est = f.L_xx * R * np.exp(- N / (2 * n**2))
		if est <= eps:
			return x, N
		if not time_max is None:
			if history[key][-1][1] - history[key][0][1] >time_max:
				return x, R
		if stop_cond(*history[key][-1][0]):
			return x, R

def get_w_sphere(x, R, c):
	d_xc =np.linalg.norm(x-c) 
	if d_xc <= R:
		return R
	lambda_ = d_xc / R - 1
	y = (x + lambda_ * c) / (lambda_ + 1)
	return y

def ellipsoid(func, 
					   grad,
					   L, mu,
					   start_point,
					   cond):
	x, R = start_point
	x_0 = x.copy()
	n = len(x)
	H = R**2 * np.identity(n)
	N = 0
	cur = None
	while True:
		if np.linalg.norm(x-x_0) <= R:
			_df = grad(x)
		else:
			_df = get_w_sphere(x, R, x_0)
		_df = _df / (np.sqrt(abs(_df@H@_df)))
		x = x - 1/(n+1) * H @ _df
		H = n**2/(n**2 - 1)*(H - (2 / (n + 1)) * (H @ np.outer(_df, _df) @ H))
		#print(H)
		N += 1
		est = L* R * np.exp(- N / (2 * n**2))
		value = func(x)
		#print(x)
		if np.linalg.norm(x-x_0) <= R and (cur is None or cur[1] > func(x)):
			cur = x, value
		if cond(x, np.sqrt(est/mu)):
			#print("Next")
			return (cur[0], np.sqrt(est/mu))
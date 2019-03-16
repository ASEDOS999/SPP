# -*- coding: utf-8 -*-

import math
import numpy as np


def big_grad(f, L, a, b, is_inter, axis, arg):
	if axis == 'x':
		return abs(f.der_y(f.get_sol_hor(segm, arg), arg)) / L
	if axis == 'y':
		return abs(f.der_x(arg, f.get_sol_vert(arg, segm))) / L

def stop_ineq(f, L, is_inter, a, b):
	if is_inter:
		if b - a <= np.linalg.norm(f.gradient) / L:
			return True
		else:
			return False
	else:
		return little_big(f, L, a, b)

def grad_descent_segment(segm, deriv, delta, x_opt):
	N = 0
	x, alpha_0 = (segm[0] + segm[1]) / 2, (segm[0] + segm[1]) / 4
	if delta == 0:
		delta = 0.01 * alpha_0
	if delta < 0:
		return x
	while abs(x - x_opt) > delta and N < 1000:
		x = x - alpha_0 / math.sqrt(N + 1) * deriv(x)
		x = min(max(x, segm[0]), segm[1])
		N += 1
	if N >= 1000:
		N = -1
	return x

def gss(f, segm, est):
	a, b = segm
	gr = (math.sqrt(5) + 1) / 2
	c = b - (b - a) / gr
	d = a + (b - a) / gr 
	f_c, f_d = f(c), f(d)
	while b - a > est:
		if f(c) < f(d):
			b = d
			d, f_d = c, f_c
			c = b - (b- a) / gr
			f_c = f(c)
		else:
			a = c
			c, f_c = d, f_d
			d = a + (b - a) / gr
			f_d = f(d)
	return (b + a) / 2

def add_cond(x, y, f, eps, a):
	if np.linalg.norm(f.gradient(x, y)) <= eps / (a * math.sqrt(2)):
		return True

def halving_square(f, eps, square, solve_hor_segm, solve_vert_segm):
	Q = square.copy()
	N = 0
	minimum = f.min
	x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
	if f.der_x(x_0, y_0) == 0 and f.der_y(x_0, y_0) == 0:
		return ((x_0, y_0), N)
	f_opt = f.calculate_function(x_0, y_0)
	while True:
		x_0 = solve_hor_segm([Q[0], Q[1]], (Q[2]+ Q[3]) / 2)
		if add_cond(x_0, (Q[2] + Q[3])/ 2, f, eps, Q[1] - Q[0]):
			return (x_0, (Q[2] + Q[3]) / 2)
		der = f.der_y(x_0, (Q[2] + Q[3]) / 2)
		if der == 0 and f.der_x(x_0, (Q[2] + Q[3]) / 2) == 0:
			return ((x_0, (Q[2] + Q[3]) / 2), N)
		if der > 0:
			Q[2], Q[3] = Q[2],  (Q[2] + Q[3]) / 2
		else:
			Q[3], Q[2] = Q[3],  (Q[2] + Q[3]) / 2
		
		y_0 = solve_vert_segm([Q[2], Q[3]], (Q[0] + Q[1]) / 2)
		der = f.der_x((Q[0] + Q[1]) / 2, y_0)
		if der == 0 and f.der_y((Q[0] + Q[1]) / 2, y_0) == 0:
			return (((Q[0] + Q[1]) / 2, y_0), N)
		if der > 0:
			Q[0], Q[1] = Q[0],  (Q[0] + Q[1]) / 2
		else:
			Q[1], Q[0] = Q[1],  (Q[0] + Q[1]) / 2
		N += 1
		
		x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
		f_opt = f.calculate_function(x_0, y_0) 
		if N >= 100 or abs(f_opt - minimum) < eps:
			if N >= 1000:
				N = -1
			return ((x_0, y_0), N)

def gradient_descent(f, Q, grad, eps, step, minimum):
	N = 0
	x = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
	x_prev = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
	while (abs(f(x[0], x[1]) - f(x_prev[0], x_prev[1])) > eps and N < 1000) or (N == 0):
		der = grad(x[0], x[1])
		x[0], x_prev[0] = min(max(x[0] - step / math.sqrt(N+1) * der[0], Q[0]), Q[1]), x[0]
		x[1], x_prev[1] = min(max(x[1] - step / math.sqrt(N+1) * der[1], Q[2]), Q[3]), x[1]
		N += 1
	if N >= 1000:
		N = -1
	return (x, N, abs(f(x[0], x[1]) - f(x_prev[0], x_prev[1])))

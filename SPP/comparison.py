# -*- coding: utf-8 -*-

import threading

from Solvers import GradientMethods, Ellipsoids, HalvCube

keys = {"FGM":"FGM",
		"Ellipsoids": "Ellipsoids",
		"Dichotomy": "Dichotomy"}

def get_stop_cond(stop_cond_args, eps, inverse, f):
	alpha_, B, c, beta = stop_cond_args
	def cond(x, y, f_est = 0):
		if inverse:
			if f_est > eps:
				return False
			x, y = y, x
		bad = B @ y - c
		if inverse:
			print(x.dot(B@y-c))
		for ind,i in enumerate(x):
			if i == 0 and bad[ind] > 0:
				return False
		return abs(x.dot(B @ y - c))<= eps
	return cond

def create_methods_dict(f, start_x_fgm, R_fgm, Q, eps, history, time_max = 1, keys = keys, inverse = False, stop_cond_args = None):
	methods = dict()
	if not stop_cond_args is None:
		stop_cond = get_stop_cond(stop_cond_args, eps, inverse = inverse, f = f)
	else:
		stop_cond = lambda *args: False
	if "FGM" in keys:
		fgm = GradientMethods.FGM_external
		methods[keys["FGM"]] = lambda : fgm(f, start_x_fgm, R_fgm, Q, eps = eps, history = history, time_max = time_max,  key = keys["FGM"], stop_cond = stop_cond)
	
	if "Ellipsoids" in keys:
		ellipsoids = Ellipsoids.delta_ellipsoid
		methods[keys["Ellipsoids"]] = lambda: ellipsoids(f, Q, eps = eps, history = history, time_max = time_max,  key = keys["Ellipsoids"], stop_cond = stop_cond)
	
	if "Dichotomy" in keys:
		dichotomy = HalvCube.Dichotomy(history = history, key = keys["Dichotomy"])
		methods[keys["Dichotomy"]] = lambda: dichotomy.Halving(f, Q, eps, time_max = time_max, stop_cond = stop_cond)
	return methods
	
def method_comparison(methods = None):
	t = []
	for key in methods:
		t.append(threading.Thread(target = methods[key], name = key))
	for i in t:
		i.start()
	for i in t:
		i.join()


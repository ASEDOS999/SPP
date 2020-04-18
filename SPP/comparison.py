# -*- coding: utf-8 -*-

import threading

from Solvers import GradientMethods, Ellipsoids, HalvCube

def create_methods_dict(f, start_x_fgm, R_fgm, Q, eps, history, time_max = 1):
	methods = dict()
	fgm = GradientMethods.FGM_external
	methods["FGM"] = lambda : fgm(f, start_x_fgm, R_fgm, Q, eps = eps, history = history, time_max = time_max)
	
	ellipsoids = Ellipsoids.delta_ellipsoid
	methods["Ellipsoids"] = lambda: ellipsoids(f, Q, eps = eps, history = history, time_max = time_max)
	
	dichotomy = HalvCube.Dichotomy(history = history)
	methods["Dichotomy"] = lambda: dichotomy.Halving(f, Q, eps, time_max = time_max)
	return methods
	
def method_comparison(methods = None):
	t = []
	for key in methods:
		t.append(threading.Thread(target = methods[key], name = key))
	for i in t:
		i.start()
	for i in t:
		i.join()


import sys

sys.path.append('..')
sys.path.append('../Tests_functions')

from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
from test_functions import quadratic_function as qf
from method_functions import halving_square, gss
import math

#Quadratic functions
results = []
epsilon = [0.1**(i) for i in range(10)]
num = 0
N = 1000
n = 0
full_results = []
for eps in epsilon:
	for j in range(N):
		param = np.random.uniform(-10, 10, 6)
		param[2] =  abs(param[2])
		f = qf(param)
		size_1, size_2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
		x_1, y_1 = f.solution[0], f.solution[1]
		Q = [x_1 - (1-size_1), x_1 + (1+size_1), y_1 - (1-size_2), y_1 + (1+size_2)]
		R = 2
		L = 3 * max(abs(param[0]), abs(param[1]), param[2]) *  max(abs(Q[1]), abs(Q[3]))
		M = (2 * param[0]**2 + 4 * abs(param[0] * param[1]) + 2 * param[1] ** 2 + 2 * param[2]) * 10
		m1 = time()
		est = eps / (2 * M * R * (math.sqrt(2) + math.sqrt(5)) * ( - math.log(eps / (L * R * math.sqrt(2)), 2)))
		res = halving_square(f, eps, Q, 
			lambda segm, y: gss(lambda x: f.calculate_function(x,y), segm, abs(f.der_y(f.get_sol_hor(segm, y), y)) / L),
			lambda segm, x: gss(lambda y: f.calculate_function(x,y), segm, abs(f.der_x(f.get_sol_vert(segm,x), x)) / L))
		m2 = time()
		est = eps / (2 * M * R * (math.sqrt(2) + math.sqrt(5)) * ( - math.log(eps / (L * R * math.sqrt(2)), 2)))
		res_2 = halving_square(f, eps, Q, 
			lambda segm, y: gss(lambda x: f.calculate_function(x,y), segm, est),
			lambda segm, x: gss(lambda y: f.calculate_function(x,y), segm, est))
		m3 = time()
		results.append((eps, res[1], m2-m1, res_2[1], m3-m2))
	list_gss = [i[2] for i in results]
	print('eps = ', eps)
	print('Mean time (Little Big) = %.2fms'%(1000 * np.mean([i[2] for i in results])))
	print('Mean time (Constant) = %.2fms'%(1000 * np.mean([i[4] for i in results])))
	full_results = full_results + results
	results = []


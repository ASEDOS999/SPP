# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
sys.path.append("../Tests_functions")

from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
from test_functions import sinuses
from test_functions import quadratic_function as qf
from method_functions import halving_square, gradient_descent
from method_functions import gss
from method_functions import grad_descent_segment as GD_s
import math

#Quadratic functions
results = []
epsilon = [0.1**(i) for i in range(3)]
num = 0
N = 10
n = 0
while len(results) < 3 * N:
		n += 1
		if n % 100 == 0:
			print(n / 10,'%')
		param = np.random.uniform(-10, 10, 6)
		param[2] =  abs(param[2])
		f = qf(param)
		size_1, size_2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
		x_1, y_1 = f.solution[0], f.solution[1]
		Q = [x_1 - (1-size_1), x_1 + (1+size_1), y_1 - (1-size_2), y_1 + (1+size_2)]
		R = 2
		L = 3 * max(abs(param[0]), abs(param[1]), param[2]) *  max(abs(Q[1]), abs(Q[3]))
		M = (2 * param[0]**2 + 4 * abs(param[0] * param[1]) + 2 * param[1] ** 2 + 2 * param[2]) * 10
		for eps in epsilon:
			m1 = time()
			est = eps / (2 * M * R * (math.sqrt(2) + math.sqrt(5)) * ( - math.log(eps / (L * R * math.sqrt(2)), 2)))
			res_1 = halving_square(f, eps, Q, 
				lambda segm, y: gss(lambda x: f.calculate_function(x,y), segm, est),
				lambda segm, x: gss(lambda y: f.calculate_function(x,y), segm, est))
			m2 = time()
			res_2 = halving_square(f, eps, Q, 
				lambda segm, y: GD_s(segm, lambda x: f.der_x(x, y), est, f.get_sol_hor(segm, y)),
				lambda segm, x: GD_s(segm, lambda y: f.der_y(x, y), est, f.get_sol_vert(segm, x)))
			m3 = time()
			results.append((eps, res_1[1], res_2[1], m2 - m1, m3 - m2))

list_gss = [i[3] for i in results]
list_grad = [i[4] for i in results]
print('GSS\n', 'Mean time = ', np.mean(list_gss))
print('Gradient Descent\n', 'Mean time = ', np.mean(list_grad))

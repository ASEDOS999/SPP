# -*- coding: utf-8 -*-

import sys
sys.path.append("/Tests_functions")

from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
from test_functions import sinuses
from test_functions import LogSumExp
from test_functions import quadratic_function as qf
from method_functions import main_solver
from method_functions import gradient_descent
from method_functions import ellipsoid
import math

def qf_comparison(epsilon = 1e-6, out = True):
	param = np.random.uniform(-100, 100, 6)
	param[2] =  abs(param[2])
	f = qf(param)
	size_1, size_2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
	x_1, y_1 = f.solution[0], f.solution[1]
	Q = [x_1 - (1-size_1), x_1 + (1+size_1), y_1 - (1-size_2), y_1 + (1+size_2)]
	results = comparison(f, Q, epsilon)
	if out:
		norm_gradient = lambda x: f.calculate_function(x[0], x[1]) - f.min
		plt.semilogy([i for i in range(len(results[0]))], [norm_gradient(i) for i in results[0]])
		plt.semilogy([i for i in range(len(results[2]))], [norm_gradient(i) for i in results[2]])
		plt.semilogy([i for i in range(len(results[4]))], [norm_gradient(i) for i in results[4]])
		plt.xlabel('Number of iterations')
		plt.ylabel(r'$f(x)-f^*$')
		plt.xticks([i for i in range(0, 101, 10)])
		plt.legend(['Gradiend Descent', 'Halving Square', 'Ellipsoid'])
		print('Gradiend Descent %.4f'%(results[1]))
		print('Halving Square %.4f'%(results[3]))
		print('Ellipsoid %.4f'%(results[5]))
	return results

def qf_test(eps):
	param = np.random.uniform(-100, 100, 6)
	param[2] =  abs(param[2])
	f = qf(param)
	a = 1
	size_1, size_2 = random.uniform(0.5 *a, a), random.uniform(0.5*a, a)
	x_1, y_1 = f.solution[0], f.solution[1]
	Q = [x_1 - (a-size_1), x_1 + (a+size_1), y_1 - (a-size_2), y_1 + (a+size_2)]
	f.lipschitz_function(Q), f.lipschitz_gradient(Q)
	solver = main_solver(f, Q, eps)
	solver.init_help_function()
	res = solver.halving_square()[2]
	norm_gradient = lambda x: f.calculate_function(x[0], x[1]) - f.min
	plt.semilogy([i for i in range(len(res))], [norm_gradient(i) for i in res])
	plt.semilogy([i for i in range(len(res))], [(i[0] -x_1)**2 + (i[1]-y_1)**2 for i in res])
	plt.grid()
	plt.legend([r"$f(x)-f^*$", r"$\|x-x^*\|$"])
	q_2 = 1./2 * np.log(f.M * a**2/(4 * eps)) / np.log(2)
	q_1 = np.log(f.L * a/(np.sqrt(2) * eps)) / np.log(2)
	plt.xlabel("Iterations Number")
	plt.ylabel("Value of Error")
	print('Theoretical Iteration Number through function constant', np.ceil(q_1))
	print('Theoretical Iteration Number through gradient constant', np.ceil(q_2))
	return res

def qf_test_2(eps):
	param = [1, 1, 1, 0, 0, 0]
	f = qf(param)
	a = 1
	size_1, size_2 = random.uniform(0.5 *a, a), random.uniform(0.5*a, a)
	x_1, y_1 = 1, 1
	f.solution[0], f.solution[1] = x_1, y_1
	f.min = 5
	Q = [1, 2, 1, 2]
	f.lipschitz_function(Q), f.lispchitz_gradient(Q)
	solver = main_solver(f, Q, eps)
	solver.init_help_function()
	res = solver.halving_square()[2]
	norm_gradient = lambda x: f.calculate_function(x[0], x[1]) - f.min
	plt.semilogy([i for i in range(len(res))], [norm_gradient(i) for i in res])
	plt.semilogy([i for i in range(len(res))], [(i[0] -x_1)**2 + (i[1]-y_1)**2 for i in res])
	plt.grid()
	plt.xlabel("Iterations Number")
	plt.ylabel("Value of Error")
	plt.legend([r"$f(x)-f^*$", r"$\|x-x^*\|$"])
	q_2 = 1./2 * np.log(f.M * a**2/(4 * eps)) / np.log(2)
	q_1 = np.log(f.L * a/(np.sqrt(2) * eps)) / np.log(2)
	print('Theoretical Iteration Number through function constant', np.ceil(q_1))
	print('Theoretical Iteration Number through gradient constant', np.ceil(q_2))
	return res

def comparison(f, Q, eps):
	n = 0
	m1 = time()
	res_1 = gradient_descent(f.calculate_function, Q, f.gradient, f.lipschitz_gradient(Q) ,eps, f.min)
	m2 = time()
	solver = main_solver(f, Q, eps)
	solver.init_help_function()
	res_2 = solver.halving_square()
	m3 = time()
	res_3 = ellipsoid(f, Q, eps = eps)
	m4 = time()
	return res_1[2], m2-m1, res_2[2], m3-m2, res_3[2], m4-m3

def comparison_LogSumExp(N = 2, time_max = 100):
	a = np.random.uniform(-100, 100, size=(N,))
	f = LogSumExp(a)
	Q = f.get_square()
	L, M = f.lipschitz_function(Q), f.lipschitz_gradient(Q)
	solver = main_solver(f, Q, eps = None)
	res = dict()
	print('Ellipsoids')
	res['Ellipsoids'] = ellipsoid(f,Q, time_max = time_max, time = True)

	print('Const_est')
	solver.init_help_function('const_est')
	res['HalvingSquare-Const']= solver.halving_square(eps = 1e-2)
	print('CurGrad')
	solver.init_help_function()
	res['HalvingSquare-CurGrad']= solver.halving_square(time_max = time_max, time = True)
	print('GD')
	res['GD'] = gradient_descent(f.calculate_function, Q, f.gradient, M, time= True, time_max = time_max)
	return res, f
if __name__ == "__main__":
	eps = [0.1**(i) for i in range(7)]
	num_iter_tests(eps)
	comparison_GD_HS_sinuses(eps)
	comparison_GD_HS_QFunc(eps)

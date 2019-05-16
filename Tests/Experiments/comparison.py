# -*- coding: utf-8 -*-

import sys
sys.path.append("/Tests_functions")

from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
from test_functions import sinuses
from test_functions import LSM_exp
from test_functions import quadratic_function as qf
from method_functions import main_solver
from method_functions import gradient_descent
from method_functions import ellipsoid
import math

#Tests for iterations number
def num_iter_tests(epsilon):
	results = []
	num = 0
	Q = [0, 1, 0, 1]
	for i in np.linspace(1.1, 1.9, 5).tolist():
		for j in np.linspace(1.1, 1.9, 5).tolist():
			a = [[0.1, 0.1], [0.1, 0.1, 0.1]]
			m, n = 1, 2
			while m != -1:
				f = sinuses(a, [i, j])
				for eps in epsilon:
					solver = main_solver(f, Q, eps)
					solver.init_help_function(stop_func = 'true')
					N = solver.halving_square()
					results.append((N[1], eps, f.L))
				m, n = 1, 2
				while m != -1 and a[m][n] == 1:
					a[m][n] = 0.1
					n = n - 1
					if n < 0:
						m = m - 1
						if m > 0:
							n = 2
						else:
							n = 1
				a[max(m, 0)][n] *= 10
			num += 1
	plt.plot([math.log(i[2] / i[1] / math.sqrt(2), 2) for i in results], [i[0] for i in results], 'ro')
	plt.plot([0, 17], [0, 17], 'b')
	plt.title("Iterations number")
	plt.grid()
	plt.legend(('Tests functions', r'Line $N = \log \frac{La}{\sqrt{2}\epsilon}$'))
	plt.ylabel(r'Iterations Number')
	plt.xlabel(r'$\log \frac{La}{\sqrt{2}\epsilon}$')
	plt.show()

#Сompetion: Gradient Descent vs New method
#Sinuses
def comparison_GD_HS_sinuses(epsilon):
	results = []
	num = 0
	Q = [0, 1, 0, 1]
	for i in np.linspace(1.1, 1.9, 5).tolist():
		for j in np.linspace(1.1, 1.9, 5).tolist():
			a = [[0.1, 0.1], [0.1, 0.1, 0.1]]
			m, n = 1, 2
			while m != -1:
				f = sinuses(a, [i, j])
				for eps in epsilon:
					m1 = time()
					res_1 = gradient_descent(f.calculate_function, [0, 1, 0, 1], f.gradient, eps, 0.25, f.min)
					m2 = time()
					solver = main_solver(f, Q, eps)
					solver.init_help_function(stop_func = 'true')
					res_2 = solver.halving_square()
					m3 = time()
					results.append((eps, res_1[1], res_2[1], m2 - m1, m3 - m2))
				m, n = 1, 2
				while m != -1 and a[m][n] == 1:
					a[m][n] = 0.1
					n = n - 1
					if n < 0:
						m = m - 1
						if m > 0:
							n = 2
						else:
							n = 1
				a[max(m, 0)][n] *= 10
			num += 1
	
	p = 1.05
	
	data = (len([i[0] for i in results
			 	if i[1] >= 0 and i[2] < 0]),
			len([i[0] for i in results
			 	if i[1] >= 0 and i[2] >= 0 and i[4] > p * i[3]]),
			len([i[0] for i in results
			 	if i[1] >= 0 and i[2] >= 0 and (1/p * i[3]) <= i[4] and i[4] <= (p * i[3])]),
			len([i[0] for i in results
			 	if i[1] >= 0 and i[2] >= 0 and i[4] < 1/p * i[3]]),
			len([i[0] for i in results
			 	if i[1] < 0 and i[2] >= 0]))
	ind = np.arange(5)
	width = 0.35
	p1 = plt.bar(ind, data, width)
	plt.grid()
	plt.ylabel('Number of tasks')
	plt.xticks(ind, ('T1', 'T2', 'T3', 'T4', 'T5'))
	plt.show()


#Quadratic functions
def comparison_GD_HS_QFunc(epsilon):
	results = []
	num = 0
	N = 1000
	n = 0
	while len(results) < 3 * N:
			n += 1
			param = np.random.uniform(-10, 10, 6)
			param[2] =  abs(param[2])
			f = qf(param)
			size_1, size_2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
			x_1, y_1 = f.solution[0], f.solution[1]
			Q = [x_1 - (1-size_1), x_1 + (1+size_1), y_1 - (1-size_2), y_1 + (1+size_2)]
			for eps in epsilon:
				m1 = time()
				res_1 = gradient_descent(f.calculate_function, Q, f.gradient, eps, 0.25, f.min)
				m2 = time()
				solver = main_solver(f, Q, eps)
				solver.init_help_function()
				res_2 = solver.halving_square()
				m3 = time()
				results.append((eps, res_1[1], res_2[1], m2 - m1, m3 - m2))
	
	p = 1.05
	
	data = (len([i[0] for i in results
			 	if i[1] >= 0 and i[2] < 0]),
			len([i[0] for i in results
			 	if i[1] >= 0 and i[2] >= 0 and i[4] > p * i[3]]),
			len([i[0] for i in results
			 	if i[1] >= 0 and i[2] >= 0 and (1/p * i[3]) <= i[4] and i[4] <= (p * i[3])]),
			len([i[0] for i in results
			 	if i[1] >= 0 and i[2] >= 0 and i[4] < 1/p * i[3]]),
			len([i[0] for i in results
			 	if i[1] < 0 and i[2] >= 0]),
			len([i[0] for i in results
			 	if i[1] < 0 and i[2] < 0]))
	ind = np.arange(6)
	width = 0.35
	p1 = plt.bar(ind, data, width)
	plt.title('QF')
	plt.grid()
	plt.ylabel('Number of tasks')
	plt.xticks(ind, ('T1', 'T2', 'T3', 'T4', 'T5', 'T6'))
	plt.show()

def qf_comparison(epsilon = 1e-6, out = True):
	param = np.random.uniform(-10, 10, 6)
	param[2] =  abs(param[2])
	f = qf(param)
	size_1, size_2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
	x_1, y_1 = f.solution[0], f.solution[1]
	Q = [x_1 - (1-size_1), x_1 + (1+size_1), y_1 - (1-size_2), y_1 + (1+size_2)]
	results = comparison(f, Q, epsilon)
	if out:
		#norm_gradient = lambda x: np.linalg.norm(f.gradient(x[0], x[1]))
		norm_gradient = lambda x: f.calculate_function(x[0], x[1]) - f.min
		plt.semilogy([i for i in range(len(results[0]))], [norm_gradient(i) for i in results[0]])
		plt.semilogy([i for i in range(len(results[2]))], [norm_gradient(i) for i in results[2]])
		plt.semilogy([i for i in range(len(results[4]))], [norm_gradient(i) for i in results[4]])
		plt.xlabel('Number of iterations')
		plt.ylabel('Norm of gradient')
		plt.xticks([i for i in range(0, 101, 10)])
		plt.legend(['Gradiend Descent', 'Halving Square', 'Ellipsoid'])
		print('Gradiend Descent %.4f'%(results[1]))
		print('Halving Square %.4f'%(results[3]))
		print('Ellipsoid %.4f'%(results[5]))
	return results

def LSM_comparison(epsilon = 1e-6, out = True):
	a = random.uniform(4, 8)
	b = random.uniform(-2, 2)
	f = LSM_exp(a, b, 10)
	Q = [4, 8, -2, 2]
	results = comparison(f, Q, epsilon)
	if out:
		norm_gradient = lambda x: f.calculate_function(x[0], x[1]) - f.min
		plt.semilogy([i for i in range(len(results[0]))], [norm_gradient(i) for i in results[0]])
		plt.semilogy([i for i in range(len(results[2]))], [norm_gradient(i) for i in results[2]])
		plt.semilogy([i for i in range(len(results[4]))], [norm_gradient(i) for i in results[4]])
		plt.xlabel('Number of iterations')
		plt.ylabel('Norm of gradient')
		plt.xticks([i for i in range(0, 101, 10)])
		plt.grid()
		plt.legend(['Gradiend Descent', 'Halving Square', 'Ellipsoid'])
		print('Gradiend Descent %.4f'%(results[1]))
		print('Halving Square %.4f'%(results[3]), results[2][-1], a, b)
		print('Ellipsoid %.4f'%(results[5]), results[4][-1])
	return results

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

if __name__ == "__main__":
	eps = [0.1**(i) for i in range(7)]
	num_iter_tests(eps)
	comparison_GD_HS_sinuses(eps)
	comparison_GD_HS_QFunc(eps)

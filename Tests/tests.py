# -*- coding: utf-8 -*-

from time import time
import matplotlib.pyplot as plt
import numpy as np
from test_functions import sinuses
from method_functions import method, gradient_descent, halving_square
import math

#Tests for iterations number
results = []
epsilon = [0.1**(1 + i) for i in range(7)] + [0.5**(1 + i) for i in range(7)]
num = 0
for i in np.linspace(1.1, 1.9, 5).tolist():
    for j in np.linspace(1.1, 1.9, 5).tolist():
        print((num * 4), '% is completed')
        a = [[0.1, 0.1], [0.1, 0.1, 0.1]]
        m, n = 1, 2
        while m != -1:
            f = sinuses(a, [i, j])
            for eps in epsilon:
                N = method(f, eps, [0, 1, 0, 1])
                results.append((N[1], eps, N[2], f.L))
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
print('100 % is completed')
plt.plot([math.log(i[3] / i[1] / math.sqrt(2), 2) for i in results], [i[0] for i in results], 'ro')
plt.plot([0, 17], [0, 17], 'b')
plt.legend(('Tests functions', r'Line $N = \log \frac{La}{\sqrt{2}\epsilon}$'))
plt.ylabel(r'$N$')
plt.xlabel(r'$\log \frac{La}{\sqrt{2}\epsilon}$')
plt.show()

#Сompetion: Gradient Descent vs New method
#Sinuses
results = []
epsilon = [0.1**(3 + i) for i in range(7)]
num = 0
for i in np.linspace(1.1, 1.9, 5).tolist():
    for j in np.linspace(1.1, 1.9, 5).tolist():
        print((num * 4), '% is completed')
        a = [[0.1, 0.1], [0.1, 0.1, 0.1]]
        m, n = 1, 2
        while m != -1:
            f = sinuses(a, [i, j])
            for eps in epsilon:
                m1 = time()
                res_1 = gradient_descent(f.calculate_function, [0, 1, 0, 1], f.gradient, eps, 0.25)
                m2 = time()
                res_2 = halving_square(f, eps, [0, 1, 0, 1])
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
print('100 % is completed')

N = 5
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
ind = np.arange(N)
width = 0.35
p1 = plt.bar(ind, data, width)
plt.ylabel('Number of tasks')
plt.xticks(ind, ('T1', 'T2', 'T3', 'T4', 'T5'))
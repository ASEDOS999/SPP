# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from method_functions import method
from test_functions import sinuses
import math

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
plt.plot([math.log(i[3] / i[1] / math.sqrt(2), 2) for i in results if i[2] == 0], [i[0] for i in results if i[2] == 0], 'ro')
plt.plot([math.log(i[3] / i[1] / math.sqrt(2), 2) for i in results if i[2] != 0], [i[0] for i in results if i[2] != 0], 'g+')
plt.plot([0, 17], [0, 17], 'b')
plt.legend(('Type 1', 'Type_2', r'Line $N = \log \frac{La}{\sqrt{2}\epsilon}$'))
plt.ylabel(r'$N$')
plt.xlabel(r'$\log \frac{La}{\sqrt{2}\epsilon}$')
plt.show()

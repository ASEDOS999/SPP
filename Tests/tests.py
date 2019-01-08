# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from method_functions import method
from test_functions import sinuses

results = []
epsilon = [0.1**(1 + i) for i in range(5)]
num = 0
for i in np.linspace(1.1, 1.9, 5).tolist():
    for j in np.linspace(1.1, 1.9, 5).tolist():
        a = [[1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
        m, n = 2, 2
        while m != 0 or n != 0:
            for eps in epsilon:
                print(num * 100.0 / 3**7 / 25 / 5,'%' )
                f = sinuses(a, [i, j])
                N = method(f, 0.1**10, [0, 1, 0, 1])
                results.append((N[1], eps, N[2]))
                num += 1
            m, n = 2, 2
            while a[m][n] == 1:
                a[m][n] = 0.1
                n = n - 1
                if n < 0:
                    if m - 1 == 0:
                        m = m - 1
                        n = 1
                    else:
                        m = m - 1
                        n = 2
            a[m][n] = a[m][n] * 10
print(results)
# -*- coding: utf-8 -*-

from math import pi, sqrt, cos
import numpy as np

class sinuses():
    def __init__(self, a, b):
        self.parameters = a
        self.coef = b
        self.min = self.calculate_function(b[0]/2, b[1]/2)
        self.min_x = b[0]/2
        self.min_y = b[1]/2
        s = a[0][1] + a[0][0]
        for i in range(1, len(a)):
            s += (i+1) * (a[i][0] + a[i][1]) * a[i][2]**i
        self.L = s
    def calculate_function(self, x, y):
        a = self.parameters
        b = self.coef
        z = -a[0][0] * np.sin(x * pi / b[0]) - a[0][1] * np.sin(y * pi / b[1])
        for i in range(1, len(a)):
            z +=  (-a[i][0] * np.sin(x * pi / b[0]) - a[i][1] * np.sin(y * pi / b[1]) + a[i][0] + a[i][1] + a[i][2])**(i+1)
        return z
    
    def der_x(self, x, y):
        a = self.parameters
        b = self.coef
        der = -a[0][0]
        for i in range(1, len(a)):
            der += (-(i+1) * a[i][0] * (-a[i][0] * np.sin(x * pi / b[0]) -
            a[i][1] * np.sin(y * pi / b[1]) + a[i][0] + a[i][1] + a[i][2])**i)
        return der * cos(x * pi / b[0]) * pi / b[0]

    def min_der_x(self, x):
        a = self.parameters
        b = self.coef
        der = a[0][0]
        for i in range(1, len(a)):
            der += (i+1) * a[i][0] * a[i][2]**i
        return der * abs(cos(x * pi / b[0])) * pi / b[0]
    
    def der_y(self, x, y):
        a = self.parameters
        b = self.coef
        der = -a[0][1]
        for i in range(1, len(a)):
            der += (-(i+1) * a[i][1] * (-a[i][0] * np.sin(x * pi / b[0]) -
            a[i][1] * np.sin(y * pi/b[1]) + a[i][0] + a[i][1] + a[i][2])**i)
        return der * cos(y * pi / b[1]) * pi / b[1]

    def min_der_y(self, y):
        a = self.parameters
        b = self.coef
        der = a[0][1]
        for i in range(1, len(a)):
            der += (i+1) * a[i][1] * a[i][2]**i
        return der * abs(cos(y * pi / b[1])) * pi / b[1]

    def max_der_xy(self):
        a = self.parameters
        b = self.coef
        der = 0
        for i in range(1, len(a)):
            der += i * (i+1) * (a[i][0] + a[i][1]+ a[i][2])**(i-1)
        return der * pi**2 / (b[0] * b[1])
    
    def get_est(self, x, num):
        if num == 0:
            est = self.min_der_x(x)
        else:
            est = self.min_der_y(x)
        return est / self.max_der_xy()
    
class almost_polinomial():
    def __init__(self, a, Q):
        self.min = 0
        self.coef = a
        self.Q = Q
        self.min_x = 0
        self.min_y = 0
        self.L_grad = (3 * (a[0]+a[1]) + 
                       2 * (abs(a[2]) + abs(a[3]))*(abs(a[2])*max(abs(Q[0]), abs(Q[1])) + abs(a[3])*max(abs(Q[3]), abs(Q[2]))))
    
    def psi(self, x):
        if x < 0:
            return 3
        else:
            return 2
    
    def calculate_function(self, x_ar, y_ar):
        ans = np.zeros(np.shape(x_ar))
        for i in range(np.shape(x_ar)[0]):
            for j in range(np.shape(x_ar)[1]):
                x, y = x_ar[i][j], y_ar[i][j]
                ans[i][j] = (self.coef[0] * self.psi(x) * x**2 / 2 + self.coef[0] * self.psi(y) * y**2 / 2 +
                    (self.coef[2] * x + self.coef[3] * y)**2)
        return ans
    def der_x(self, x, y):
        return self.psi(x) * x + 2 * self.coef[2] * (self.coef[2] * x + self.coef[3] * y)
    
    def der_y(self, x, y):
        return self.psi(y) * y + 2 * self.coef[3] * (self.coef[2] * x + self.coef[3] * y)
    
    def min_der_x(self, x):
        flag = - (self.psi(x) + self.coef[2]) * x / (2 * self.coef[2] * self.coef[3])
        if flag >= self.Q[2] and flag <= self.Q[3]:
            return 0
        flag = ((self.psi(x) + self.coef[2]) * x + 
                2 * self.coef[2] * self.coef[3] * self.Q[3])
        if flag < 0:
            return flag
        else:
            return ((self.psi(x) + self.coef[2]) * x + 
                    2 * self.coef[2] * self.coef[3] * self.Q[2])
            
    def min_der_y(self, y):
        flag = - (self.psi(y) + self.coef[3]) * y / (2 * self.coef[2] * self.coef[3])
        if flag >= self.Q[0] and flag <= self.Q[1]:
            return 0
        flag = ((self.psi(y) + self.coef[3]) * y + 
                2 * self.coef[2] * self.coef[3] * self.Q[1])
        if flag < 0:
            return flag
        else:
            return ((self.psi(y) + self.coef[3]) * y + 
                    2 * self.coef[2] * self.coef[3] * self.Q[0])
            
    def get_est(self, x, num):
        if num == 0:
            est = self.min_der_x(x)
        else:
            est = self.min_der_y(x)
        return est / self.L_grad
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
            der += -(i+1) * a[i][0] * (-a[i][0] * np.sin(x * pi / b[0]) - a[i][1] * np.sin(y * pi / b[1]) + a[i][0] + a[i][1] + a[i][2])**i
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
            der += -(i+1) * a[i][1] * (-a[i][0] * np.sin(x * pi / sqrt(2)) - a[i][1] * np.sin(y * pi / sqrt(2)) + a[i][0] + a[i][1] + a[i][2])**i
        return der * cos(y * pi / b[1]) * pi / b[1]

    def min_der_y(self, y):
        a = self.parameters
        b = self.coef
        der = a[0][1]
        for i in range(1, len(a)):
            der += (i+1) * a[i][1] * a[i][2]**i
        return der * abs(cos(y * pi / b[1])) * pi / b[1]

    def max_der_xy(self, x, num):
        a = self.parameters
        b = self.coef
        der = 0
        for i in range(1, len(a)):
            der += i * (i+1) * (a[i][0] + a[i][1]+ a[i][2])**(i-1)
        return der * pi**2 / (b[0] * b[1])
    
    def gradient(self, x, y):
        return (self.derivative_x(x, y), self.derivative_y(x, y))
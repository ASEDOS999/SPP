# -*- coding: utf-8 -*-

from math import pi, sqrt, cos
import numpy as np

class sinuses():
    def __init__(self, a):
        self.parameters = a
        self.min = - (a[0][0] + a[0][1])
        
    def calculate_function(self, x, y):
        a = self.parameters
        z = -a[0][0] * np.sin(x * pi / sqrt(2)) - a[0][1] * np.sin(y * pi / sqrt(2))
        for i in range(1, len(a)):
            z +=  (-a[i][0] * np.sin(x * pi / sqrt(2)) - a[i][1] * np.sin(y * pi / sqrt(2)) + a[i][0] + a[i][1] + a[i][2])**(i+1)
        return z
    
    def derivative_x(self, x, y):
        a = self.parameters
        der = -a[0][0]
        for i in range(1, len(a)):
            der += -(i+1) * a[i][0] * (-a[i][0] * np.sin(x * pi / sqrt(2)) - a[i][1] * np.sin(y * pi / sqrt(2)) + a[i][0] + a[i][1] + a[i][2])**i
        return der * cos(x * pi / sqrt(2)) * pi / sqrt(2)
        
    def derivative_y(self, x, y):
        a = self.parameters
        der = -a[0][1]
        for i in range(1, len(a)):
            der += -(i+1) * a[i][1] * (-a[i][0] * np.sin(x * pi / sqrt(2)) - a[i][1] * np.sin(y * pi / sqrt(2)) + a[i][0] + a[i][1] + a[i][2])**i
        return der * cos(y * pi / sqrt(2)) * pi / sqrt(2)
    
    def derivarive_xy(self, x, y):
        a = self.parameters
        der = 0
        for i in range(1, len(a)):
            der += i * (i+1) * (-a[i][0] * np.sin(x * pi / sqrt(2)) - a[i][1] * np.sin(y * pi / sqrt(2)) + a[i][0] + a[i][1]+ a[i][2])**(i-1)
        return der * pi**2 / 2 * cos(y * pi / sqrt(2)) * cos(y * pi / sqrt(2))
    
    def gradient(self, x, y):
        return (self.derivative_x(x, y), self.derivative_y(x, y))
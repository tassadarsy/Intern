
# coding: utf-8

from numpy import *
from pandas import *
from pylab import *
import numpy as np
import pandas as pd
import random

test = []
with open('/Users/Tass/Desktop/Data/test') as f: # Read data
    for line in f.readlines():
        data = line.split()
        test.append(data)

for i in range(0,len(test)):
    for j in range(0,len(test[0])):
        test[i][j] = float(test[i][j])

def empty_array(num): # create an empty array with length num
    l = []
    for i in range(num):
        l.append([])
    return l

def BallTree(x, n): # input data x and tree height n
    m = 0
    l = empty_array(2**n - 1); l[0] = x;
    lb = zeros(len(x[0]))
    ub = zeros(len(x[0]))
    C = zeros(len(x[0]))
    for i in range(1,len(x[0])):
        lb[i] = max([y[i] for y in x])
        ub[i] = min([y[i] for y in x])
        C[i - 1] = random.uniform(lb[i], ub[i]) # the initial random point
    s = empty_array(2**n - 1); s[0] = C[0:128];
    
    while m < n - 1:
        
        for p in range(2**m - 1,2**(m + 1) - 1):
            
    # find p1
            y = zeros(len(l[p]))
            for i in range(0,len(l[p])):
                k = 0
                for j in range(1,len(l[p][0])): # range(1:129)
                    d = (l[p][i][j] - s[p][j - 1])**2 # 
                    k = k + d
                    y[i] = sqrt(k);
            maxy = max(y)
            for i in range(0,len(l[p])):
                if y[i] == maxy:
                    p1 = i
   
    # find p2
            z = zeros(len(l[p]))
            for i in range(0,len(l[p])):
                k = 0
                for j in range(1,len(l[p][0])):
                    d = (l[p][i][j] - l[p][p1][j])**2
                    k = k + d
                    z[i] = sqrt(k)
            maxz = max(z)
            for i in range(0,len(l[p])):
                if z[i] == maxz:
                    p2 = i
        
    # dicide points belong to which cluster
            g1 = zeros(len(l[p]))
            g2 = zeros(len(l[p]))
            left = []
            right = []
            for i in range(0,len(l[p])):
                k1 = 0
                k2 = 0
                for j in range(1,len(l[p][0])):
                    d1 = (l[p][i][j] - l[p][p1][j])**2 
                    # distance between points and p1 in jth dim
                    k1 = k1 + d1
                    
                    d2 = (l[p][i][j] - l[p][p2][j])**2
                    # distance between points and p2 in jth dim
                    k2 = k2 + d2
                g1[i] = sqrt(k1)
                g2[i] = sqrt(k2)
                    
                if g1[i] < g2[i]: # closer to g1
                    left.append(l[p][i]) # p-th array in l and i-th row in p
                else:right.append(l[p][i])
        
            l[2*p + 1] = left
            l[2*p + 2] = right
            
    # re-calculate centers
            
            temp1 = mat(l[2*p + 1])
            t1 = np.mean(temp1[:,1:129], axis = 0)
            s[2*p + 1] = array(t1[0,:])[0]
            
            temp2 = mat(l[2*p + 2])
            t2 = np.mean(temp2[:,1:129], axis = 0)
            s[2*p + 2] = array(t2[0,:])[0]
    
        output = empty_array(2**n - 1)
        for i in range(0,len(l)):
            sth = []
            for j in range(0,len(l[i])):
                sth.append(l[i][j][0])
                output[i] = sth
        
        m = m + 1
   
    return output


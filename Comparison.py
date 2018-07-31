
# coding: utf-8

from numpy import *
from pylab import *
import numpy as np
import random
import time

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

start = time.clock()

d = zeros(len(test)) # create an empty array with zeros
u = int(random.uniform(0,len(test))) # generate a random number btw
for i in range(0,len(test)): # calculate the distance between u and others
    k = 0
    if i != u:
        for j in range(1,len(test[i])):
            l = (test[i][j] - test[u][j])**2
            k = k + l
        d[i] = sqrt(k)
mind = sort(d)[0:201] # extract the minimum 200 distances

labels1 = []
bruceforce = []
for i in range(0,200):
    for j in range(0,len(d)):
        if mind[i] == d[j]: # extract the index of the minimum 200 distances
            labels1.append(test[j][0])
    bruceforce.append((labels1[i],mind[i]))

end = time.clock()
timebf = end - start
print('Running time: %s Seconds'%(timebf))

def BallTree(x, n): # input data x and tree height n
    m = 0 # set initial iteration times
    l = empty_array(2**n - 1); l[0] = x; # binary tree labels and features
    lb = zeros(len(x[0]))
    ub = zeros(len(x[0]))
    C = zeros(len(x[0]))
    for i in range(1,len(x[0])): # only features
        lb[i] = max([y[i] for y in x]) # lower bound of the initial random point
        ub[i] = min([y[i] for y in x]) # upper bound of the initial random point
        C[i - 1] = random.uniform(lb[i], ub[i]) # the initial random point
    s = empty_array(2**n - 1); s[0] = C[0:128]; # binary tree nodes
    
    while m < n - 1: # when m = 0, n = 1, hence stop when m = n - 1
        
        for p in range(2**m - 1,2**(m + 1) - 1):  # nodes values in the same tier
            
    # find p1
            y = zeros(len(l[p])) # for a specfic node, create an empty array
            for i in range(0,len(l[p])):
                k = 0
                for j in range(1,len(l[p][0])): # range(1:129)
                    d = (l[p][i][j] - s[p][j - 1])**2 # distance in one dim
                    k = k + d # cumulative distance^2
                y[i] = sqrt(k); # cumulative distance
            maxy = max(y)  # the largest distance
            for i in range(0,len(l[p])):
                if y[i] == maxy:
                    p1 = i # the index of the largest distance point
   
    # find p2
            z = zeros(len(l[p])) # for another specfic node
            for i in range(0,len(l[p])):
                k = 0
                for j in range(1,len(l[p][0])): # range(1:129)
                    d = (l[p][i][j] - l[p][p1][j])**2 
                    k = k + d
                z[i] = sqrt(k)
            maxz = max(z) # the largest distance
            for i in range(0,len(l[p])):
                if z[i] == maxz:
                    p2 = i # the index of the largest distance point
        
    # dicide points belong to which cluster
            g1 = zeros(len(l[p]))
            g2 = zeros(len(l[p]))
            left = [] # the left branch
            right = [] # the right branch
            for i in range(0,len(l[p])):
                k1 = 0
                k2 = 0
                for j in range(1,len(l[p][0])): # 0? i?
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
        
            l[2*p + 1] = left # the left branch
            l[2*p + 2] = right # the right branch
            
    # re-calculate centers
            
            temp1 = mat(l[2*p + 1]) # transform into matrix
            t1 = np.mean(temp1[:,1:129], axis = 0) # mean of all columns
            s[2*p + 1] = array(t1[0,:])[0] # new center coodinates 0? except 0?
            
            temp2 = mat(l[2*p + 2]) # transform into matrix
            t2 = np.mean(temp2[:,1:129], axis = 0) # mean of all columns
            s[2*p + 2] = array(t2[0,:])[0]
    
        output = empty_array(2**n - 1)
        for i in range(0,len(l)):
            sth = []
            for j in range(0,len(l[i])):
                sth.append(l[i][j][0])
            output[i] = sth
        
        m = m + 1
   
    return s,output,l

# set x = test, n = 9

start = time.clock()

output = BallTree(test,9)
cpt = output[0] # return nodes
labels2 = output[1] # return labels
features2 = output[2] # return labels and features

end = time.clock()
timebt = end - start
print('Running time: %s Seconds'%(timebt))

start = time.clock()

a = zeros(511 - 255)
for i in range(255, 511):
    k = 0
    for j in range(1, 129):
        if len(cpt[i]) != 0:
            d = (cpt[i][j - 1] - test[u][j])**2 # distances btw center and u
            k = k + d
    a[i - 255] = sqrt(k)

mina = min(a)
for i in range(0,len(a)):
    if a[i] == mina:
        w = i + 255 # the nearest hypersphere

f = zeros(len(features2[w]))
for i in range(0,len(features2[w])):
    k = 0
    for j in range(1,len(features2[w][i])):
        l = (features2[w][i][j] - test[u][j])**2 # list
        k = k + l
    f[i] = sqrt(k)
minf = sort(f)[1:201] # extract the minimum 200 distances

labels = []
balltree = []
for i in range(0,200):
    for j in range(0,len(f)):
        if minf[i] == f[j]: # extract the index of the minimum 200 distances
            labels.append(test[j][0])
    balltree.append((labels[i],minf[i]))

end = time.clock()
timebt = end - start
print('Running time: %s Seconds'%(timebt))

# calculate the correct respond rate
k = 0
for i in range(0,len(labels2[w])):
    for j in range(0,len(labels1)):
        if labels2[w][i] == labels1[j]:
            k = k + 1
r = k/min(len(labels2[w]),len(labels1))
print(r)

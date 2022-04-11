#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 22:56:27 2022

@author: trikotash
"""
import cython
cimport cython
import numpy as np
cimport numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly.graph_objs as go

cdef double norm(double [:,:] x1,double [:,:] x2):
    cdef:
        double t
    t = 0
    for i in range(x1.shape[0]):
        for j in range(x1.shape[0]):
            t = t + (x1[i,j]-x2[i,j])**2
    
    return math.sqrt(t)

cdef double [:,:] mult(double [:,:] x1,double y):
    cdef: 
        double [:,:] a
    
    a = np.zeros((x1.shape[0],x1.shape[1])) 
    
    
    for i in range(x1.shape[0]):
         for j in range(x1.shape[0]):
             a[i,j] = y*x1[i,j]
             
    return a

cdef double [:,:] sum(double [:,:] x1,double [:,:] x2):
    cdef: 
        double [:,:] a
    
    a = np.zeros((x1.shape[0],x1.shape[1]))
    
    for i in range(x1.shape[0]):
         for j in range(x1.shape[0]):
             a[i,j] = x1[i,j]+x2[i,j]
             
    return a
 



cdef  class diff():
    
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    
    cpdef jacobi(self,init,n):
                
                cdef:
                    np.ndarray matrix_tech1,matrix_tech2
                    double eps_tech
                    int i
     
                matrix_tech2 = np.copy(init)
                matrix_tech1 = np.copy(init)
                eps_tech = 0
                for i in range(n):
                    matrix_tech2 = np.copy(matrix_tech1)
                    matrix_tech1[1:100,1:100] = (matrix_tech2[2:101,1:100]+matrix_tech2[0:99,1:100]
                                      +matrix_tech2[1:100,2:101]+matrix_tech2[1:100,0:99])*0.25
                    eps_tech = np.linalg.norm(matrix_tech1-matrix_tech2)                   
             
                return [matrix_tech2,eps_tech]


    cpdef gauss_zaidel(self,init,n): #для конденсатора
        
        cdef:
            double [:,:] matrix,tech1
            double eps_tech
            int i,j,l
            
            
       
        matrix = memoryview(init)
        tech1 = np.copy(init)
        eps_tech = 0
        for l in range(n):
            tech1[:,:] = matrix        
            for i in range(1, matrix.shape[0]-1):
                for j in range(1,matrix.shape[1]-1):
                    if i ==40 and j in range(20,80):
                            continue
                    if i ==60 and j in range(20,80):
                            continue
                    else:
                        matrix[i, j] = 0.25 * (matrix[i + 1, j] + matrix[i - 1, j] 
                                           + matrix[i, j + 1] + matrix[i, j - 1])
                      
           
            eps_tech = norm(tech1,matrix)
        
        return [init,eps_tech]
    
    cpdef gauss_zaidel_rel(self,init,n,omega):
        
        cdef:
            double [:,:] matrix,tech1,r,const_mat
            double eps_tech
            int i,j,l
            np.ndarray tech_arr
            
        const_mat = np.copy(init)
        tech1 = np.copy(init)
        matrix = memoryview(init)
        r = np.copy(init)
        eps_tech = 0
        matrix[:,:] = sum(const_mat,mult(r,omega))
        for l in range(n):
            tech1[:,:] = matrix        
            for i in range(1, r.shape[0]-1):
                    for j in range(1, r.shape[1]-1):
                        r[i, j] = 0.25 * (r[i + 1, j] + r[i - 1, j] 
                                           + r[i, j + 1] + r[i, j - 1])
            
            
            matrix[:,:] = sum(const_mat,mult(r,omega))
            
            eps_tech = norm(tech1,matrix)
        
        return [init,eps_tech]

    cpdef precision_graph(self,init,n):
         
         cdef:
             double [:] tech
             np.ndarray tech1
         tech1 = np.full(n,1,dtype = 'float') 
         tech = memoryview(tech1)
         
        
         for i in range(n):
                 tech[i] = diff().gauss_zaidel(init,i+1)[1]
                 print(i)
         return tech1
    
    cpdef precision_graph_j(self,init,n):
         
         cdef:
             double [:] tech
             np.ndarray tech1
         tech1 = np.full(n,1,dtype = 'float') 
         tech = memoryview(tech1)
         
        
         for i in range(n):
                 tech[i] = diff().jacobi(init,i+1)[1]
                 print(i)
         return tech1





    
    
    
    
    
    
    
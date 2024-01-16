from math import factorial
import numpy as np
from ast import literal_eval

#test that calculation of p score is correcct - it is

lam_smoothing = 0.01

def p_score(array1,array2):
    N = len(array1)
    C = np.dot(array1,array2)
    C = int(C)
    print(C)
    if C < 0:
        array2 = -array2
        C = -C

    
    p_arr1_pos = (array1 == 1).sum()/N
    p_arr1_neg = (array1 == -1).sum()/N

    p_arr2_pos = (array2 == 1).sum()/N
    p_arr2_neg = (array2 == -1).sum()/N

    print(p_arr1_pos)
    print(p_arr2_pos)
    print(p_arr1_neg)
    print(p_arr2_neg)
    
    p_arr1_pos = p_arr1_pos * (1 - lam_smoothing) + 0.5 * lam_smoothing
    p_arr1_neg = p_arr1_neg * (1 - lam_smoothing) + 0.5 * lam_smoothing

    p_arr2_pos = p_arr2_pos * (1 - lam_smoothing) + 0.5 * lam_smoothing
    p_arr2_neg = p_arr2_neg * (1 - lam_smoothing) + 0.5 * lam_smoothing
    
    p1 = p_arr1_pos * p_arr2_pos + p_arr1_neg * p_arr2_neg #probability of positive correlation
    p2 = p_arr1_pos * p_arr2_neg + p_arr1_neg * p_arr2_pos #probability of negative correlation

    p3 = 1 - p1 - p2 # probability of no correlation



    val = 0
    # k1 number of pointwise positive correlations
    # k2 number of pointwise negative corerlations
    # we need k1 - k2 >= C
    for k1 in range(C, N+1):
        for k2 in range(0, min(k1 - C + 1, N - k1 + 1)):
            k3 = N - k1 - k2 #number of pointwise neutral correlations
            val += factorial(N)//(factorial(k1)*factorial(k2)*factorial(k3)) * p1**k1 * p2**k2 * p3**k3


    choices = [1, -1, 0]
    N_samples = 100000
    N_pos = 0
    for _ in range(N_samples):
        array1 = np.random.choice(choices, size=(N,), p=[p_arr1_pos,p_arr1_neg,1 - p_arr1_pos - p_arr1_neg])
        array2 = np.random.choice(choices, size=(N,), p=[p_arr2_pos,p_arr2_neg,1 - p_arr2_pos - p_arr2_neg])

        C_sample = np.dot(array1,array2)

        if C_sample >= C:
            N_pos += 1


    print(N_pos/N_samples)
    print(val)
 

# a = np.array([2,1,1,2,2,2,1,0,2,2,2,2,2,1,2,0,0,0]) - 1
# b = np.array([2,1,2,2,2,1,1,1,2,2,1,2,2,1,2,0,0,0]) - 1


a = """[
  0.  0.  0. -1.  0.  0.  0.  0. -1.  0.  1.  1. -1.  1. -1.  0. -1.  0.
 -1.  0.  0.  0.  0.  1.  1. -1. -1.  0.  1.  1.  0.  1.  0.  1.  0.  1.
  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  1.  0.  0.  1.  0.  0.
  0.  1.  0.  0.  1.  1.  0. -1.  1. -1.  1.  0.  0.  0. -1.  1.  0.  0.
  0.  0.  0.  1. -1.  0. -1.  1.  1.  0.  1.  0.  1.  0. -1. -1. -1.  1.
 -1. -1.  0.  1.  0.  0.  0. -1.  0.  0.]"""
b = """
[ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  1.  0. -1. -1.  1. -1. -1.
  0. -1.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  1.  0.  1.  1.  1.
  0.  0.  1.  0.  1.  0.  1.  0.  0.  0.  0.  0.  1.  0.  0.  0. -1. -1.
  0.  1.  0.  0.  0.  0.  1.  0.  0.  0.  1.  1.  0.  0. -1.  0.  1.  0.
  1. -1. -1.  1.  0.  0.  0.  0.  1.  1.  1.  1.  1.  0. -1.  0.  0.  1.
  1.  0.  1.  0. -1. -1.  1.  0.  0.  0.]"""

a = a.replace(".",",")
b = b.replace(".",",")

a = np.array(literal_eval(a))
b = np.array(literal_eval(b))

p_score(a,b)

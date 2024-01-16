from processing import *
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
#from scipy.stats import binom
from ast import literal_eval
# from sklearn.cluster import KMeans

data_dir = "../Data3"

lam_smoothing = 0.0

sigma = 3

Normal = lambda x, mean: e**(-0.5*((x-mean)/sigma)**2) / (sigma * sqrt(2*pi))

arange_1d = np.arange(1000)
arange_2d = arange_1d[:,np.newaxis] + np.zeros((1000,1000)) #probably better way of doing this

N = Normal(arange_2d, arange_1d)

arange_1d = np.arange(100)
arange_2d = arange_1d[:,np.newaxis] + np.zeros((100,100)) #probably better way of doing this
N2 = Normal(arange_2d, arange_1d)

def score(inds_pred, ind_true):
    for i, ind in enumerate(inds_pred):
        if abs(ind[0] - ind_true[0]) < 20 and abs(ind[1] - ind_true[1]) < 20:
            return 100 #i
        
    return 0 #len(inds_pred) * 3 //2

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

    
    p_arr1_pos = p_arr1_pos * (1 - lam_smoothing) + 0.5 * lam_smoothing
    p_arr1_neg = p_arr1_neg * (1 - lam_smoothing) + 0.5 * lam_smoothing

    p_arr2_pos = p_arr2_pos * (1 - lam_smoothing) + 0.5 * lam_smoothing
    p_arr2_neg = p_arr2_neg * (1 - lam_smoothing) + 0.5 * lam_smoothing
    

    p1 = p_arr1_pos * p_arr2_pos + p_arr1_neg * p_arr2_neg #probability of positive correlation
    p2 = p_arr1_pos * p_arr2_neg + p_arr1_neg * p_arr2_pos #probability of negative correlation

    p3 = 1 - p1 - p2 # probability of no correlationnum_min



    val = 0
    # k1 number of pointwise positive correlations
    # k2 number of pointwise negative corerlations
    # we need k1 - k2 >= C
    for k1 in range(C, N+1):
        for k2 in range(0, min(k1 - C + 1, N - k1 + 1)):
            k3 = N - k1 - k2 #number of pointwise neutral correlations
            val += factorial(N)//(factorial(k1)*factorial(k2)*factorial(k3)) * p1**k1 * p2**k2 * p3**k3


    return val

def test(file_number, num_min):
    sampling_file = data_dir + "/sampled_genotypes/sample_" + str(file_number)
    with open(sampling_file, "r") as f:
        lines = f.readlines()

    X = [[float(l) for l in line[:-1]] for line in lines]

    X = np.array(X) - 1

    num_chrom, num_samples = X.shape

    # X_mean = X.mean(axis=0)
    # X = X - X_mean
    # X_norm = np.linalg.norm(X, axis=0,ord=2)
    # X = X/X_norm

    #X = X @ N

    C = X.T @ X / num_chrom #- 0.1 * S

    #C = C - N @ C @ C @ N / 100

    C = np.triu(C,k=20)

    inds_pred = []
    indices = np.argsort(abs(C).reshape(-1))[-num_min:][::-1]
    for ind in indices:
        ind = np.unravel_index(ind, C.shape)
        inds_pred.append(ind)
        print("Minimum index", ind)
        print(C[ind])

    print(inds_pred)
    # inds_pred = sorted(inds_pred, key= lambda i: p_score(X[:,i[0]], X[:,i[1]]))

    with open(data_dir + "/commands/command_" + str(file_number), "r") as f:
        s = f.readlines()[0].split()

    points = [float(s[6]), float(s[7])]
    inds = [round(num_samples*point - 0.5) for point in points]
    regular_site = round(num_samples*float(s[10]) - 0.5)

    ind_true = tuple(sorted(inds))

    print("True answer", ind_true)
    print(C[ind_true])

    print("Regular site", regular_site)

    regular1 = tuple(sorted([ind_true[0],regular_site]))
    regular2 = tuple(sorted([ind_true[1],regular_site]))

    X = []
    Y = []
    for x, y in inds_pred:
        X.append(x)
        X.append(y)
        Y.append(y)
        Y.append(x)

    plt.scatter(X,Y)
    plt.show()

    # plt.imshow(C)
    # plt.colorbar()
    # plt.show()

    # with np.printoptions(threshold=100000,linewidth=92):
    #     print(X[:,ind_true[0]-3:ind_true[0]+3])
    #     print()
    #     print(X[:,ind_true[1]-3:ind_true[1]+3])
    #     print()
    #     print(X[:,regular_site-3:regular_site+3])


    return score(inds_pred, ind_true), max(score(inds_pred, regular1), score(inds_pred, regular2))#, 15 <= score(inds_pred, ind_true) 

if __name__ == "__main__":

    # a = factorial(100)/factorial(50)

    # print(int(a)) 
    # exit()

    # for _ in range(500):
    #     A = np.random.rand(50,50)
    #     A = A/np.linalg.norm(A,"fro")
    #     A = A.T @ A
    #     print(A)
    #     print()
    # exit()

    # X = np.array([[-1,-1,-1,-1,-1,-1,-1, 0],
    #              [-1,-1,-1,-1,-1,-1, 0, 0],
    #              [ 1, 1, 1, 1, 1, 1, 0, 0]]).T
    
    # X_mean = X.mean(axis=0)
    # print(X_mean)

    # X = X - X_mean

    # print(X.T @ X)

    # exit()


    # array1 = np.array([2,1,1,2,2,2,1,0,2,2,2,2,2,1,2]) - 1
    # array2 = np.array([2,1,2,2,2,1,1,1,2,2,1,2,2,1,2]) - 1
    # array3 = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]) - 1

    # print(p_score(array1,array2))
    # exit()




    num_test = 500
    num_min = 200
    i = 7779


    # np.random.seed(177)
    # with open("smaller_strength.txt","r") as f:
    #     idx = literal_eval(f.read())

    # print(len(idx))
    idx = list(range(87500))
    idx = np.random.permutation(idx)[:num_test]
    # idx = [192]
    total_score = 0
    regular_score = 0
    n = 0
    for i in idx:
        try:
            a, b = test(i,num_min)
            total_score += a
            regular_score += b
            n += 1
        except ValueError:
            pass

    print("REGULAR SCORE", regular_score/n)
    print("FINAL SCORE", total_score/n)

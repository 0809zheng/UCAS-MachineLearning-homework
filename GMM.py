# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:30:57 2020

@author: zhijiezheng
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 设置参数
N = 200      #训练样本数
K = 2        #子模型个数
epochs = 20  #迭代次数
epsilon = 1e-6 #迭代停止精度

# 构造伪数据训练集
def init_data(N):
    mean1 = (1, 2)
    cov1 = [[1, 0], 
             [0, 0.5]]
    x1 = np.random.multivariate_normal(mean1, cov1, N//2)
    mean2 = (-2, -0.5)
    cov2 = [[1, 0], 
             [0, 1]]
    x2 = np.random.multivariate_normal(mean2, cov2, N//2)
    x = np.concatenate([x1, x2], axis = 0)
    return x

# 展示数据
def show_data(x):
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('数据展示', fontproperties = 'SimHei')
    plt.show

# 高斯混合模型
def GMM(x, K, epochs):
    # 设置初值
    N, d = x.shape
    param = {}
    for k in range(K):
        param['mean' + str(k)] = np.random.randn(d,)
        param['cov' + str(k)] = np.eye(d)
    w = np.full((K,), 1/K)
    error = 0.
    
    for epoch in range(epochs):
        # E-step
        R = np.zeros((N, K))
        for k in range(K):
            Gaussian = multivariate_normal(param['mean' + str(k)], param['cov' + str(k)])
            R[:, k] = w[k]*Gaussian.pdf(x)
        R /= np.sum(R, axis = 1, keepdims = True)
    
        # M-step
        for k in range(K):
            old_mean = param['mean' + str(k)]
            param['mean' + str(k)] = np.sum(R[:, k].reshape(N, -1)*x, axis = 0)/np.sum(R[:, k])
            error += np.sum(np.abs(old_mean - param['mean' + str(k)]))
            temp_cov = np.zeros((d, d))
            for n in range(N):
                temp_cov += R[n, k]*(x[n] - param['mean' + str(k)]).reshape(-1, 1).dot((x[n]
                - param['mean' + str(k)]).reshape(1, -1))
            param['cov' + str(k)] = temp_cov/np.sum(R[:, k])
        w = np.sum(R, axis = 0)/N
        
        # 迭代停止条件
        if error < epsilon:
            break
        
    param['w'] = w
    return param
    
# 展示模型
def show_model(param, K):
    print(K,'个子模型')
    for k in range(K):
        print('\n第%.d个子模型的参数为：'%(k+1))
        print('均值 =',param['mean' + str(k)])
        print('协方差矩阵 =',param['cov' + str(k)])

def main():
    x = init_data(N)
    show_data(x)
    param = GMM(x, K, epochs)
    show_model(param, K)
    
if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:30:40 2020

@author: zhijiezheng
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 设置参数
C = 2           # 惩罚系数
kernel = 'rbf'  # 核函数
epochs = 75     # 迭代次数
epsilon = 1e-8  # 精度

# 构造伪数据训练集
def load_data():
    # 仿照老师提供的Matlab程序，以Iris数据集为例。
    dataset = datasets.load_iris()
    x = dataset['data'][50:, 2:]
    y = dataset['target'][50:,]
    assert x.shape == (100, 2)
    assert y.shape == (100,)
    y = 2*y-3 # 修改标签为-1、1
    return x, y
    
# 构造核函数
def kernel_func(x, y):
    # 线性核
    if kernel == 'linear':
        return np.dot(x, y)
    # 高斯核
    if kernel == 'rbf':
        sigma = 1.
        return np.exp(-np.linalg.norm(x-y)/(2*sigma**2))
    
# 序列最小最优化算法
def SMO(x, y):
    # 初始化 α、b
    N = x.shape[0]
    alpha = np.zeros((N,))
    b = 0
    # 启发式学习
    for epoch in range(epochs):
        for i in range(N):
            # 计算 g(xi)、Ei
            gxi = b
            for j in range(N):
                gxi += alpha[j]*y[j]*kernel_func(x[j], x[i])
            Ei = gxi - y[i]
            # 不满足KKT条件，则更新参数
            if (gxi*y[i]<1 and abs(alpha[i])<epsilon) or (abs(gxi*y[i]-1)>epsilon and 0<alpha[i]<C) or (gxi*y[i]>1 and abs(alpha[i]-C)<epsilon): 
                # 选择 j ≠ i
                j = i
                while i == j:
                    j = np.random.randint(0, N)
                # 计算 g(xj)、Ej
                gxj = b
                for k in range(N):
                    gxj += alpha[k]*y[k]*kernel_func(x[k], x[j])
                Ej = gxj - y[j]
                # 保存 αi_old、αj_old
                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()
                # 计算 L、H
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                # 计算 η
                eta = kernel_func(x[i], x[i]) + kernel_func(x[j], x[j]) - kernel_func(x[i], x[j])*2
                # 更新 αj
                alpha[j] += y[j]*(Ei - Ej)/(eta + 1e-8)
                if alpha[j] >= H:
                    alpha[j] = H
                elif alpha[j] <= L:
                    alpha[j] = L
                # 更新 αi
                alpha[i] += y[i]*y[j]*(alpha_j_old - alpha[j])
                # 更新 b
                b1 = b -Ei - y[i]*kernel_func(x[i], x[i])*(alpha[i] - alpha_i_old) - y[j]*kernel_func(x[j], x[i])*(alpha[j] - alpha_j_old)
                b2 = b -Ej - y[i]*kernel_func(x[i], x[j])*(alpha[i] - alpha_i_old) - y[j]*kernel_func(x[j], x[j])*(alpha[j] - alpha_j_old)
                if 0<alpha[i]<C and 0<alpha[j]<C:
                    b = b1
                else:
                    b = (b1 + b2)/2
    return alpha, b
    
# 支持向量机模型
def SVM(x, y, alpha, b):
    N_new = x.shape[0]
    N = y.shape[0]
    gx = np.full((N_new,), b)
    for i in range(N_new):
        for j in range(N):
            gx[i] += alpha[j]*y[j]*kernel_func(x[j], x[i])
    y_pred = np.ones(gx.shape)
    y_pred[gx < 0] = -1
    return y_pred

# 模型展示
def show_model(x, y, alpha, b):
    x_min, x_max = x[:, 0].min() - 0.2, x[:, 0].max() + 0.2
    y_min, y_max = x[:, 1].min() - 0.2, x[:, 1].max() + 0.2
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    zz = np.c_[xx.ravel(), yy.ravel()]
    y_pred = SVM(zz, y, alpha, b)
    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c = np.squeeze(y), cmap=plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('模型展示', fontproperties = 'SimHei')
    plt.show()
    
if __name__ == '__main__':
    x, y = load_data()
    alpha, b = SMO(x, y)
    y_pred = SVM(x, y, alpha, b)
    show_model(x, y, alpha, b)
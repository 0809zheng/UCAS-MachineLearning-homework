# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:45:58 2020

@author: zhijiezheng
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from layers import *
from tensorflow.keras.datasets import mnist

#  one-hot编码
def one_hot_label(y):
    one_hot_label = np.zeros((y.shape[0],10))
    y = y.reshape(y.shape[0])
    one_hot_label[range(y.shape[0]),y] = 1
    return one_hot_label

#  加载 MNIST 数据集
def datasets():
    (x_train_origin,y_train_origin),(x_test_origin,y_test_origin) = mnist.load_data()
    X_train = x_train_origin/255.0
    X_test = x_test_origin/255.0
    m,h,w = x_train_origin.shape
    X_train = X_train.reshape((m,1,h,w))
    y_train = one_hot_label(y_train_origin)
    
    m,h,w = x_test_origin.shape
    X_test = X_test.reshape((m,1,h,w))
    y_test = one_hot_label(y_test_origin)
    return X_train, y_train, X_test, y_test
    

#  定义模型
class ConvNet:
    def __init__(self):
        self.X = None
        self.Y= None
        self.layers = []

    def add_conv_layer(self,n_filter,n_c , f, stride=1, pad=0):
        W = np.random.randn(n_filter, n_c, f, f)*0.01
        fb = np.zeros((1, n_filter))
        Conv = Convolution(W, fb, stride=stride, pad=pad)
        return Conv

    def add_maxpool_layer(self, pool_shape, stride=1, pad=0):
        pool_h, pool_w = pool_shape
        pool = Pooling(pool_h, pool_w, stride=stride, pad=pad)
        return pool
    
    def add_affine(self,n_x, n_units):
        W= np.random.randn(n_x, n_units)*0.01
        b = np.zeros((1, n_units))
        fc_layer = Affine(W,b)
        return fc_layer
    
    def add_relu(self):
        relu_layer =  Relu()
        return relu_layer
    
    def add_softmax(self):
        softmax_layer = SoftMax()
        return softmax_layer
    
    def cacl_out_hw(self,HW,f,stride = 1,pad = 0):
        return (HW+2*pad - f)/stride+1
    
    
    def init_model(self,train_X,n_classes):
        N,C,H,W = train_X.shape
        n_filter = 4
        f = 7
        
        conv_layer = self.add_conv_layer(n_filter= n_filter,n_c=C,f=f,stride=1)
        
        out_h = self.cacl_out_hw(H,f)
        out_w = self.cacl_out_hw(W,f)
        out_ch = n_filter
        
        self.layers.append(conv_layer)
        relu_layer = self.add_relu()
        self.layers.append(relu_layer)
        
        f = 2
        pool_layer = self.add_maxpool_layer(pool_shape=(f,f),stride=2)
        out_h = self.cacl_out_hw(out_h,f,stride=2)
        out_w = self.cacl_out_hw(out_w,f,stride=2)
        
        self.layers.append(pool_layer)
        
        n_x = int(out_h*out_w*out_ch)
        n_units = 32
        fc_layer = self.add_affine(n_x=n_x,n_units=n_units)
        self.layers.append(fc_layer)
        
        relu_layer = self.add_relu()
        self.layers.append(relu_layer)
        
        fc_layer = self.add_affine(n_x=n_units,n_units=n_classes)
        self.layers.append(fc_layer)
        
        softmax_layer = self.add_softmax()
        self.layers.append(softmax_layer)
        
        
    def forward_progation(self,train_X, print_out = False):
        N,C,H,W = train_X.shape
        index = 0

        conv_layer = self.layers[index]
        X = conv_layer.forward(train_X)
        index =index+1
        if print_out:
            print("卷积之后："+str(X.shape))

        relu_layer =  self.layers[index]
        index =index+1
        X = relu_layer.forward(X)
        if print_out:
            print("Relu："+str(X.shape))
            
        pool_layer = self.layers[index]
        index =index+1
        X = pool_layer.forward(X)
        if print_out:
            print("池化："+str(X.shape))

        fc_layer = self.layers[index]
        index =index+1
        X = fc_layer.forward(X)
        if print_out:
            print("Affline 层的X："+str(X.shape))

        relu_layer = self.layers[index]
        index =index+1
        X = relu_layer.forward(X)
        if print_out:
            print("Relu 层的X："+str(X.shape))
        
        fc_layer = self.layers[index]
        index =index+1
        X = fc_layer.forward(X)
        if print_out:
            print("Affline 层的X："+str(X.shape))

        sofmax_layer = self.layers[index]
        index =index+1
        A = sofmax_layer.forward(X)
        if print_out:
            print("Softmax 层的X："+str(A.shape))
    
        return A
    
    def back_progation(self,train_y,learning_rate):
        index = len(self.layers)-1
        sofmax_layer = self.layers[index]
        index -= 1
        dz = sofmax_layer.backward(train_y)
        
        fc_layer = self.layers[index]
        dz = fc_layer.backward(dz,learning_rate=learning_rate)
        index -= 1
        
        relu_layer = self.layers[index]
        dz = relu_layer.backward(dz)
        index -= 1
        
        fc_layer = self.layers[index]
        dz = fc_layer.backward(dz,learning_rate=learning_rate)
        index -= 1
        
        pool_layer = self.layers[index]
        dz = pool_layer.backward(dz)
        index -= 1
        
        relu_layer =  self.layers[index]
        dz = relu_layer.backward(dz)
        index -= 1
        
        conv_layer = self.layers[index]
        conv_layer.backward(dz,learning_rate=learning_rate)
        index -= 1
        
    def get_minibatch(self,batch_data,minibatch_size,num):
        m_examples = batch_data.shape[0]
        minibatches = math.ceil( m_examples / minibatch_size)
 
        if(num < minibatches):
            return batch_data[num*minibatch_size:(num+1)*minibatch_size]
        else:
            return batch_data[num*minibatch_size:m_examples]
    
    def optimize(self,train_X, train_y,minibatch_size,learning_rate=0.05,num_iters=500):
        m = train_X.shape[0]
        num_batches  = math.ceil(m / minibatch_size)
        
        costs = []
        for iteration in range(num_iters):
            iter_cost = 0
            for batch_num in range(num_batches):
                minibatch_X = self.get_minibatch(train_X,minibatch_size,batch_num)
                minibatch_y = self.get_minibatch(train_y,minibatch_size,batch_num)
                
                A = self.forward_progation(minibatch_X,print_out=False)

                cost = compute_cost (A,minibatch_y)

                self.back_progation(minibatch_y,learning_rate)
                if(iteration%100 == 0):
                    iter_cost += cost/num_batches
                    
            if(iteration%100 == 0):
                print("After %d iters ,loss is :%g" %(iteration,iter_cost))
                costs.append(iter_cost)
            
        plt.plot(costs)
        plt.xlabel("iterations/hundreds")
        plt.ylabel("loss")
        plt.show()
        
    def predicate(self, train_X):
        logits = self.forward_progation(train_X)
        one_hot = np.zeros_like(logits)
        one_hot[range(train_X.shape[0]),np.argmax(logits,axis=1)] = 1
        return one_hot   

    def fit(self,train_X, train_y):
        self.X = train_X
        self.Y = train_y
        n_y = train_y.shape[1]
        m = train_X.shape[0]
        
        self.init_model(train_X,n_classes=n_y)
        self.optimize(train_X, train_y,minibatch_size=10,learning_rate=0.05,num_iters=500)
        logits = self.predicate(train_X)
        accuracy = np.sum(np.argmax(logits,axis=1) == np.argmax(train_y,axis=1))/m
        
        
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = datasets()
    convNet = ConvNet()

    #  使用全部数据训练比较慢,用10张图像测试网络是否工作
    train_X = X_train[:10]
    train_y = y_train[:10]
    convNet.fit(train_X,train_y)
    logits = convNet.predicate(X_train)
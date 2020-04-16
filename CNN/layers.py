# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:24:48 2020

@author: zhijiezheng
"""

import numpy as np

#  定义ReLU激活函数
def relu(input_X):
    return np.where(input_X < 0 ,0,input_X)

#  定义Softmax函数
def softmax(input_X):
    exp_a = np.exp(input_X)
    sum_exp_a = np.sum(exp_a,axis=1)
    sum_exp_a = sum_exp_a.reshape(input_X.shape[0],-1)
    return exp_a/sum_exp_a

#  定义交叉熵
def cross_entropy_error(labels,logits):
    return -np.sum(labels*np.log(logits))

#  定义卷积层
class Convolution:
    def __init__(self,W,fb,stride = 1,pad = 0):
        self.W = W
        self.fb  = fb  
        self.stride = stride
        self.pad = pad
        
        self.col_X = None
        self.X = None
        self.col_W = None
        
        self.dW = None
        self.db = None
        self.out_shape = None
        
    def forward (self ,input_X):
        self.X = input_X
        FN,NC,FH,FW = self.W.shape
        
        m,input_nc, input_h,input_w = self.X.shape
    
        out_h = int((input_h+2*self.pad-FH)/self.stride + 1)
        out_w = int((input_w+2*self.pad-FW)/self.stride + 1)
    
        self.col_X = col_X = im2col2(self.X,FH,FW,self.stride,self.pad)
        
        self.col_W = col_W = self.W.reshape(FN,-1).T
        out = np.dot(col_X,col_W)+self.fb
        out = out.T
        out = out.reshape(m,FN,out_h,out_w)
        self.out_shape = out.shape
        return out
    
    def backward(self, dz,learning_rate):
        assert(dz.shape == self.out_shape)
    
        FN,NC,FH,FW = self.W.shape
        o_FN,o_NC,o_FH,o_FW = self.out_shape
        
        col_dz  = dz.reshape(o_NC,-1)
        col_dz = col_dz.T
        
        self.dW = np.dot(self.col_X.T,col_dz)  #shape is (FH*FW*C,FN)
        self.db = np.sum(col_dz,axis=0,keepdims=True)

        
        self.dW = self.dW.T.reshape(self.W.shape)
        self.db = self.db.reshape(self.fb.shape)
        
    
        d_col_x = np.dot(col_dz,self.col_W.T) #shape is (m*out_h*out_w,FH,FW*C)
        dx = col2im2(d_col_x,self.X.shape,FH,FW,stride=1)
        
        assert(dx.shape == self.X.shape)
        
        self.W = self.W - learning_rate*self.dW
        self.fb = self.fb -learning_rate*self.db
        
        return dx

#  定义池化层
class Pooling:
    def __init__(self,pool_h,pool_w,stride = 1,pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad 
        self.X = None
        self.arg_max = None
        
    def forward ( self,input_X) :
        self.X = input_X
        N , C, H, W = input_X.shape
        out_h = int(1+(H-self.pool_h)/self.stride)
        out_w = int(1+(W-self.pool_w)/self.stride)
        
        col = im2col2(input_X,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)
        arg_max = np.argmax(col,axis=1)

        out = np.max(col,axis=1)
        out =out.T.reshape(N,C,out_h,out_w)
        self.arg_max = arg_max
        return out
    
    def backward(self ,dz):
        pool_size = self.pool_h*self.pool_w
        dmax = np.zeros((dz.size,pool_size))
        dmax[np.arange(self.arg_max.size),self.arg_max.flatten()] = dz.flatten()
        
        dx = col2im2(dmax,out_shape=self.X.shape,fh=self.pool_h,fw=self.pool_w,stride=self.stride)
        return dx
    
class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self ,X):
        self.mask = X <= 0
        out = X
        out[self.mask] = 0
        return out
    
    def backward(self,dz):
        dz[self.mask] = 0
        dx = dz 
        return dx
    
class SoftMax:
    def __init__ (self):
        self.y_hat = None
        
    def forward(self,X):
        
        self.y_hat = softmax(X)
        return self.y_hat
    
    def backward(self,labels):
        m = labels.shape[0]
        dx = (self.y_hat - labels)
        
        return dx
    
def compute_cost(logits,label):
    return cross_entropy_error(label,logits)

#  定义仿射层
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b  = b
        self.X = None
        self.origin_x_shape = None
        
        self.dW = None
        self.db = None
        
        self.out_shape =None
        
    def forward(self,X):
        self.origin_x_shape = X.shape 
        self.X = X.reshape(X.shape[0],-1)
        out =  np.dot(self.X, self.W)+self.b
        self.out_shape = out.shape
        return out
    
    def backward(self,dz,learning_rate):
        assert(dz.shape == self.out_shape)
        
        m = self.X.shape[0]
        
        self.dW = np.dot(self.X.T,dz)/m
        self.db = np.sum(dz,axis=0,keepdims=True)/m
        
        assert(self.dW.shape == self.W.shape)
        assert(self.db.shape == self.b.shape)
        
        dx = np.dot(dz,self.W.T)
        assert(dx.shape == self.X.shape)
        
        dx = dx.reshape(self.origin_x_shape) 
        
        self.W = self.W-learning_rate*self.dW
        self.b = self.b - learning_rate*self.db
        return dx
    
#  图像 → 矩阵
def im2col2(input_data,fh,fw,stride=1,pad=0):
    N,C,H,W = input_data.shape
    
    out_h = (H + 2*pad - fh)//stride+1
    out_w = (W+2*pad-fw)//stride+1
    
    img = np.pad(input_data,[(0,0),(0,0),(pad,pad),(pad,pad)],"constant")
    
    col = np.zeros((N,out_h,out_w,fh*fw*C))
    
    for y in range(out_h):
        y_start = y * stride
        y_end =  y_start + fh
        for x in range(out_w):
            x_start = x*stride
            x_end = x_start+fw
            col[:,y,x] = img[:,:,y_start:y_end,x_start:x_end].reshape(N,-1)
    col = col.reshape(N*out_h*out_w,-1)
    return col

#  矩阵 → 图像
def col2im2(col,out_shape,fh,fw,stride=1,pad=0):
    N,C,H,W = out_shape
    
    col_m,col_n = col.shape
    
    out_h = (H + 2*pad - fh)//stride+1
    out_w = (W+2*pad-fw)//stride+1

    img = np.zeros((N, C, H , W))

    for c in range(C):
        for y in range(out_h):
            for x in range(out_w):
                col_index = (c*out_h*out_w)+y*out_w+x
                ih = y*stride
                iw =  x*stride
                img[:,c,ih:ih+fh,iw:iw+fw] = col[col_index].reshape((fh,fw))
    return img
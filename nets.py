import tensorflow as tf
import tensorflow.contrib.slim as slim
import os.path
import glob
from tensorflow.python.platform import gfile
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import *
import time

layer = 16
regularizer = tf.contrib.layers.l2_regularizer(0.0005)
def con1(inputs,Training = True,Reuse = False,alpha=0.2, name = 'net'):
    with tf.variable_scope(name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[3, 3 , 3, 16],tf.float32,tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1',[16],tf.float32,tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 4, 4, 1], padding='VALID', name='deconv1')
        mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.maximum(alpha*net,net)             #64
        net = tf.nn.max_pool(net,[1,2,2,1],[1,1,1,1],'VALID')
    return net

def con2(inputs,Training = True,Reuse = False,alpha=0.2, name = 'net'):           
    with tf.variable_scope(name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[5, 5, 16, 32],tf.float32,tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1',[32],tf.float32,tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 2, 2, 1], padding='SAME', name='deconv1')
        mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.maximum(alpha*net,net)             #32
        net = tf.nn.max_pool(net,[1,2,2,1],[1,1,1,1],'VALID')
    return net

def con3(inputs,Training = True,Reuse = False,alpha=0.2, name = 'net'):              
    with tf.variable_scope(name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[3, 3, 32, 64],tf.float32,tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1',[64],tf.float32,tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 1, 1, 1], padding='SAME', name='deconv1')
        #mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        #net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.nn.bias_add(conv1,bias1)
        net = tf.maximum(alpha*net,net)             #16
    return net

def con4(inputs,Training = True,Reuse = False,alpha=0.2, name = 'net'):       
    with tf.variable_scope(name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[3, 3, 64, 128],tf.float32,tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1',[128],tf.float32,tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 1, 1, 1], padding='SAME', name='deconv1')
        net = tf.nn.bias_add(conv1,bias1)
        net = tf.maximum(alpha*net,net)
        net = tf.nn.max_pool(net,[1,2,2,1],[1,1,1,1],'VALID')             #3
        code = net                      
    return net  

   
def fc1(inputs,Training = True,Reuse = False,alpha=0.2, name = 'net'):   
    code_shape = inputs.get_shape().as_list()
    nodes = code_shape[1]*code_shape[2]*code_shape[3]
    inputs = tf.reshape(inputs,[code_shape[0],nodes])
    with tf.variable_scope(name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[nodes,1024],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable('bias1',[1024],tf.float32,initializer=tf.zeros_initializer())
        net = tf.matmul(inputs,weight1)
        #mean, variance = tf.nn.moments(net, [0, 1])
        #net = tf.nn.batch_normalization(net, mean, variance, bias1, None, 1e-5)
        net = net + bias1
        net = tf.maximum(alpha*net,net)
        if Training: net = tf.nn.dropout(net,0.5)
        tf.add_to_collection('losses',regularizer(weight1)) 
    return net
def fc2(inputs,Training = True,Reuse = False,alpha=0.2, name = 'net'):                   
    with tf.variable_scope(name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[1024,512],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable('bias1',[512],tf.float32,initializer=tf.zeros_initializer())
        net = tf.matmul(inputs,weight1)
        #mean, variance = tf.nn.moments(net, [0,1])
        #net = tf.nn.batch_normalization(net, mean, variance, bias1, None, 1e-5)
        net = net + bias1
        net = tf.maximum(alpha*net,net)
        if Training: net = tf.nn.dropout(net,0.5)
        tf.add_to_collection('losses',regularizer(weight1))
    return net
def fc3(inputs,Training = True,Reuse = False,alpha=0.2, name = 'net'):        
    with tf.variable_scope(name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[512,128],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable('bias1',[128],tf.float32,initializer=tf.zeros_initializer())
        net = tf.matmul(inputs,weight1) + bias1
        #net = tf.nn.tanh(net)
        #net = tf.maximum(alpha*net,net)
        #net = tf.nn.l2_normalize(net, dim=1)
    return net  #1



def classfilter(inputs,Training=True,Reuse = False):
    with tf.variable_scope('classfilter',reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[4096,262],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable('bias1',[262],tf.float32,initializer=tf.zeros_initializer())
        net = tf.matmul(inputs,weight1) + bias1
    return net


       
       
       
       
       
       
       
       
       
       
       

       
            


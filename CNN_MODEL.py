import tensorflow as tf
import pandas as pd
import numpy as np
import random

IMAGE_SIZE = 48
CLIPED_SIZE = 42
EMO_NUM = 7
NUM_CHANNEL = 1
SAVE_PATH = './saved_model'

def GetSymmetric(pixel, size):
    '''
    pixel: np.array with shape (count,size,size,1); 
    size: picture size; 
    return: symmetric np.array with shape (count,size,size,1); 
    '''
    count = pixel.shape[0]
    sym = np.zeros((count, size, size, NUM_CHANNEL))
    for i in range(count):
        for j in range(size):
            for k in range(size):
                sym[i,j,k,0] = pixel[i,j,size-k-1,0]
    return sym

def GetClippedImage(pixel, start):
    '''
    pixel: raw 48*48 pixel data with shape (count, 48, 48, 1); 
    start: a tuple such as (0,0),(2,3),(4,2), represents start point of clipped 42*42 image; 
    returns: clipped 42*42 pixel data with shape (count, 42, 42, 1); 
    '''
    count = pixel.shape[0]
    out = np.zeros((count, CLIPED_SIZE, CLIPED_SIZE, NUM_CHANNEL))
    for i in range(count):
        for j in range(CLIPED_SIZE):
            out[i,j,:,0] = pixel[i,start[0]+j,start[1]:start[1]+CLIPED_SIZE,0]
    return out

def DataPreprocess(pixel):
    '''
    pixel: pixel data with shape (count,48,48,1); 
    label: optical, corresponding label of pixel; 
    '''
    a = random.randint(0,2)
    b = random.randint(3,5)
    c = random.randint(0,2)
    d = random.randint(3,5)
    pixel1 = GetClippedImage(pixel, (a,c))
    pixel2 = GetClippedImage(pixel, (a,d))
    pixel3 = GetClippedImage(pixel, (b,c))
    pixel4 = GetClippedImage(pixel, (b,d))
    out_p = np.concatenate((pixel1, pixel2, pixel3, pixel4), axis = 0)
    return out_p

def Initialize():
    '''
    returns: a session with loaded graph
    '''
    sess = tf.Session()
    loader = tf.train.import_meta_graph(SAVE_PATH+'.meta')
    loader.restore(sess, SAVE_PATH)
    return sess

def Predict(pixel, sess):
    '''
    pixel: np.array with shape (48, 48); 
    returns: np.array with shape (7,); 
    '''
    max = pixel.max()+0.001
    for i in range(IMAGE_SIZE):
        pixel[i] = pixel[i]/max
    pixel = pixel.reshape((1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL))
    in_data = DataPreprocess(pixel)
    in_data = np.concatenate((in_data, GetSymmetric(in_data, CLIPED_SIZE)), axis = 0)
    loaded_graph = tf.get_default_graph()
    load_x = loaded_graph.get_tensor_by_name('INPUT:0')
    load_y = loaded_graph.get_tensor_by_name('LABEL:0')
    load_log = loaded_graph.get_tensor_by_name('LOGITS:0')
    load_keep = loaded_graph.get_tensor_by_name('KEEP:0')
    logit = sess.run(load_log, feed_dict={
        load_x: in_data, load_y: np.zeros((8,EMO_NUM)), load_keep: 1.0
    })
    log = np.zeros((1, EMO_NUM))
    for i in range(8):
        log += logit[i]
    log = sess.run(tf.nn.softmax(log))
    return log


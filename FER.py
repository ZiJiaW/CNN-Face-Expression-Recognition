import tensorflow as tf
import pandas as pd
import numpy as np
import random
from PIL import Image
import CNN_MODEL as MODEL

IMAGE_SIZE = 48
CLIPED_SIZE = 42
EMO_NUM = 7
TRAIN_SIZE = 4*(35887*2-10000)
VALID_SIZE = 1500
TEST_SIZE = 5000
BATCH_SIZE = 50
NUM_CHANNEL = 1
EPOCHS = 50
SAVE_PATH = './saved_model'
emo_dict = {
    0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Suprise', 6:'Neutral'
}
def GetSymmetric(pixel, size):
    '''
    pixel: np array with shape (count,size,size,1)
    '''
    count = pixel.shape[0]
    sym = np.zeros((count, size, size, NUM_CHANNEL))
    for i in range(count):
        for j in range(size):
            for k in range(size):
                sym[i,j,k,0] = pixel[i,j,size-k-1,0]
    return sym

def GetClipedImage(pixel, start):
    '''
    pixel: raw 48*48 pixel data with shape (count, 48, 48, 1)
    start: a tuple such as (0,0),(2,3),(4,2), represents start point of clipped 42*42 image
    '''
    count = pixel.shape[0]
    out = np.zeros((count, CLIPED_SIZE, CLIPED_SIZE, NUM_CHANNEL))
    for i in range(count):
        for j in range(CLIPED_SIZE):
            out[i,j,:,0] = pixel[i,start[0]+j,start[1]:start[1]+CLIPED_SIZE,0]
    return out

def GetInput():
    all_data = pd.read_csv('fer2013.csv')
    label = np.array(all_data['emotion'])
    data = np.array(all_data['pixels'])
    sample_count = len(label)  # should be 35887

    pixel_data = np.zeros((sample_count, IMAGE_SIZE * IMAGE_SIZE))# 像素点数据
    label_data = np.zeros((sample_count, EMO_NUM), dtype = int)# 标签数据，独热
    for i in range(sample_count):
        x = np.fromstring(data[i], sep = ' ')
        max = x.max()
        x = x / (max + 0.001)  # 灰度归一化
        pixel_data[i] = x
        label_data[i, label[i]] = 1
    pixel_data = pixel_data.reshape(sample_count, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
    x_test = pixel_data[30000:35000]
    y_test = label_data[30000:35000]

    x_train = np.concatenate((pixel_data[0:30000],pixel_data[35000:]), axis = 0)
    symmetric_x_train = GetSymmetric(x_train, IMAGE_SIZE)
    x_train = np.concatenate((x_train, symmetric_x_train), axis = 0)
    y_train = np.concatenate((label_data[0:30000],label_data[35000:],label_data[0:30000],label_data[35000:]))

    return (x_train, y_train, x_test, y_test)

def DataPreprocess(pixel, label = []):
    '''
    pixel: pixel data with shape (count,48,48,1)
    label: optical, corresponding label of pixel
    '''
    a = random.randint(0,2)
    b = random.randint(3,5)
    c = random.randint(0,2)
    d = random.randint(3,5)
    pixel1 = GetClipedImage(pixel, (a,c))
    pixel2 = GetClipedImage(pixel, (a,d))
    pixel3 = GetClipedImage(pixel, (b,c))
    pixel4 = GetClipedImage(pixel, (b,d))
    out_p = np.concatenate((pixel1, pixel2, pixel3, pixel4), axis = 0)
    if len(label) == 0:
        return out_p
    else:
        out_l = np.concatenate((label, label, label, label), axis = 0)
        return (out_p, out_l)

def model(data, keep_prob):
    # first layer IN: 42*42*1  OUT: 20*20*32
    kernel1 = tf.Variable(tf.truncated_normal([5,5,NUM_CHANNEL,32], stddev = 5e-2))
    conv1 = tf.nn.conv2d(data, kernel1, [1,1,1,1], padding = 'SAME')
    bias1 = tf.Variable(tf.zeros([32]))
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
    pool1 = tf.nn.max_pool(relu1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')

    # second layer IN: 20*20*32  OUT: 10*10*32
    kernel2 = tf.Variable(tf.truncated_normal([4,4,32,32],stddev = 5e-2))
    conv2 = tf.nn.conv2d(pool1, kernel2, [1,1,1,1], padding = 'SAME')
    bias2 = tf.Variable(tf.zeros([32]))
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
    pool2 = tf.nn.max_pool(relu2, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

    # third layer IN: 10*10*32  OUT: 5*5*64
    kernel3 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev = 5e-2))
    conv3 = tf.nn.conv2d(pool2, kernel3, [1,1,1,1], padding = 'SAME')
    bias3 = tf.Variable(tf.zeros([64]))
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, bias3))
    pool3 = tf.nn.max_pool(relu3, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')

    # fully connected layers
    fc1_data = tf.reshape(pool3, shape = [-1, 5*5*64])
    fc1 = tf.contrib.layers.fully_connected(fc1_data, 1024, activation_fn = tf.nn.relu)
    fc1_out = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.contrib.layers.fully_connected(fc1_out, 512, activation_fn = tf.nn.relu)
    fc2_out = tf.nn.dropout(fc2, keep_prob)

    logits = tf.contrib.layers.fully_connected(fc2_out, 7, activation_fn = None)
    logits = tf.identity(logits, name = 'LOGITS')
    return logits

def train(x_train, y_train, x_val, y_val):
    x_data = tf.placeholder(tf.float32, shape = (None, CLIPED_SIZE, CLIPED_SIZE, NUM_CHANNEL), name = 'INPUT')
    y_data = tf.placeholder(tf.int16, shape = (None, EMO_NUM), name = 'LABEL')
    keep_prob = tf.placeholder(tf.float32, name = 'KEEP')
    y_pred = model(x_data, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_pred, labels = y_data))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1,global_step,300,0.99,staircase=True)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'ACCURACY')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        for epoch in range(EPOCHS):
            for batch_i in range(TRAIN_SIZE//BATCH_SIZE):
                step += 1
                x_feats = x_train[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
                y_feats = y_train[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
                feed = {x_data: x_feats, y_data: y_feats, keep_prob: 0.6}
                sess.run(optimizer, feed_dict=feed)
                if step % 128 == 0:
                    (loss, acc) = sess.run([cost, accuracy], feed_dict=feed)
                    feed_v = {x_data: x_val, y_data: y_val, keep_prob: 1.0}
                    acc_v = sess.run(accuracy, feed_dict=feed_v)
                    print("In epoch %d, batch %d, loss: %.3f, accuracy: %.3f, validation accuracy: %.3f" % (epoch, batch_i, loss, acc, acc_v))
        saver = tf.train.Saver()
        saver_path = saver.save(sess, SAVE_PATH)
        print('Finished!')

def test(x_test, y_test):
    loaded_graph = tf.Graph()
    with tf.Session(graph = loaded_graph) as sess:
        # load the model
        loader = tf.train.import_meta_graph(SAVE_PATH + '.meta')
        loader.restore(sess, SAVE_PATH)
        load_x = loaded_graph.get_tensor_by_name('INPUT:0')
        load_y = loaded_graph.get_tensor_by_name('LABEL:0')
        load_acc = loaded_graph.get_tensor_by_name('ACCURACY:0')
        load_log = loaded_graph.get_tensor_by_name('LOGITS:0')
        load_keep = loaded_graph.get_tensor_by_name('KEEP:0')
        # record accuracy
        total_batch_acc = 0
        batch_count = TEST_SIZE//BATCH_SIZE
        (x_test, y_test) = DataPreprocess(x_test, y_test)
        for batch_i in range(batch_count):
            log = np.zeros((BATCH_SIZE, EMO_NUM))
            y_feats = y_test[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
            for k in range(4):
                x_feats = x_test[batch_i*BATCH_SIZE + k*TEST_SIZE: (batch_i+1)*BATCH_SIZE + k*TEST_SIZE]
                log1 = sess.run(load_log, feed_dict = {
                    load_x : x_feats, load_y : y_feats, load_keep: 1.0
                })
                x_feats = GetSymmetric(x_feats, CLIPED_SIZE)
                log2 = sess.run(load_log, feed_dict = {
                    load_x : x_feats, load_y : y_feats, load_keep: 1.0
                })
                log += log1 + log2
            emos = sess.run(tf.argmax(log, 1))
            correct_emos = sess.run(tf.argmax(y_feats, 1))
            tmp = emos == correct_emos
            acc = tmp.sum()/tmp.shape[0]
            total_batch_acc += acc
            print('In test batch %d: the accuracy is %.3f' % (batch_i, acc))
        print('Total accuracy in test set is %.3f' % (total_batch_acc / batch_count))

def classify(files):
    '''
    files: 需要识别的图片路径列表，放在同目录下的相对路径即可，如['im1.jpg','im2.jpg']
    提取像素用的Pillow库的Image模块
    '''
    file_count = len(files)
    pixel = np.zeros((file_count,IMAGE_SIZE*IMAGE_SIZE))
    for file_index in range(file_count):
        im = Image.open(files[file_index]).convert('L')
        im = im.resize((IMAGE_SIZE,IMAGE_SIZE))
        for i in range(IMAGE_SIZE*IMAGE_SIZE):
            pixel[file_index,i] = im.getpixel((i//IMAGE_SIZE, i%IMAGE_SIZE))
        pixel[file_index] = pixel[file_index]/(pixel[file_index].max()+0.001)
    pixel = pixel.reshape(file_count, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
    pixel = DataPreprocess(pixel)
    loaded_graph = tf.Graph()
    with tf.Session(graph = loaded_graph) as sess:
        loader = tf.train.import_meta_graph(SAVE_PATH + '.meta')
        loader.restore(sess, SAVE_PATH)
        load_x = loaded_graph.get_tensor_by_name('INPUT:0')
        load_y = loaded_graph.get_tensor_by_name('LABEL:0')
        load_log = loaded_graph.get_tensor_by_name('LOGITS:0')
        load_keep = loaded_graph.get_tensor_by_name('KEEP:0')
        logit = sess.run(load_log, feed_dict={
            load_x: pixel, load_y: np.zeros((file_count*4,EMO_NUM)), load_keep: 1.0
        })
        log = np.zeros((file_count, EMO_NUM))
        for i in range(4):
            log += logit[i*file_count:(i+1)*file_count]
        emos = sess.run(tf.argmax(log, 1))
        for emo in emos:
            print(emo_dict[emo])

if __name__ == '__main__':
    
    (x_train, y_train, x_test, y_test) = GetInput()
    
    
    #print(x_train.shape)
    (x_train, y_train) = DataPreprocess(x_train, y_train)
    print(x_train.shape)
    print('Start!')
    x_val = x_test[0:500]
    y_val = y_test[0:500]
    (x_val, y_val) = DataPreprocess(x_val, y_val)
    train(x_train, y_train, x_val, y_val)
    
    
    #print(x_test.shape)
    test(x_test, y_test)
    
    #classify(['test1.JPG','test2.JPG','test3.JPG','test4.JPG','test5.JPG','test6.JPG','test7.JPG','test8.JPG','test9.JPG'])
'''
    pixel = np.zeros((48,48))
    im = Image.open('test1.JPG').convert('L')
    im = im.resize((IMAGE_SIZE,IMAGE_SIZE))
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            pixel[i,j]=im.getpixel((i,j))
    sess = MODEL.Initialize()
    print(MODEL.Predict(pixel, sess))
    
'''
    
    
    
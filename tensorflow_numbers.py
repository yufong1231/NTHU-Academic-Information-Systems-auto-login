import cv2
import tensorflow as tf
import os
import numpy as np
import random as r
import math

def get_File(file_dir):
    images = []
    subfolders = []

    for dirPath, dirNames, fileNames in os.walk(file_dir):

        names = []
        for name in fileNames:
            names.append(os.path.join(dirPath, name))
        for name in dirNames:
            subfolders.append(os.path.join(dirPath, name))

        r.shuffle(names)
        if names != []:
            images.append(names)

    mincount = float("Inf")
    for num_folder in subfolders:
        n_img = len(os.listdir(num_folder))

        if n_img < mincount:
            mincount = n_img

    for i in range(len(images)):
        images[i] = images[i][0:mincount]

    images = np.reshape(images, [mincount*len(subfolders), ])

    labels = []

    for count in range(len(subfolders)):
        print('index: ', count, ' = ', subfolders[count])
        labels = np.append(labels, mincount * [count])


    subfolders = np.array([images, labels])
    subfolders = subfolders[:, np.random.permutation(subfolders.shape[1])].T

    image_list = list(subfolders[:, 0])
    label_list = list(subfolders[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_TFRecord(images, labels, filename):
    n_samples = len(labels)
    TFWriter = tf.python_io.TFRecordWriter(filename)

    print('\nTransform start...')

    for i in np.arange(0, n_samples):
        try:
            image = cv2.imread(images[i], 0)

            if image is None:
                print('Error image : ' + images[i])
            else:
                image_raw = image.tostring()

            label = int(labels[i])

            ftrs = tf.train.Features(
                    feature = {'Label': int64_feature(label),
                               'image_raw': bytes_feature(image_raw)}
                   )
            example = tf.train.Example(features=ftrs)

            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')

    TFWriter.close()
    print('Transform done')

def read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    img_features = tf.parse_single_example(
                    serialized_example,
                    features={ 'Label' : tf.FixedLenFeature([], tf.int64),
                               'image_raw' : tf.FixedLenFeature([], tf.string)}
                   )
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28])

    label = tf.cast(img_features['Label'], tf.int64)

    image_batch, label_batch = tf.train.shuffle_batch(
                                [image, label],
                                batch_size=batch_size,
                                capacity=10000 + 3 * batch_size,
                                min_after_dequeue=1000
                               )

    return image_batch, label_batch

def generate_tfrecord():
    train_dataset_dir = './dataset/'

    images, labels = get_File(train_dataset_dir)

    convert_to_TFRecord(images, labels, './Train.tfrecords')

def softmax(z):

    z_exp = [math.exp(i) for i in z]


    # Result: [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]

    sum_z_exp = sum(z_exp)

    # Result: 114.98

    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    print(softmax)
def train_model():

    filename = './Train.tfrecords'
    batch_size = 10
    Label_size = 10


    image_batch, label_batch = read_and_decode(filename, batch_size)

    image_batch_train = tf.reshape(image_batch, [-1, 28*28])
    label_batch_train = tf.one_hot(label_batch, Label_size)

    #W = tf.Variable(tf.zeros([28*28, Label_size]))
    W =  tf.Variable(tf.truncated_normal([784, Label_size], stddev=0.1))
    #b = tf.Variable(tf.zeros([Label_size]))
    b = tf.Variable(tf.zeros([1, Label_size]) + 0.1,)
    x = tf.placeholder(tf.float32, [None, 28*28])

    #important , need to convert x(0~255) to x(0~1)
    y = tf.nn.softmax(tf.matmul(x/255, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y + 1e-10),reduction_indices=[1]))
    #cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    #cross_entropy = -tf.reduce_sum(y_*tf.log(y + 1e-10))
    train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cross_entropy)
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ##########################################################
    y_pred = tf.argmax(y, 1)

    saver = tf.train.Saver()

    tf.add_to_collection('input', x)
    tf.add_to_collection('output', y_pred)

    iscontinue = 0

    save_path = './model/test_model'
    ##########################################################

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            if not iscontinue:
                for count in range(1000):
                    ##load data
                    image_data, label_data = sess.run([image_batch_train, label_batch_train])


                    ##train model
                    sess.run(train_step, feed_dict={x: image_data, y_: label_data})

                    ##print accuracy
                    if count % 100 == 0:
                        yy = sess.run(y, feed_dict={x: image_data, y_: label_data})
                        loss =  sess.run(cross_entropy, feed_dict={x: image_data, y_: label_data})
                        train_accuracy = sess.run(accuracy, feed_dict={x: image_data, y_: label_data})
                        #print(image_data)
                        #train_accuracy = accuracy.eval(feed_dict={x: image_data, y_: label_data})
                        #print('loss : ' ,loss, 'y : ', yy)
                        print('Iter %d, accuracy %4.2f%%' % (count, train_accuracy*100))

                        ##save Variable
                        spath = saver.save(sess, save_path, global_step=count)
                        #print("Model saved in file: %s" % spath)
            else:
                ##rebuild the Model
                last_ckp = tf.train.latest_checkpoint("./model")
                saver = tf.train.import_meta_graph(last_ckp + '.meta')
                saver.restore(sess, last_ckp)

                for count in range(100,200):
                    ##load data
                    image_data, label_data = sess.run([image_batch_train, label_batch_train])

                    ##train model
                    sess.run(train_step, feed_dict={x: image_data, y_: label_data})

                    ##print accuracy
                    if count % 100 == 0:
                        train_accuracy = sess.run(accuracy, feed_dict={x: image_data, y_: label_data})
                        #train_accuracy = accuracy.eval(feed_dict={x: image_data, y_: label_data})
                        #print('W : ' ,sess.run(W), ' b : ', sess.run(b))
                        print('Iter %d, accuracy %4.2f%%' % (count, train_accuracy*100))

                        ##save Variable
                        spath = saver.save(sess, save_path, global_step=count)
                        #print("Model saved in file: %s" % spath)

        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()

        coord.join(threads)

############################################################
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

############################################################
def train_model_CNN():

    filename = './Train.tfrecords'
    batch_size = 100
    Label_size = 10

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [None, 10])
    #keep_prob = tf.placeholder(tf.float32)

    image_batch, label_batch = read_and_decode(filename, batch_size)
    image_batch_train = tf.reshape(image_batch, [-1, 28, 28, 1])
    label_batch_train = tf.one_hot(label_batch, Label_size)

    #conv1 layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #conv2 layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #fc1 layer
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    #fc2 layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
    #train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ##########################################################
    y_pred = tf.argmax(y, 1)

    saver = tf.train.Saver()

    tf.add_to_collection('input', x)
    tf.add_to_collection('output', y_pred)

    iscontinue = 0

    save_path = './model/test_model'
    ##########################################################

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            if not iscontinue:
                for count in range(1000):
                    ##load data
                    image_data, label_data = sess.run([image_batch_train, label_batch_train])

                    ##train model
                    sess.run(train_step, feed_dict={x: image_data, y_: label_data})

                    ##print accuracy
                    if count % 100 == 0:
                        train_accuracy = sess.run(accuracy, feed_dict={x: image_data, y_: label_data})
                        #train_accuracy = accuracy.eval(feed_dict={x: image_data, y_: label_data})
                        print('Iter %d, accuracy %4.2f%%' % (count, train_accuracy*100))

                        ##save Variable
                        spath = saver.save(sess, save_path, global_step=count)
                        #print("Model saved in file: %s" % spath)
            else:
                ##rebuild the Model
                last_ckp = tf.train.latest_checkpoint("./model")
                saver = tf.train.import_meta_graph(last_ckp + '.meta')
                saver.restore(sess, last_ckp)

                for count in range(100,200):
                    ##load data
                    image_data, label_data = sess.run([image_batch_train, label_batch_train])

                    ##train model
                    sess.run(train_step, feed_dict={x: image_data, y_: label_data})

                    ##print accuracy
                    if count % 100 == 0:
                        train_accuracy = sess.run(accuracy, feed_dict={x: image_data, y_: label_data})
                        #train_accuracy = accuracy.eval(feed_dict={x: image_data, y_: label_data})
                        print('Iter %d, accuracy %4.2f%%' % (count, train_accuracy*100))

                        ##save Variable
                        spath = saver.save(sess, save_path, global_step=count)
                        #print("Model saved in file: %s" % spath)

        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()

        coord.join(threads)

def predict_image():
    with tf.Session() as sess:
        save_path = './model/test_model-900.meta'

        saver = tf.train.import_meta_graph(save_path)

        saver.restore(sess, "./model/test_model-900")

        x = tf.get_collection("input")[0]
        y = tf.get_collection("output")[0]

        img = cv2.imread('./test/0_0.jpg', 0)

        result = sess.run(y, feed_dict = {x: img.reshape((-1, 28*28))})
        answer = '0'
        if result[0] == 0:
            answer = '9'
        elif result[0] == 1:
            answer = '0'
        elif result[0] == 2:
            answer = '7'
        elif result[0] == 3:
            answer = '6'
        elif result[0] == 4:
            answer = '1'
        elif result[0] == 5:
            answer = '8'
        elif result[0] == 6:
            answer = '4'
        elif result[0] == 7:
            answer = '3'
        elif result[0] == 8:
            answer = '2'
        else:
            answer = '5'
        print('predict: ', answer)


def test():

    filename = './Train.tfrecords'

    filename_queue = tf.train.string_input_producer([filename],
                                                     shuffle=False,
                                                     num_epochs=1)

    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    img_features = tf.parse_single_example(
                serialized_example,
                features={ 'Label'    : tf.FixedLenFeature([], tf.int64),
                           'image_raw': tf.FixedLenFeature([], tf.string), })

    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28])

    label = tf.cast(img_features['Label'], tf.int64)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord)

        count = 0
        try:
            while count<100:

                image_data, label_data = sess.run([image, label])

                cv2.imwrite('./test/tf_%d_%d.jpg' % (label_data, count), image_data)
                count += 1

            print('Done!')

        except tf.errors.OutOfRangeError:
            print('Done!')

        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    #----change single number images to tfrecord----
    generate_tfrecord()
    #----train model using softmax----
    train_model()
    #----train model using CNN(not necessary here)----
    #train_model_CNN()
    #----test tfrecord is correct or not---
    #test()
    #----predict single number image---
    #predict_image()


######################################
#----because os.walk would not read path in order , so we record the index correspond to which number----
"""
index:  0  =  ./dataset/9
index:  1  =  ./dataset/0
index:  2  =  ./dataset/7
index:  3  =  ./dataset/6
index:  4  =  ./dataset/1
index:  5  =  ./dataset/8
index:  6  =  ./dataset/4
index:  7  =  ./dataset/3
index:  8  =  ./dataset/2
index:  9  =  ./dataset/5
"""
########################################

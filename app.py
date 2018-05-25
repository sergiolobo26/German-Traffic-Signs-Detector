#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:53:01 2018

@author: sd.lobo251
"""
import os
import shutil
import random
import requests 
import zipfile
import io
import click
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import tensorflow as tf

random.seed(0)

TEST_IMG_DIR = 'images/test/'
TRAIN_IMG_DIR = 'images/train/'
n_clusters = 200
BATCH_SIZE = 32       
EPOCHS = 15

tf.logging.set_verbosity(tf.logging.INFO)

@click.group()

def cli():
    pass

@cli.command()
def download():
    '''
    This functions calls the main_download() function which
    do the following:\n
        - Downloand and unzip the data set \n
        - Select from the dataset only the classification folders\n
        - Duplicate data to ensure every class has at least 60 images \n
        - shuffle the data set and saves the images in the train - test folders.\n
        - Save the image name and label in a txt file in the /images/test-train folder. 
    this function DOES NOT create the folders, so, for it to work, the directory tree must
    be created first. 
    '''
    click.echo('Downloading the data set...')
    main_download()
    click.echo('Download finished.')
    

@cli.command()
@click.option('-m', default='LeNet', help ='Model to infer. Valid names: LeNet, tf_Logistic, Logistic')
@click.option('-d', default='images/user/', help = 'User directory to infer. Defaul: images/user/')
#@cli.argument('model')

def infer(m, d):
    '''
    Infer the class of an image in a direcory with a specific model. Model valid names:\n
    - LeNet \n
    - tf_Logistic \n
    - Logistic
    '''

    if m=='LeNet':
        LeNet_infer(d, show_window=False)
        
    elif m=='tf_Logistic':
        log_model_tf_infer(d, show_window=False )
    
    elif m=='Logistic':
        if os.path.exists(d+'infer_data_file.txt'):
            os.remove(d+'infer_data_file.txt')
        dir_list =  os.listdir(d)
    
        infer_data_file_path = d+'infer_data_file.txt'
        infer_data_file = open(infer_data_file_path, 'w')
    
        for list_index, image_name in enumerate(dir_list):
            #default class is set to cero in all the unclissified images
            infer_data_file.write("%s;%s\n" %(image_name, str(list_index))) 
        
        infer_data_file.close()
                    
        predicted_labels = BOV_test(test_path = d, file_path = infer_data_file_path, show_window = False)[1]
        
        
              
            
        
    click.echo('Infered class for images in directory {} by model {}'.format(d, m))

@cli.command()
@click.option('-m', default='LeNet', help ='Model to Train. Valid names: LeNet, tf_Logistic, Logistic')
@click.option('-d', default='images/train/', help = 'User directory to Train. Defaul: images/train/')

def train(m, d):

    '''
    Train model m with images in directory d. Valid model names: \
    - LeNet \n
    - tf_Logistic\n
    - Logistic
    '''
    if m=='LeNet':
        main_LeNeT(d, mode='train')   
        
    elif m=='tf_Logistic':
        main_tf_logistic(d, 'train')
    
    elif m=='Logistic':
        BOV_train(d)
    
    click.echo('Trained model: {}, with images in directory {}'.format(m, d)) 

@cli.command()
@click.option('-m', default='LeNet', help ='Model to Test. Valid names: LeNet, tf_Logistic, Logistic')
@click.option('-d', default='images/test/', help = 'User directory to Test. Defaul: images/test/')

def test(m, d):

    '''
    Test model m with images in directory d. Valid model names: \
    - LeNet \n
    - tf_Logistic\n
    - Logistic
    '''
    if m=='LeNet':
        main_LeNeT(d, mode='test')   
        
    elif m=='tf_Logistic':
        main_tf_logistic(d, 'test')
    
    elif m=='Logistic':
        phrase, predicted_labels = BOV_test(test_path = d, show_window = False)
        print(phrase)
    
    click.echo('Tested model: {}, with images in directory {}'.format(m, d)) 




###-------------------------------------------------------------------------------------------------------####
###------------------------Download-----------------------------------------------------------------------####

def duplicate_images(img_dir):
    #directory with the images to duplicate up to 60. -finished with /
    img_list_complete = os.listdir(img_dir)
    img_name = '3000'
    n_of_images = len(img_list_complete)
    
    while n_of_images < 60:
        if n_of_images <= 20:
            img_list = img_list_complete.copy()
        else:
            img_list = random.sample(img_list_complete, (60 - n_of_images)//2 + 1)
    
        for img in img_list:
            rot_angle = random.uniform(-15, 15)
            img_read = cv2.imread(img_dir+img)
            img_read = cv2.resize(img_read, (70, 70),interpolation = cv2.INTER_CUBIC)
            rows, cols = (70, 70)
            M = cv2.getRotationMatrix2D((cols/2,rows/2),rot_angle,1)
            img_read = cv2.warpAffine(img_read,M,(cols,rows))
            cv2.imwrite(img_dir+img_name+'.ppm', img_read)
            img_name = '{}'.format(int(img_name) + 1 )
        img_list_complete = os.listdir(img_dir)
        n_of_images = len(img_list_complete)
    

        
def main_download():
    
    DATA_SET_URL = 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'
    image_dir = 'FullIJCNN2013/'
    global TEST_IMG_DIR 
    global TRAIN_IMG_DIR 
    TRAIN_SET_SIZE = 0.8   
    
    url = requests.get(DATA_SET_URL)
    file = zipfile.ZipFile(io.BytesIO(url.content))
    file.extractall(image_dir)
    #remove the direcotry to ensure they are empty
    shutil.rmtree('images/train/')
    shutil.rmtree('images/test/')
    os.mkdir('images/train/')
    os.mkdir('images/test/')
    
    dir_list =  os.listdir(image_dir+image_dir) #list the elements in the downloaded file
    dir_list = [directory for directory in dir_list if not directory.endswith('ppm') and directory.isalnum()] # remove all but the directories of interest
    
    temp_folder = 'temp_folder/'
    os.mkdir(temp_folder)
    
    img_name = '000'
    
    #labels(image_name, class)
    labels = []
    
    for directory in dir_list:
        duplicate_images(image_dir+image_dir +directory+'/')
        img_list = os.listdir(image_dir +image_dir + directory)
        for img in img_list:
            labels.append((img_name, int(directory)))
            shutil.copy(image_dir +image_dir + directory + '/' + img, temp_folder + img_name)        
            img_name = '{:03d}'.format(int(img_name) + 1 )
            
    #randomizar la entrada, mover y remover carpetas
    
    #shffle the labels list        
    random.shuffle(labels)
    
    #train_test_split of sklearn
    
    file_train =  open('label_train.txt', 'w')
    file_test =  open('label_test.txt', 'w')
    
    for count, label in enumerate(labels):
        if count <= len(labels)*TRAIN_SET_SIZE:
            shutil.move(temp_folder +label[0], TRAIN_IMG_DIR)
            file_train.write("%s;%s\n" %label)
        else:
            shutil.move(temp_folder+label[0], TEST_IMG_DIR)
            file_test.write("%s;%s\n" %label)
      
    file_train.close()
    file_test.close()      
    shutil.rmtree(temp_folder)  
    shutil.rmtree(image_dir)
    shutil.move('label_train.txt', 'images/train/label_train.txt')    
    shutil.move('label_test.txt', 'images/test/label_test.txt')     
    


###----------------------------------------------------------------------------------------------------------####
###------------------------SKLEARN LOGISTIC C----------------------------------------------------------------####
#We use a Bag of Visual Words algoritms to train the logistic regression. 
    
    
def BOV_get_files(txt_file_path, image_file_paths):
    '''
    Return a dictionary of 
    
    @arg image_file_paths is the direcory of the images to be loaded\n
    '''
    
    file = open(txt_file_path, 'r')
    img_dict = {'{}'.format(i) : [] for i in range(43)}
    n_images = 0
    for line in file:
        img_name, categ = line.split(';')
        img_file = cv2.imread(image_file_paths+img_name, 0)
        img_file = img_file.astype(np.float32)
        img_dict[categ.rstrip()].append(img_file)
        n_images += 1

    return img_dict, n_images
        
    
def BOV_extract_features(img):
    template_image_path = 'images/train/003'
    template_image = cv2.imread(template_image_path)
    sift_object = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.03, edgeThreshold = 10)
    key_points, descriptors = sift_object.detectAndCompute(img, None)
    template_key, template_descriptor = sift_object.detectAndCompute(img, None)
    if not key_points:
        key_points, descriptors = sift_object.detectAndCompute(template_image, None)
    return [key_points, descriptors]


def BOV_cluster_K_means(descriptor_list, n_clusters, n_images):
    #reshape
    vStack = np.array(descriptor_list[0])
    for remaining in descriptor_list[1:]:
        vStack = np.vstack((vStack, remaining))
    
    
    kmeans_obj = KMeans(n_clusters = n_clusters, n_jobs = 4)
    kmeans_ret = kmeans_obj.fit_predict(vStack.copy())

    mega_histogram = np.array([np.zeros(n_clusters) for i in range(n_images)])
    old_count = 0
    for i in range(n_images):
        l = len(descriptor_list[i]) #number of keypoints
        for j in range(l):
            idx = kmeans_ret[old_count+j]
            mega_histogram[i][idx] += 1
        old_count += l
    
    #save the k_means model to pkl file. 
    joblib.dump(kmeans_obj, 'models/model1/k_means_object.pkl')
    return mega_histogram

    
def BOV_train(train_path =  'images/train/' ):

    file_path = train_path + 'label_train.txt'
    images, n_images = BOV_get_files(file_path, train_path)
    #just in case ^o^    
    indices = []
    
    for key in images:
        if not images[key]:
            indices.append(key)
            
    for empty in indices:
        del images[empty]
        
    descriptor_list = []
    labels = []
    
    for key in images:
        for img_file in images[key]:
            img_file = img_file.astype(np.uint8)
            key_points, descriptors = BOV_extract_features(img_file)
            labels.append(int(key))
            descriptor_list.append(descriptors)
    
    data = list(zip(descriptor_list, labels))
    # I discovered later the shuffle method od scikitlearn... worth trying
    np.random.shuffle(data)
    
    descriptor_list, labels = zip(*data)
    descriptor_list, labels = list(descriptor_list), list(labels)
    mega_histogram = BOV_cluster_K_means(descriptor_list, 200, n_images)
    mega_histogram = normalize(mega_histogram, norm='l2')
    

    log_clf = LogisticRegression(C = 100, solver='saga', max_iter=1000, random_state=42, multi_class='multinomial', penalty = 'l2', n_jobs = 4).fit(mega_histogram, labels)
    
    joblib.dump(log_clf, 'models/model1/logistic_classifier.pkl')
    print("training score : %.3f (%s)" % (log_clf.score(mega_histogram, labels), 'Sklearn logistic reg.'))
    

def BOV_infer(test_img, show_window = True):
    key_points, descriptors = BOV_extract_features(test_img)
    k_means_obj = joblib.load('models/model1/k_means_object.pkl')
    k_means_prediction = k_means_obj.predict(descriptors)
    histogram = np.array([[0 for i in range(200)]])
    
    for index in k_means_prediction:
        histogram[0][index] += 1
        
    histogram = normalize(histogram, norm='l2')
    
    log_clisifier = joblib.load('models/model1/logistic_classifier.pkl')
    label_predicted = log_clisifier.predict(histogram)
    
    if show_window:
        name = 'Category: {}'.format(label_predicted[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(test_img, cmap='gray')
        #ax.annotate(name,  xy=(15, 15), xytext=(18, 18),arrowprops=dict(facecolor='white', shrink=0.05))
        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 20,
        }
        plt.figtext(0.3, 0.5, name, fontdict=font )
        plt.show()
        print('category = ', name)
    
    return label_predicted[0]
    
def BOV_test(test_path = 'images/test/', file_path = 'images/test/label_test.txt', show_window = False):
    images, n_images = BOV_get_files(file_path, test_path)
    
    predicted_labels = []
    true_labels = []
    for key in images:
        for img_file in images[key]:
            img_file = img_file.astype(np.uint8)
            predicted_labels.append(BOV_infer(img_file, show_window=show_window))
            true_labels.append(int(key))  
    phrase = "accuracy score : %.3f (%s)" % (accuracy_score(true_labels, predicted_labels), 'Sklearn logistic reg.')
    return phrase, predicted_labels
    

###-------------------------------------------------------------------------------------------------------------------####
###-------------------------------TF LOGISTIC C-----------------------------------------------------------------------####

def tf_logistic_classifier(x, y, mode):
    
    x = flatten(x["x"])
    mu = 0
    sigma = 0.1
    conv1_w = tf.Variable(tf.truncated_normal(shape=[1024,43], mean = mu, stddev=sigma ), name='conv1_w')
    conv1_b = tf.Variable(tf.zeros(43), name = 'conv1_b')
    
    logits = tf.matmul(x, conv1_w, name='fc2') + conv1_b

    predictions={
            "classes": tf.arg_max(input=logits, dimension=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")  
            }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=y)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss= loss,
                global_step=tf.train.get_global_step()
                )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=y, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main_tf_logistic(path, mode):
    
    if mode=='train':
        
        train_path = path
        #train_path = 'images/train/'
        file_path = 'images/train/label_train.txt'
        X_train, y_train = load_data(train_path, file_path)
    
    elif mode=='test':
        test_path = 'images/test/'
        test_file_path = 'images/test/label_test.txt'
        X_test, y_test = load_data(test_path, test_file_path)
        
    else:
        print('mode must be train or test')
        return 'error: mode'

    #create the estimator
    
    log_classifier = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode: tf_logistic_classifier(features, labels, mode), model_dir='models/model2/log_classifier')
    
    tensor_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensor_to_log, every_n_iter=50)
    
    #train te model
    if mode == 'train':
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train},
            y=y_train,
            batch_size=BATCH_SIZE,
            num_epochs=EPOCHS,
            shuffle=True
            )
        log_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook]
            )
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train},
            y=y_train,
            num_epochs=1,
            shuffle=False
            )
        eval_results = log_classifier.evaluate(input_fn=eval_input_fn)
        print('Results for training: \n', eval_results)        
        
    if mode == 'test':
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_test},
            y=y_test,
            num_epochs=1,
            shuffle=False
            )
        eval_results = log_classifier.evaluate(input_fn=eval_input_fn)
        print('Results for testing: \n', eval_results)
    

def log_model_tf_infer(image_dir, show_window=True):
    ''' 
    infers the class of an image
    '''
    #create file with data, this is needed to use the load_data and BOV_get_file
    #functions
    if os.path.exists(image_dir+'infer_data_file.txt'):
        os.remove(image_dir+'infer_data_file.txt')
    dir_list =  os.listdir(image_dir)
    
    infer_data_file_path = 'images/user/infer_data_file.txt'
    infer_data_file = open(infer_data_file_path, 'w')
    
    for list_index, image_name in enumerate(dir_list):
        #default class is set to cero in all the unclissified images
        infer_data_file.write("%s;%s\n" %(image_name, str(list_index)))
    
    
    infer_data_file.close()
    
    #load data
    X_infer, y_infer = load_data(image_dir, infer_data_file_path)
    
    log_classifier = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode: tf_logistic_classifier(features, labels, mode), model_dir='models/model2/log_classifier')
        
    infer_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_infer},
            num_epochs=1,
            shuffle=False
            )
    infer_results = log_classifier.predict(input_fn=infer_input_fn)
    
    i = 0
    class_list=[]
    for result in infer_results:
        class_id = result['classes']
        probability = result['probabilities'][class_id]
        name = dir_list[y_infer[i]]
        class_list.append((name, class_id))
        print('image {} is of class {} ({:.2f}%)'.format(name, class_id, 100*probability))
        i+=1
    
    if show_window:
        i = 0
        for label in class_list:
            name = 'Category: {}'.format(label[1])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(X_infer[i], cmap='gray')
            i += 1
            #ax.annotate(name,  xy=(15, 15), xytext=(18, 18),arrowprops=dict(facecolor='white', shrink=0.05))
            font = {'family': 'serif',
                    'color':  'darkred',
                    'weight': 'normal',
                    'size': 20,
                    }
            plt.figtext(0.3, 0.5, name, fontdict=font )
            plt.show()      

###-----------------------------------------------------------------------------------------------------------####
###------------------------TF - LENET-5-----------------------------------------------------------------------####
    
def LeNet_model_tf(x, y, mode):
    
    '''
    implementatio of the LeNet-5 CNN of Yann LeCun.
    '''

    x = tf.reshape(x["x"], [-1, 32, 32, 1])
    #reshape the features:
    mu = 0
    sigma = 0.1
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5,5,1,6], mean = mu, stddev=sigma ), name='conv1_w')
    conv1_b = tf.Variable(tf.zeros(6), name = 'conv1_b')
    conv1 = tf.add(tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='VALID', name='conv1'), conv1_b)

    conv1 = tf.nn.relu(conv1)
    
    #  Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name ='pool_1' )
    
    #  Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma), name= 'conv2_w')
    conv2_b = tf.Variable(tf.zeros(16), name= 'conv2_b')
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID', name='conv2') + conv2_b
    #  Activation.
    conv2 = tf.nn.relu(conv2)

    #  Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID', name='pool_2') 
    
    #  Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    #  Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma), name='fc1_w')
    fc1_b = tf.Variable(tf.zeros(120), name='fc1_b')
    fc1 = tf.matmul(fc1,fc1_w, name='fc1') + fc1_b
    
    #  Activation.
    fc1 = tf.nn.relu(fc1)

    #  Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma), name='fc2_w')
    fc2_b = tf.Variable(tf.zeros(84), name='fc2_b')
    fc2 = tf.matmul(fc1,fc2_w, name='fc2') + fc2_b
    #  Activation.
    fc2 = tf.nn.relu(fc2)
    
    #  Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,43), mean = mu , stddev = sigma), name='fc3_w')
    fc3_b = tf.Variable(tf.zeros(43), name='fc3_b')
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    
    predictions={
            "classes": tf.arg_max(input=logits, dimension=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")  
            }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=y)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss= loss,
                global_step=tf.train.get_global_step()
                )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=y, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    
def main_LeNeT(path, mode):
    
    if mode=='train':
        
        train_path = path
        #train_path = 'images/train/'
        file_path = 'images/train/label_train.txt'
        X_train, y_train = load_data(train_path, file_path)
    
    elif mode=='test':
        test_path = 'images/test/'
        test_file_path = 'images/test/label_test.txt'
        X_test, y_test = load_data(test_path, test_file_path)
        
    else:
        print('mode must be train or test')
        return 'error: mode'

    #create the estimator
    
    LeNet_classifier = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode: LeNet_model_tf(features, labels, mode), model_dir='models/model3/lenet')
    
    tensor_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensor_to_log, every_n_iter=50)
    
    #train te model
    if mode == 'train':
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train},
            y=y_train,
            batch_size=BATCH_SIZE,
            num_epochs=20,
            shuffle=True
            )
        LeNet_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook]
            )
        
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train},
            y=y_train,
            num_epochs=1,
            shuffle=False
            )
        eval_results = LeNet_classifier.evaluate(input_fn=eval_input_fn)
        print('Results for training: \n', eval_results)
    if mode == 'test':
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_test},
            y=y_test,
            num_epochs=1,
            shuffle=False
            )
        eval_results = LeNet_classifier.evaluate(input_fn=eval_input_fn)
        print('Results for training: \n', eval_results)
    
def LeNet_infer(image_dir, show_window=True):
    
    #create file with data, this is needed to use the load_data and BOV_get_file
    #functions
    if os.path.exists(image_dir+'infer_data_file.txt'):
        os.remove(image_dir+'infer_data_file.txt')
    dir_list =  os.listdir(image_dir)
    
    infer_data_file_path = 'images/user/infer_data_file.txt'
    infer_data_file = open(infer_data_file_path, 'w')
    
    for list_index, image_name in enumerate(dir_list):
        #default class is set to cero in all the unclissified images
        infer_data_file.write("%s;%s\n" %(image_name, str(list_index)))
    
    
    infer_data_file.close()
    
    #load data
    X_infer, y_infer = load_data(image_dir, infer_data_file_path)
    
    LeNet_classifier = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode: LeNet_model_tf(features, labels, mode), model_dir='models/model3/lenet')
        
    infer_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_infer},
            num_epochs=1,
            shuffle=False
            )
    infer_results = LeNet_classifier.predict(input_fn=infer_input_fn)
    
    i = 0
    class_list=[]
    for result in infer_results:
        class_id = result['classes']
        probability = result['probabilities'][class_id]
        name = dir_list[y_infer[i]]
        class_list.append((name, class_id))
        print('image {} is of class {} ({:.2f}%)'.format(name, class_id, 100*probability))
        i+=1
        #for i, class_ in enumerate([result['classes']]):
           # print('{} {}'.format(i+1, class_))
    
    if show_window:
        i = 0
        for label in class_list:
            name = 'Category: {}'.format(label[1])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(X_infer[i], cmap='gray')
            i += 1
            #ax.annotate(name,  xy=(15, 15), xytext=(18, 18),arrowprops=dict(facecolor='white', shrink=0.05))
            font = {'family': 'serif',
                    'color':  'darkred',
                    'weight': 'normal',
                    'size': 20,
                    }
            plt.figtext(0.3, 0.5, name, fontdict=font )
            plt.show()      


def load_data(image_path, txt_path):
    img_dict, n_images = BOV_get_files(txt_path, image_path)
    
    #reshape image to 32x32
    X = []
    y = []
    for key in img_dict:
        for img in img_dict[key]:
            reshaped_img = cv2.resize(img, (32, 32),interpolation = cv2.INTER_CUBIC)
            X.append(reshaped_img)#.reshape(32, 32, 1))
            y.append(int(key))
   
    #array_X = np.array(X[0])
    #for img in X[1:]:
        #array_X = np.vstack((array_X, img))
    X = np.array(X)    
    y = np.array(y)    
    X, y = shuffle(X, y, random_state=0)
    
    return X, y

if __name__ == '__main__':
    cli()
#################################################################################################
#END



# TODO: corregir infer, o BOV_get_files, due to the fact that if /user/has mor than 43 elements the method may crash. 



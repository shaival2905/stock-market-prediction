# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:14:43 2018

@author: shaival
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import features
from sklearn import preprocessing


def read_dataset():
    #X=[]
    #y=[]
    #simavg, weiavg, ROC, stoK, stoD, rsi, MACD, WR, ado, CCI, label = features.features("yahoostock.csv")
    X,y = features.features("yahoostock.csv")
    """X.append(simavg)
    X.append(weiavg)
    X.append(ROC)
    X.append(stoK)
    X.append(stoD)
    X.append(rsi)
    X.append(MACD)
    X.append(WR)
    X.append(ado)
    X.append(CCI)
    y.append(label)
 
    #print(CCI)
    X = np.array(X)
    X = X.transpose()"""
    #print(X[:,9])
    X = X[27:5010]
    print(X.shape)

    X = pd.DataFrame(X)
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled)
    X = X.values
    
    #y = np.array(y)
    
    #y = y.transpose()
    y = y[27:5010]
    y = pd.DataFrame(y)

    #print(y)
    
    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y.values.ravel())
    y = encoder.transform(y.values.ravel())
    Y = one_hot_encode(y)
    #print(X.shape)
    return (X,Y)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode



# Read the dataset
X, Y = read_dataset()
#print(Y)
#shuffle the dataset to mix up the rows

#X, Y = shuffle(X, Y, random_state=1)

# Convert the dataset into train and test part
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.20, random_state=415)

# Inpect the shape of the training and testing
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

# Define the important parameters and variable to work with the tensors
learning_rate = 0.5
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = 2
model_path = "C:\\Users\\shaiv\\Documents\\Stock market\\TensorFlow\\NMI"

# Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
w = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


def multilayer_perceptron(x, weights, biases):
    
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)
    
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

    
weights = {
        'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
    }

biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
        'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
        'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
        'out': tf.Variable(tf.truncated_normal([n_class]))
    }

# Initialize all the variables

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Call your model defined

y = multilayer_perceptron(x, weights, biases)

# Define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
print(cost_function.shape)
sess = tf.Session()
sess.run(init)

# Calculate the cost and the accuracy for each epock
mse_history = []
accuracy_history = []
print("trainy: ",train_y.shape)
print("trainx: ",train_x.shape)
for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)
    
    print('epoch: ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mse_, " - Train Accuracy: ", accuracy)
    
save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)
print("Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))
#plot mse and accuracy graph

plt.plot(mse_history, 'r')
plt.show()
plt.plot(accuracy_history)
plt.show()

#print the final accuracy

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Print the final mean square error

pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))
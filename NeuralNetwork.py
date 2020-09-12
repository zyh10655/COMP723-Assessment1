# Adopted from https://pythonprogramming.net/train-test-tensorflow-deep-learning-tutorial/
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# train_x, train_y, test_x, test_y = create_feature_sets_and_labels('data/sentiment2/pos.txt', 'data/sentiment2/neg.txt')
pickle_in = open('assessment model.pickle','rb')
train_x, train_y, test_x, test_y = pickle.load(pickle_in)
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500
n_nodes_hl4 = 1500
n_nodes_hl5 = 1500
# n_nodes_hl6 = 1500
# n_nodes_hl7 = 1500
# n_nodes_hl8 = 1500
# n_nodes_hl9 = 1500
# n_nodes_hl10 = 1500
# 45.51
n_classes = 5
batch_size = 100
hm_epochs = 5

x = tf.placeholder('float')
y = tf.placeholder('float')
# Construct the NN by creating individual layers. The first hidden layer is the input layer.
hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum': n_nodes_hl3,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

hidden_4_layer = {'f_fum': n_nodes_hl4,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl4]))}

hidden_5_layer = {'f_fum': n_nodes_hl5,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl5]))}

# hidden_6_layer = {'f_fum': n_nodes_hl6,
#                   'weight': tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6])),
#                   'bias': tf.Variable(tf.random_normal([n_nodes_hl6]))}
#
# hidden_7_layer = {'f_fum': n_nodes_hl7,
#                   'weight': tf.Variable(tf.random_normal([n_nodes_hl6, n_nodes_hl7])),
#                   'bias': tf.Variable(tf.random_normal([n_nodes_hl7]))}
#
# hidden_8_layer = {'f_fum': n_nodes_hl8,
#                   'weight': tf.Variable(tf.random_normal([n_nodes_hl7, n_nodes_hl8])),
#                   'bias': tf.Variable(tf.random_normal([n_nodes_hl8]))}
#
# hidden_9_layer = {'f_fum': n_nodes_hl9,
#                   'weight': tf.Variable(tf.random_normal([n_nodes_hl8, n_nodes_hl9])),
#                   'bias': tf.Variable(tf.random_normal([n_nodes_hl9]))}
#
# hidden_10_layer = {'f_fum': n_nodes_hl10,
#                   'weight': tf.Variable(tf.random_normal([n_nodes_hl9, n_nodes_hl10])),
#                   'bias': tf.Variable(tf.random_normal([n_nodes_hl10]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl5, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# Construct the model by summing the previous inputs and passing it through a nonlinear activation function
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weight']), hidden_4_layer['bias'])
    l4 = tf.nn.relu(l4)

    l5 = tf.add(tf.matmul(l4, hidden_4_layer['weight']), hidden_5_layer['bias'])
    l5 = tf.nn.relu(l5)

    # l6 = tf.add(tf.matmul(l5, hidden_5_layer['weight']), hidden_6_layer['bias'])
    # l6 = tf.nn.relu(l6)
    #
    # l7 = tf.add(tf.matmul(l6, hidden_6_layer['weight']), hidden_7_layer['bias'])
    # l7 = tf.nn.relu(l7)
    #
    # l8 = tf.add(tf.matmul(l7, hidden_7_layer['weight']), hidden_8_layer['bias'])
    # l8 = tf.nn.relu(l8)
    #
    # l9 = tf.add(tf.matmul(l8, hidden_8_layer['weight']), hidden_9_layer['bias'])
    # l9 = tf.nn.relu(l9)
    #
    # l10 = tf.add(tf.matmul(l9, hidden_9_layer['weight']), hidden_10_layer['bias'])
    # l10 = tf.nn.relu(l10)

    output = tf.matmul(l5, output_layer['weight']) + output_layer['bias']


    return output


# Train the network by calculating the error and adjusting the weights hm_epochs number of times.
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
#100 0.45 0.465:0.32
    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)


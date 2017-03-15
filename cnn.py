import helper
import numpy as np
import pickle
import tensorflow as tf

from sys import exit

class CNN(object):
    def __init__(self, epochs=40, batch_size=256, keep_probability=.8):
        self.batch_size = batch_size
        self.epochs = epochs
        self.keep_probability = keep_probability

    def conv_net(self, x, keep_prob):
        conv_layer_1 = self.conv2d_maxpool(x, 64, (3, 3), (1, 1), (2, 2), (2, 2))
        conv_layer_2 = self.conv2d_maxpool(conv_layer_1, 64, (3, 3), (2, 2), \
            (2, 2), (2, 2))
        conv_layer_3 = self.conv2d_maxpool(conv_layer_2, 64, (3, 3), (1, 1), \
            (2, 2), (2, 2))

        cnn = self.flatten(conv_layer_3)
        cnn = self.fully_conn(cnn, 512)
        cnn = self.fully_conn(cnn, 256)
        return self.output(cnn, 10)

    def conv2d_maxpool(self, x_tensor, conv_num_outputs, conv_ksize, \
        conv_strides, pool_ksize, pool_strides):
        weight = tf.Variable(tf.random_normal([*conv_ksize, \
            int(x_tensor.shape[3]), conv_num_outputs], stddev = 0.1))
        bias = tf.Variable(tf.zeros(conv_num_outputs))

        cnn = tf.nn.conv2d(x_tensor, weight, [1, *conv_strides, 1], \
            padding='SAME')
        cnn = tf.nn.bias_add(cnn, bias)
        cnn = tf.nn.relu(cnn)
        cnn = tf.nn.max_pool(cnn, [1, *conv_ksize, 1], [1, *pool_strides, 1], \
            padding='SAME')

        return cnn

    def flatten(self, x_tensor):
        return tf.contrib.layers.flatten(x_tensor)

    def fully_conn(self, x_tensor, num_outputs):
        weight = tf.Variable(tf.random_normal((int(x_tensor.shape[1]), \
            num_outputs), stddev = 0.1))
        bias = tf.Variable(tf.zeros(num_outputs))

        fcl = tf.add(tf.matmul(x_tensor, weight), bias)
        fcl = tf.nn.relu(fcl)

        return fcl

    def initialize(self):
        self.x = cnn_helper.neural_net_image_input((32, 32, 3))
        self.y = cnn_helper.neural_net_label_input(10)
        self.keep_prob = cnn_helper.neural_net_keep_prob_input()

        self.logits = cnn_helper.conv_net(self.x, self.keep_prob)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits\
            (logits = self.logits, labels = self.y))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1),
            tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), \
                name='accuracy')

    def neural_net_image_input(self, image_shape):
        return tf.placeholder(tf.float32, (None, *image_shape), name = "x")

    def neural_net_keep_prob_input(self):
        return tf.placeholder(tf.float32, name="keep_prob")

    def neural_net_label_input(self, n_classes):
        return tf.placeholder(tf.float32, (None, n_classes), name = "y")

    def normalize(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def one_hot_encode(self, x):
        return np.identity(10)[x]

    def output(self, x_tensor, num_outputs):
        weight = tf.Variable(tf.random_normal((int(x_tensor.shape[1]), \
            num_outputs), stddev = 0.1))
        bias = tf.Variable(tf.zeros(num_outputs))

        return tf.add(tf.matmul(x_tensor, weight), bias)

    def preprocess(self):
        helper.preprocess_and_save_data('cifar-10-batches-py', self.normalize,
            self.one_hot_encode)

    def print_stats(self, session, feature_batch, label_batch, cost, accuracy):
        _ = {
                self.x: feature_batch,
                self.y: label_batch,
                self.keep_prob: 1.0
            }
        loss = session.run(cost, feed_dict = _)

        f = {
                self.x: self.valid_features,
                self.y: self.valid_labels,
                self.keep_prob: 1.0
            }
        validation_accuracy = session.run(accuracy, feed_dict = f)

        print('Loss:', loss, 'Validation Accuracy:', validation_accuracy)

    def reset_graph(self):
        tf.reset_default_graph()

    def train_neural_network(self, session, optimizer, keep_probability, \
        feature_batch, label_batch):
        _ = {
                self.x: feature_batch,
                self.y: label_batch,
                self.keep_prob: keep_probability
            }
        session.run(optimizer, feed_dict = _)

    def state_valids(self):
        self.valid_features, self.valid_labels =  pickle.load(open(\
            'preprocess_validation.p', mode='rb'))

if __name__ == '__main__':
    cnn_helper = CNN()
    cnn_helper.preprocess()
    cnn_helper.state_valids()

    cnn_helper.reset_graph()
    cnn_helper.initialize()

    print('Training Image Classifier')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(cnn_helper.epochs):
            for batch_i in range(1, 6):
                for batch_features, batch_labels in \
                    helper.load_preprocess_training_batch(batch_i,
                        cnn_helper.batch_size):

                    cnn_helper.train_neural_network(sess, cnn_helper.optimizer,
                        cnn_helper.keep_probability,
                        batch_features, batch_labels)

                    print('Epoch {:>2}, CIFAR-10 Batch {}:  '\
                        .format(epoch + 1, batch_i), end='')
                    cnn_helper.print_stats(sess, batch_features, \
                        batch_labels, cnn_helper.cost, cnn_helper.accuracy)

    print('Saving Model')
    save_path = tf.train.Saver().save(sess, save_model_path)

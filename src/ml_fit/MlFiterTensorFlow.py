# -*- encoding:utf-8 -*-
"""
tf

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division

import subprocess
import webbrowser
from abc import ABCMeta, abstractmethod

import ZEnv
import ZLog
import ZCommonUtil
import numpy as np
import pandas as pd
import six
import tensorflow as tf
import time
from sklearn.cross_validation import KFold
from PIL import Image

K_LOG_FILE = ZEnv.g_project_root + '/data/cache/tensorflow_logs'
K_CNN_IMG_SIZE_M = 28
K_CNN_IMG_SIZE_L = 224


def show_log_board():
    """
    打开可视化log试图
    :return:
    """
    p = subprocess.Popen('tensorboard --logdir=' + K_LOG_FILE, shell=True)
    time.sleep(3)
    webbrowser.open('http://localhost:6006/', new=0, autoraise=True)
    return p


class BaseTF(six.with_metaclass(ABCMeta, object)):
    @abstractmethod
    def fit(self):
        pass

    def predict(self, will_pre):
        if not hasattr(self, 'pred'):
            raise RuntimeError('not hasattr pred')

        with tf.Session() as sess:
            sess.run(self.init)
            return sess.run(self.pred, feed_dict={self.x: will_pre})

    @classmethod
    def do_tf_tt(cls, x, y, n_folds=10, **kwargs):
        """
        如果需要扩张除init四个外的参数, 子类继续自己扩张把
        :param x:
        :param y:
        :param n_folds:
        :return:
        """
        kf = KFold(len(y), n_folds=n_folds, shuffle=True)
        acs = list()
        for i, (train_index, test_index) in enumerate(kf):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            m_l__t_f = cls(x_train, y_train, x_test, y_test, **kwargs)
            ac = m_l__t_f.fit()
            if ac is not None:
                acs.append(ac)
        if len(acs) > 0:
            ZLog.info('acs mean = {}'.format(np.array(acs).mean()))


class LogisticTF(BaseTF):
    """
        简单逻辑分类
    """

    def __init__(self, x_train, y_train, x_test=None, y_test=None, batch_size=-1, learning_rate=0.01,
                 training_epochs=25):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size

        if y_test is not None:
            y_test = pd.get_dummies(y_test).as_matrix()
        y_train = pd.get_dummies(y_train).as_matrix()

        self.x = tf.placeholder("float", [None, x_train.shape[1]])
        self.y = tf.placeholder("float", [None, y_train.shape[1]])

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def fit(self):
        w = tf.Variable(tf.zeros([self.x_train.shape[1], self.y_train.shape[1]]))
        b = tf.Variable(tf.zeros([self.y_train.shape[1]]))

        activation = tf.nn.softmax(tf.matmul(self.x, w) + b)
        cost = -tf.reduce_sum(self.y * tf.log(activation))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        self.init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(self.init)
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                if self.batch_size == -1:
                    self.batch_size = int(self.x_train.shape[0] / 10)
                total_batch = int(self.x_train.shape[0] / self.batch_size)
                for i in range(total_batch):
                    batch_xs = self.x_train[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_ys = self.y_train[i * self.batch_size: (i + 1) * self.batch_size]
                    sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys})
                    avg_cost += sess.run(cost, feed_dict={self.x: batch_xs, self.y: batch_ys}) / total_batch
            ZLog.info("Optimization Finished!")

            self.pred = tf.argmax(activation, 1)
            if self.x_test is not None:
                correct_prediction = tf.equal(self.pred, tf.argmax(self.y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                ZLog.info("Accuracy:" + str(accuracy.eval({self.x: self.x_test, self.y: self.y_test})))


class KnnTF(BaseTF):
    """
        简单knn分类
    """

    def __init__(self, x_train, y_train, x_test=None, y_test=None):
        self.xtr = tf.placeholder("float", [None, x_train.shape[1]])
        self.xte = tf.placeholder("float", [x_train.shape[1]])

        if y_test is not None:
            y_test = pd.get_dummies(y_test).as_matrix()
        y_train = pd.get_dummies(y_train).as_matrix()

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def predict(self, will_pre):
        if not hasattr(self, 'pred'):
            raise RuntimeError('not hasattr pred')

        with tf.Session() as sess:
            sess.run(self.init)
            nn_index = sess.run(self.pred, feed_dict={self.xtr: self.x_train, self.xte: will_pre})
            return np.argmax(self.y_train[nn_index])

    def fit(self):
        self.init = tf.initialize_all_variables()

        distance = tf.reduce_sum(tf.abs(tf.add(self.xtr, tf.neg(self.xte))), reduction_indices=1)
        self.pred = tf.arg_min(distance, 0)
        accuracy = 0.

        if self.x_test is None:
            return

        with tf.Session() as sess:
            sess.run(self.init)

            for i in range(len(self.x_test)):
                nn_index = sess.run(self.pred, feed_dict={self.xtr: self.x_train, self.xte: self.x_test[i, :]})
                # print "Test", i, "Prediction:", np.argmax(self.y_train[nn_index]), \
                #     "True Class:", np.argmax(self.y_test[i])
                if np.argmax(self.y_train[nn_index]) == np.argmax(self.y_test[i]):
                    accuracy += 1. / len(self.x_test)
            print "Done!"
            print "Accuracy:", accuracy
            return accuracy


class MnnTF(BaseTF):
    """
        普通多层神经网络
    """

    def __init__(self, x_train, y_train, x_test=None, y_test=None, n_hidden_1=256, n_hidden_2=256, batch_size=100,
                 learning_rate=0.01, training_epochs=30,
                 display_step=-1):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        if display_step == -1:
            display_step = int(training_epochs / 26)
        self.display_step = display_step

        self.n_hidden_1 = n_hidden_1  # 1st layer num features
        self.n_hidden_2 = n_hidden_2  # 2nd layer num features

        y_train = pd.get_dummies(y_train).as_matrix()
        if y_test is not None:
            y_test = pd.get_dummies(y_test).as_matrix()

        self.n_input = x_train.shape[1]
        self.n_classes = y_train.shape[1]

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def multilayer_perceptron(self, p_x, _weights, _biases):
        # Hidden layer with RELU activation
        layer_1 = tf.nn.relu(tf.add(tf.matmul(p_x, _weights['h1']), _biases['b1']))
        # Hidden layer with RELU activation
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
        return tf.matmul(layer_2, _weights['out']) + _biases['out']

    def fit(self):
        weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        mul_predict = self.multilayer_perceptron(self.x, weights, biases)
        # Softmax loss
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(mul_predict, self.y))
        # Adam Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        self.init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(self.init)
            # Training cycle
            total_batch = int(len(self.x_train) / self.batch_size) + 1
            tf.train.SummaryWriter(K_LOG_FILE, graph=sess.graph)
            for epoch in range(self.training_epochs):
                avg_cost = 0.

                perm = np.arange(len(self.x_train))
                np.random.shuffle(perm)
                self.x_train = self.x_train[perm]
                self.y_train = self.y_train[perm]

                for i in range(total_batch):
                    batch_xs = self.x_train[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_ys = self.y_train[i * self.batch_size: (i + 1) * self.batch_size]
                    # Fit training using batch data
                    sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys})
                    # Compute average loss
                    avg_cost += sess.run(cost, feed_dict={self.x: batch_xs, self.y: batch_ys}) / total_batch

                if epoch % self.display_step == 0:
                    print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)
            ZLog.info("Optimization Finished!")

            self.pred = tf.argmax(mul_predict, 1)

            if self.x_test is not None:
                correct_prediction = tf.equal(tf.argmax(mul_predict, 1), tf.argmax(self.y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                ac = accuracy.eval({self.x: self.x_test, self.y: self.y_test})
                ZLog.info("Accuracy:" + str(ac))
                return ac


class TensorBatchGen(object):
    """
        feed cnn需要的img
    """

    def __init__(self, train, val, n_classes, img_root_dir, resize, one_hot=True, one_hot_offset=0):
        """

        Parameters
        ----------
        train
        val
        n_classes
        resize
        one_hot
        one_hot_offset: 记录y的class的offset如果从0开始mark就是0，1开始one_hot_offset=1
        img_root_dir: 文件的root dir如果train文件中是绝对路径，这里''就可以了
        """
        self.train_np = pd.read_csv(train, sep=' ', header=None).values
        self.val_np = pd.read_csv(val, sep=' ', header=None).values
        self.resize = resize
        self.one_hot = one_hot
        self.n_classes = n_classes
        self.one_hot_offset = one_hot_offset
        self.img_root_dir = img_root_dir
        self.img_cache = dict()

    def gen_shuffle_train_batch(self, batch_size):
        self.step = 1
        self.batch_size = batch_size
        np.random.shuffle(self.train_np)

    def gen_val_batch(self, batch_size=-1):
        if batch_size == -1:
            self.batch_size = len(self.val_np)
            self.step = 1
        return self.next_batch(train=False)

    def dense_to_one_hot(self, labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def next_batch(self, batch_size=-1, train=True):
        dmg_np = self.train_np if train else self.val_np
        step = self.step
        if batch_size == -1:
            batch_size = self.batch_size
        else:
            batch_size = batch_size

        start = (step - 1) * batch_size
        if start >= len(dmg_np):
            self.gen_shuffle_train_batch(batch_size)
            return self.next_batch(batch_size=batch_size, train=train)

        end = step * batch_size
        if end > len(dmg_np):
            end = len(dmg_np)

        img_data = None
        label = list()
        for ind in np.arange(start, end):
            img_ind = dmg_np[ind]

            img_key = img_ind[0] + str(img_ind[1])

            img_path = self.img_root_dir + img_ind[0]
            img_class = img_ind[1] - self.one_hot_offset

            try:
                if img_key in self.img_cache:
                    """
                        从缓存加载已经加载过了的
                    """
                    img = self.img_cache[img_key]
                else:
                    img = Image.open(img_path)
                    img = img.resize((self.resize, self.resize))
                    self.img_cache[img_key] = img

                if img_data is None:
                    img_data = np.asarray(img, dtype="int32")
                    img_data = img_data.reshape(-1, img_data.shape[0], img_data.shape[1], img_data.shape[2])
                else:
                    img_data = np.concatenate((img_data,
                                               np.asarray(img, dtype="int32").reshape(-1, img_data.shape[1],
                                                                                      img_data.shape[2],
                                                                                      img_data.shape[3])),
                                              axis=0)
            except Exception:
                # 过滤非rgb的也可通过img.mode判断
                continue
            label.append(img_class)
        self.step += 1
        label = np.array(label)
        if self.one_hot:
            """
                这里不能用pd.get_dummies因为不能保证抽样到所有classes
            """
            label = self.dense_to_one_hot(label, self.n_classes)
            # label = pd.get_dummies(np.array(label)).as_matrix()
        images = img_data.reshape(img_data.shape[0],
                                  img_data.shape[1] * img_data.shape[2] * img_data.shape[3])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        # label = label.astype(np.float32)
        return images, label


class CnnTF(BaseTF):
    """
        卷积神经网络
    """

    def __init__(self, train_path, test_path, n_classes, img_root_dir, one_hot_offset=0, channel_cnt=3,
                 img_size=K_CNN_IMG_SIZE_M, batch_size=256, learning_rate=0.01, training_iters=100000,
                 data_provide=None):
        self.channel_cnt = channel_cnt
        """
            TensorBatchGener也同样使用K_CNN_IMG_SIZE如果要概需要同步
        """
        self.img_size = img_size
        self.dropout = 0.8
        # self.n_classes = len(np.unique(y_train))
        self.n_classes = n_classes

        self.checkpoint_path = ZEnv.g_project_root + '/data/tensor/tf_model.ckpt'
        ZCommonUtil.ensure_dir(self.checkpoint_path)
        self.n_input = self.img_size * self.img_size * self.channel_cnt

        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size

        """
            整个训练过程diplay20次
        """
        self.display_step = int(training_iters / batch_size / 20)
        """
            整个训练过程save 3次
        """
        self.save_step = int(training_iters / batch_size / 3)

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        if data_provide is None:
            self.batch_gen = TensorBatchGen(train_path,
                                            test_path,
                                            n_classes=n_classes,
                                            img_root_dir=img_root_dir,
                                            resize=self.img_size,
                                            one_hot_offset=one_hot_offset)
        else:
            if not hasattr(data_provide, 'next_batch'):
                raise RuntimeError('data_provide need only one func next_batch!')
            self.batch_gen = data_provide

    def _conv2d(self, img, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1],
                                                      padding='SAME'), b))

    def _max_pool(self, img, k):
        return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def _conv_net(self, _x, _weights, _biases, _dropout):
        _x = tf.reshape(_x, shape=[-1, self.img_size, self.img_size, self.channel_cnt])

        conv1 = self._conv2d(_x, _weights['wc1'], _biases['bc1'])
        conv1 = self._max_pool(conv1, k=2)
        conv1 = tf.nn.dropout(conv1, _dropout)

        conv2 = self._conv2d(conv1, _weights['wc2'], _biases['bc2'])
        conv2 = self._max_pool(conv2, k=2)
        conv2 = tf.nn.dropout(conv2, _dropout)

        dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
        dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
        return out

    def fit(self):
        weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, self.channel_cnt, 32])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # 每一次池化size减半
            'wd1': tf.Variable(tf.random_normal([int(self.img_size / 2 / 2 * self.img_size / 2 / 2 * 64), 1024])),
            'out': tf.Variable(tf.random_normal([1024, self.n_classes]))
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        pred = self._conv_net(self.x, weights, biases, self.keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver(tf.all_variables())

        tf.scalar_summary("loss", cost)
        tf.scalar_summary("accuracy", accuracy)
        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            step = 1

            summary_writer = tf.train.SummaryWriter(K_LOG_FILE, graph=sess.graph)

            if hasattr(self.batch_gen, 'gen_shuffle_train_batch'):
                self.batch_gen.gen_shuffle_train_batch(self.batch_size)

            while step * self.batch_size < self.training_iters:
                batch_xs, batch_ys = self.batch_gen.next_batch(self.batch_size)
                sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: self.dropout})

                summary_str = sess.run(summary_op, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                              self.keep_prob: 1.})
                summary_writer.add_summary(summary_str, step)

                if step % self.display_step == 0:
                    acc = sess.run(accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1.})
                    loss = sess.run(cost, feed_dict={self.x: batch_xs, self.y: batch_ys, self.keep_prob: 1.})
                    print "Iter " + str(step) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

                if step % self.save_step == 0 or ((step + 1) * self.batch_size) >= self.training_iters:
                    saver.save(sess, self.checkpoint_path, global_step=step)
                    print 'save session step {} has done'.format(step)

                step += 1
            print "Optimization Finished!"

            if hasattr(self.batch_gen, 'step'):
                self.batch_gen.step = 1
                val_xs, val_ys = self.batch_gen.gen_val_batch()
                print "Testing Accuracy:", sess.run(accuracy, feed_dict={self.x: val_xs,
                                                                         self.y: val_ys,
                                                                         self.keep_prob: 1.})



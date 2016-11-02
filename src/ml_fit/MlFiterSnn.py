# -*- encoding:utf-8 -*-
"""
简单低效神经网络测试使用
主要作用是查看Loss等值的
变化率可以灵活测试使用

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division

import ZLog
import numpy as np
from sklearn.cross_validation import KFold


class SnnClass(object):
    def __init__(self, x_train, y_train, x_test, y_test, nn_hdim=3, epsilon=0.01, reg_lambda=0.01,
                 num_passes=20000, print_loss=False):
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda
        self.x_train = x_train
        self.y_train = y_train.astype(int)
        self.x_test = x_test
        self.y_test = y_test.astype(int)
        self.nn_hdim = nn_hdim
        self.num_passes = num_passes
        self.print_loss = print_loss

        self.model = None

    def calculate_loss(self, model):
        w1, b1, w2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation to calculate our predictions
        z1 = self.x_train.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(self.num_examples), self.y_train])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        return 1. / self.num_examples * data_loss

    def predict(self, x):
        if self.model is None:
            raise ValueError('self.model is None!!!')

        model = self.model
        w1, b1, w2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return np.argmax(probs, axis=1)

    def build_model(self):
        num_passes = self.num_passes
        print_loss = self.print_loss

        num_examples = len(self.x_train)
        self.num_examples = num_examples
        nn_input_dim = self.x_train.shape[1]
        nn_output_dim = len(np.unique(self.y_train))

        np.random.seed(0)
        w1 = np.random.randn(nn_input_dim, self.nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, self.nn_hdim))
        w2 = np.random.randn(self.nn_hdim, nn_output_dim) / np.sqrt(self.nn_hdim)
        b2 = np.zeros((1, nn_output_dim))

        model = {}
        # Gradient descent. For each batch...
        for i in xrange(0, num_passes):
            # Forward propagation
            z1 = self.x_train.dot(w1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(w2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(num_examples), self.y_train] -= 1
            d_w2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(w2.T) * (1 - np.power(a1, 2))
            d_w1 = np.dot(self.x_train.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            d_w2 += self.reg_lambda * w2
            d_w1 += self.reg_lambda * w1

            # Gradient descent parameter update
            w1 += -self.epsilon * d_w1
            b1 += -self.epsilon * db1
            w2 += -self.epsilon * d_w2
            b2 += -self.epsilon * db2

            # Assign new parameters to the model
            model = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" % (i, self.calculate_loss(model))
        return model

    def fit(self):
        self.model = self.build_model()

        accuracy = 0.
        for ind in range(len(self.x_test)):
            pred = self.predict(self.x_test[ind, :])
            if pred == self.y_test[ind]:
                accuracy += 1. / len(self.x_test)
        print "Done!"
        print "Accuracy:", accuracy
        return accuracy

    @classmethod
    def do_snn_tt(cls, x, y, n_folds=10, nn_hdim=3, num_passes=20000, print_loss=False):
        kf = KFold(len(y), n_folds=n_folds, shuffle=True)
        acs = list()
        for i, (train_index, test_index) in enumerate(kf):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            m_l__t_f = cls(x_train, y_train, x_test, y_test, nn_hdim=nn_hdim, num_passes=num_passes,
                           print_loss=print_loss)
            accuracy = m_l__t_f.fit()
            acs.append(accuracy)
        ZLog.info('accuracys mean = {}'.format(np.array(acs).mean()))

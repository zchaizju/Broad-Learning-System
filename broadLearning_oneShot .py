# This code is a one-shot implementation of Broad-Learning-System
# Cite https://github.com/zchaizju/Broad-Learning-System/ and \
# Chen C L P, Liu Z. Broad learning system: an effective and efficient incremental learning system without the need for deep architecture[J]. \
# IEEE transactions on neural networks and learning systems, 2018, 29(1): 10-24.

# Author: Chai Zheng @ ZJU
# Email: zchaizju@gmail.com

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class BroadNet():
    def __init__(self, N1, N2, N3, C):
        # N1, number of windows
        # N2, number of features each window
        # N3: number of enhanced nodes
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.C = C

    def preprocess(self, data):
        scaler = preprocessing.StandardScaler().fit(data)
        normalized_x_train = scaler.transform(data)
        return normalized_x_train, scaler

    def oneHot(self, label):
        enc = OneHotEncoder()
        enc.fit(label)
        one_hot_label = enc.transform(label).toarray()
        return one_hot_label

    def ADMM(self, Z, x_aug, lam, iters):
        ZZ = Z.T.dot(Z)
        wk = ok = uk = np.zeros((Z.shape[1], x_aug.shape[1]))
        L1 = np.linalg.inv(ZZ+np.eye(Z.shape[1]))
        L2 = L1.dot(Z.T).dot(x_aug)

        for i in range(iters):
            tempc = ok-uk
            ck = L2+L1.dot(tempc)
            ok = self.shrinkage(ck+uk, lam)
            uk = uk+(ck-ok)
            wk = ok

        return wk.T

    def shrinkage(self, x, kappa):
        return (np.maximum(x-kappa, np.zeros((x-kappa).shape))-np.maximum(-x-kappa, np.zeros((x-kappa).shape)))

    def ridgeRegression(self, x, y):
        return (np.linalg.inv(x.T.dot(x)+self.C*np.eye(x.shape[1])).dot(x.T).dot(y))

    def sigmoid(self, x):
        return .5*(1+np.tanh(.5*x))

    def ReLU(self, x):
        x[x < 0] = 0
        return x

    def tanh(self, x):
        return np.tanh(x)

    def make_model(self, x_train, y_train):
        x_train, self.scaler = self.preprocess(x_train)
        y_train = self.oneHot(y_train)
        x_aug = np.hstack((x_train, 0.1*np.ones([x_train.shape[0], 1])))

        # feature nodes
        self.W_mapped = np.zeros((x_aug.shape[1], self.N2, self.N1))
        Z = np.zeros((x_aug.shape[0], self.N1*self.N2))
        for i in range(self.N1):
            we = 2*np.random.rand(x_aug.shape[1], self.N2)-1
            Zi = self.sigmoid(x_aug.dot(we))
            w = self.ADMM(Zi, x_aug, 1e-3, 100)
            self.W_mapped[:, :, i] = w
            Zi = self.sigmoid(x_aug.dot(w))
            Z[:, self.N2*i:self.N2*(i+1)] = Zi

        # enhanced nodes
        Z_aug = np.hstack((Z, 0.1*np.ones([Z.shape[0], 1])))
        self.W_enhanced = 2*np.random.rand(Z_aug.shape[1], self.N3)-1
        H = self.sigmoid(Z_aug.dot(self.W_enhanced))

        features = np.hstack((Z, H))
        # features = np.hstack((x_train, Z, H))
        self.beta = self.ridgeRegression(features, y_train)

    def forward(self, x_test):
        x_test = self.scaler.transform(x_test)
        x_aug = np.hstack((x_test, 0.1*np.ones([x_test.shape[0], 1])))

        # feature nodes
        Z = np.zeros((x_aug.shape[0], self.N1*self.N2))
        for i in range(self.N1):
            w = self.W_mapped[:, :, i]
            Zi = self.sigmoid(x_aug.dot(w))
            Z[:, self.N2*i:self.N2*(i+1)] = Zi

        # enhanced nodes
        Z_aug = np.hstack((Z, 0.1*np.ones([Z.shape[0], 1])))
        H = self.sigmoid(Z_aug.dot(self.W_enhanced))

        features = np.hstack((Z, H))
        # features = np.hstack((x_test, Z, H))
        y_value_predict = features.dot(self.beta)
        y_pre = np.zeros(y_value_predict.shape[0])
        for i in range(y_value_predict.shape[0]):
            y_pre[i] = np.argmax(y_value_predict[i, :])+1

        return y_pre

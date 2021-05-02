import argparse
import numpy as np
import sys
import warnings
from sklearn.metrics import classification_report, precision_score, f1_score
import time
from joblib import dump, load
from sklearn.base import BaseEstimator, ClassifierMixin
from decimal import *
import matplotlib.pyplot as plt
import csv
import os

rng = np.random.default_rng(seed=42)

getcontext().prec = 20

if not sys.warnoptions:
    warnings.simplefilter("ignore")

print('-' * 50)
print(
    "topics are as follows:\n[1] GENERAL\n[2] BUZZFEED\n[3] POLITIFACT\n[4] COVID\n[5] charliehebdo\n")
selection = input("What topic would you like to train?")

if selection == '1':
    TOPIC = 'GENERALTHREADS'
    SPLIT = 0.7
elif selection == '2':
    TOPIC = 'BUZZFEED'
    SPLIT = 0.8
elif selection == '3':
    TOPIC = 'POLITIFACT'
    SPLIT = 0.8
elif selection == '4':
    TOPIC = 'COVIDTHREADS'
    SPLIT = 0.7
elif selection == '5':
    TOPIC = 'charliehebdo'
    SPLIT = 0.7

print(TOPIC)

SEED = 42

d = 50

#

X = np.load('data/{}_results/X.npy'.format(TOPIC))

# get A
A = np.load('data/{}_results/A.npy'.format(TOPIC))

# get B
B = np.load('data/{}_results/B.npy'.format(TOPIC))

# get W
W = np.load('data/{}_results/W.npy'.format(TOPIC))
# Wtf = tf.Tensor(W, dtype=np.int32)

y = np.load('data/{}_results/y.npy'.format(TOPIC))

# get o

o = np.load('data/{}_results/o.npy'.format(TOPIC))
#c = np.load('data/{}_results/c.npy'.format(TOPIC))

# get e
e = np.load('data/{}_results/e.npy'.format(TOPIC))
y = y.flatten()
if TOPIC == 'COVIDTHREADS' or TOPIC == 'charliehebdo':
    X = X.T # for covid.
n, t = X.shape[0], X.shape[1]  # news, terms
m = A.shape[0]  # number of users
l = B.shape[0]  # number of publishers
r = int(X.shape[0] * SPLIT) #68 for covid

#A = rng.choice(a=[0, 1], size=(m, m), p=[.85, .15])

scoring = W*(y* -1)
print(X.shape)



#print(y.shape)
c = [(sum(row)+len(row))/(len(row)*2) for row in scoring]
SS = np.load('data/{}_results/SS.npy'.format(TOPIC))
# determine sizes of matricies

np.random.seed(SEED)

# rng.random.shuffle(X)
'''
A = rng.random(A)
B = rng.random(B)
W = rng.random(W)
y = rng.random(y)
o = rng.random(o)
e = rng.random(e)

'''

c = np.reshape(c, newshape=(m, 1))  # credibility score
e = np.reshape(e, newshape=(l, 1))
# c = np.load(r'C:\Users\minim\PycharmProjects\SeniorPrj\results\c.npy')
# print(y.shape, y)
# etf = tf.Tensor(e, dtype=np.int32)

# get yL
y = np.reshape(y, newshape=(n, 1))
o = np.reshape(o, newshape=(l, 1))

#np.random.shuffle(X) #   X n x t   bag-of-word
#np.random.shuffle(B.T) #   B l x n  publisher news interaction
#np.random.shuffle(W.T) #   W m x n   user-news interaction
#np.random.shuffle(y) #   y n x 1   news labels
# random shuffle of all of the data, seeded to the same value so any vector of vectors of size a, n is shuffled the same way



# f = int(r + (X.shape[0] * (1 - SPLIT)) / 2)
# f = int(r + ((X.shape[0] * (1-SPLIT))/2))
# labeled-unlabeled boundary #fix this to do random test, train split and then re format them to be labeled, 'unlabeled'
print(n, t)
# reshape o, e, c, etc.



# otf = tf.Tensor(o, dtype=np.int32)

# print(o.shape, o)

# ytf = tf.Tensor(y, dtype=np.int32)
Y = A.copy()

# print(y.shape, y)
# yL, yU = train_test_split(y, test_size=0.2, random_state=42)
yL = y[:r, :]  # labeled
yU = y[r:, :]  # unlabeled
# yT = y[f:, :] #testing set
# norm = np.linalg.norm(B)
B_bar = B / np.sum(B, axis=1).reshape(-1, 1)  # normalized B
# B_bartf = tf.Tensor(B_bar, dtype=np.int32)

# B_bar = B / norm  # normalized B
E = np.diag(e.reshape([-1]))
I = np.diag(np.ones([d]))
F = np.zeros([m + r, m + r])  # eq. 8
S = np.zeros([m + r, m + r])  # eq. 8
L = np.zeros([m + r, m + r])  # eq. 8
G = np.zeros([m + r, m + r])
L11 = L[0:m, 0:m]
L12 = L[0:m, m:]
L21 = L[m:, 0:m]
L22 = L[m:, m:]
U = np.random.uniform(0, 1, [m, d])  # user embedding

T = np.random.uniform(0, 1, [d, d])  # user-user correlation
D = np.random.uniform(0, 1, [n, d])  # news embedding
DL = D[:r, :]  # labeled
DU = D[r:, :]  # unlabeled
testDU = np.copy(DU)
# DF = D[f:, :]
p = np.random.uniform(0, 1, [d, 1])  # mapper of labeled news embedding
q = np.random.uniform(0, 1, [d, 1])  # mapper for publisher embedding

V = np.random.uniform(0, 1, [t, d])  # word embedding
Vc = np.random.uniform(0, 1, [t, d])



for k in range(d):
    V[:, k] /= np.linalg.norm(V[:, k])
    Vc[:, k] /= np.linalg.norm(Vc[:, k])




print("CHECKING SHAPES")
print('-'*50)
print()
print("X shape should be size n x t")
print("actual size: ", X.shape)

print("A shape should be size m x m")
print("actual size: ", A.shape)

print("B shape should be size l x n")
print("actual size: ", B.shape)

print("W shape should be size m x n")
print("actual size: ", W.shape)

print("Y shape should be size m x m")
print("actual size: ", Y.shape)

print("o shape should be size l x 1")
print("actual size: ", o.shape)

print("yl shape should be size r x 1")
print("actual size: ", yL.shape)

print('-'*50)





#(np.concatenate((X, SS.T)) - np.concatenate((D, Vc)).dot(V.T))



class TriFNClassify(BaseEstimator):

    def __init__(self, d=50, alpha=1e-5, beta=1e-4, gamma=10, lambda_=0.1, eta=1, epochs=200):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lmbda = lambda_
        self.eta = eta
        self.d = d
        self.epochs = epochs
        self.r = r
        self.D = D
        self.DL = DL
        self.DU = DU
        self.p = p
        self.pos_score = 0
        self.neg_score = 0
        self.V = V
        self.n = n
        self.SS = SS
        self.Vc = Vc
        self.minclip = 1e-20

    def __computeG(self):
        for i in range(m):
            for j in range(self.r):
                G[i, j] = (W[i, j] * ((c[j] * (1 - (1 + (yL[j])) / 2)) + (1 - c[j]) * ((1 + yL[j]) / 2)))

    def __compute_L(self):

        self.__computeG()
        # compute F
        for i in range(m + self.r):
            for j in range(m + self.r):
                if 1 <= i + 1 <= m and m + 1 <= j + 1 <= m + self.r:
                    F[i, j] = G[i, j - m]
                if m + 1 <= i + 1 <= m + self.r and 1 <= j + 1 <= m:
                    F[i, j] = G[i - m, j]

        # compute S
        for i in range(m + self.r):
            S[i, i] = np.sum(F[i])
        # print("Here is L: ", S-F)

        return S - F

    def __update_D(self):
        def pos(x):
            return (np.abs(x) + x) * 0.5
        def neg(x):
            return (np.abs(x) - x) * 0.5

        Adc1 = np.dot(B_bar.T, E.T)
        Bdc1 = np.dot(Adc1, E)
        Cdc1 = np.dot(Bdc1, o)
        D_caret_1 = np.dot(Cdc1, q.T)
        Adc2 = np.dot(self.DL, self.p)
        Bdc2 = neg(np.dot(Adc2, self.p.T))
        Cdc2 = pos(np.dot(yL, self.p.T))
        Ddc2 = neg(np.dot(L21, U))
        Edc2 = neg(np.dot(L22, self.DL))

        D_caret_2 = self.eta * Bdc2 + self.eta * Cdc2 + self.beta * Ddc2 + self.beta * Edc2

        D_caret = X.dot(self.V) + \
                  self.gamma * pos(B_bar.T.dot(E.T).dot(E).dot(o).dot(q.T)) + \
                  self.gamma * neg(B_bar.T.dot(E.T).dot(E).dot(B_bar).dot(self.D).dot(q).dot(q.T)) + \
                  np.pad(D_caret_2, ((D_caret_1.shape[0] - D_caret_2.shape[0], 0), (0, 0)))

        D_tilde_1 = self.beta * pos(L21.dot(U)) + self.beta * pos(L22.dot(self.DL)) + self.eta * pos(
            self.DL.dot(self.p).dot(self.p.T)) + self.eta * neg(
            yL.dot(self.p.T))
        D_tilde = self.D.dot(self.V.T.dot(self.V)) + \
                  self.lmbda * self.D + \
                  self.gamma * pos(B.T.dot(E.T).dot(E).dot(B_bar).dot(self.D).dot(q).dot(q.T)) + \
                  self.gamma * neg(B_bar.T.dot(E.T).dot(E).dot(o).dot(q.T)) + \
                  np.pad(D_tilde_1, ((D.shape[0] - D_tilde_1.shape[0], 0), (0, 0))
                         )
        divres = np.sqrt((D_caret / D_tilde))
        return self.D * np.sqrt(divres)

    def __update_U(self):

        def pos(x):
            return (np.abs(x) + x) * 0.5

        def neg(x):
            return (np.abs(x) - x) * 0.5

        ya = Y * A
        Auc = np.dot((ya), U)
        Buc = np.dot(Auc, T.T)
        Cuc = np.dot((ya).T, U)
        Duc = np.dot(Cuc, T)
        Euc = neg(np.dot(L11, U))
        Fuc = neg(L12.dot(self.DL))
        U_caret = self.alpha * Buc + self.alpha * Duc + self.beta * Euc + self.beta * Fuc

        Aut = np.dot(U, T)
        But = np.dot(Aut, U.T)
        Cut = np.dot(self.alpha * (Y * But), U)
        Dut = np.dot(Cut, T.T)
        Eut = np.dot(self.alpha * ((Y * But).T), U)
        Fut = np.dot(Eut, T)

        U_tilde = Dut + Fut + self.lmbda * U + self.beta * pos(L11.dot(U)) + self.beta * pos(L12.dot(self.DL))
        divres = np.divide(U_caret, U_tilde, out=np.zeros_like(U_caret), where=U_tilde != 0)
        return U * np.sqrt((divres))

    def __update_V(self):

        XD = np.dot(X.T, self.D)
        SVc = np.dot(self.SS, self.Vc)
        DtD = np.dot(self.D.T, self.D)
        VctVc = np.dot(self.Vc.T, self.Vc)
        '''
        for k in range(self.d):
            num0 = DtD[k, k] * self.V[:, k] + VctVc[k, k] * self.V[:, k]
            num1 = XD[:, k] + SVc[:, k]
            num2 = np.dot(self.V, DtD[:, k]) + np.dot(self.V, VctVc[:, k])*self.lmbda
            self.V[:, k] = num0 + num1 - num2
            self.V[:, k] = np.maximum(self.V[:, k], self.minclip)  # project > 0

            self.V[:, k] /= np.linalg.norm(self.V[:, k]) + self.minclip  # normalize
        '''
        for k in range(self.d):
            num0 = DtD[k, k] * self.V[:, k] + VctVc[k, k] * self.V[:, k]
            num1 = XD[:, k] + SVc[:, k]
            num2 = np.dot(self.V, DtD[:, k]) + np.dot(self.V, VctVc[:, k])* self.lmbda
            self.V[:, k] = num0 + num1 - num2
            self.V[:, k] = np.maximum(self.V[:, k], self.minclip)  # project > 0
        #self.V *= np.sqrt(temp)
            self.V[:, k] /= np.linalg.norm(self.V[:, k]) + self.minclip  # normalize
        # Update Vc
        VtV = self.V.T.dot(self.V)
        SV = np.dot(self.SS, self.V)
        for k in range(self.d):
            self.Vc[:,k] = self.Vc[:,k] + SV[:,k] - np.dot(self.Vc, VtV[:,k])
            self.Vc[:, k] = np.maximum(self.Vc[:, k], self.minclip)
        
        return self.V, self.Vc

    def __update_T(self):
        T1 = self.alpha * U.T.dot(Y * A).dot(U)
        T2 = (self.alpha * U.T.dot(Y * (U.dot(T).dot(U.T))).dot(U) + self.lmbda * T)
        divres = np.divide(T1, T2)

        return T * np.sqrt(divres)

    def __update_p(self):
        return np.linalg.inv(self.eta * self.DL.T.dot(self.DL) + self.lmbda * I).dot(self.eta * self.DL.T).dot(yL)

    def __update_q(self):
        return np.linalg.inv(self.gamma * self.D.T.dot(B_bar.T).dot(E).dot(B_bar).dot(self.D) + self.lmbda * I).dot( \
            self.gamma * self.D.T).dot(B_bar.T).dot(E).dot(o)

    def __compute_yU(self):
        return np.sign(self.DU.dot(self.p))

    def fit(self):
        print("fitting...")
        print('computing L ...')

        L[:, :] = self.__compute_L()
        print('L done ...')
        print('-' * 50)
        print('updating parameters ...')
        objectives = []
        accs = []
        mindiff = 1e20
        minval = 1e20
        minvalEpoch = 0
        mindiffEpoch = 0
        maxMacro = 0
        maxWeighted = 0
        precisionEpoch = 0
        macroEpoch = 0
        f1Epoch = 0
        maxf1 = 0
        np.save('results/{}_DU.npy'.format(TOPIC), testDU)
        np.save('results/{}_yU.npy'.format(TOPIC), yU)
        w = open('results/{}_objectives_precision_{}.csv'.format(TOPIC, self.d), 'w', newline='')
        writer = csv.writer(w, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for _ in range(self.epochs):

            self.D[:, :] = self.__update_D()
            U[:, :] = self.__update_U()
            self.V[:, :], self.Vc[:, :] = self.__update_V()
            T[:, :] = self.__update_T()
            self.p[:, :] = self.__update_p()
            q[:, :] = self.__update_q()
            H = np.concatenate((U, self.DL), axis=0)
            objective_value = np.linalg.norm((np.concatenate((X, self.SS.T)) - np.concatenate((self.D, self.Vc)).dot(self.V.T)), ord='fro') ** 2 + \
                              self.alpha * (np.linalg.norm(np.multiply(Y, (A - U.dot(T.dot(U.T)))), ord='fro') ** 2) + \
                              self.beta * np.trace(H.T.dot(L).dot(H)) + \
                              self.gamma * (np.linalg.norm(np.multiply(e, (B_bar.dot(self.D).dot(q) - o)), ord=2) ** 2) \
                              + self.eta * (np.linalg.norm(DL.dot(self.p) - yL, ord=2) ** 2) + self.lmbda * (
                                      np.linalg.norm(self.D, ord='fro') ** 2 + np.linalg.norm(self.V, ord='fro') ** 2 \
                                      + np.linalg.norm(U, ord='fro') ** 2 + np.linalg.norm(T, ord='fro') ** 2 \
                                      + np.linalg.norm(self.p, ord=2) ** 2 + np.linalg.norm(q, ord=2) ** 2 +\
                                      np.linalg.norm(self.Vc, ord='fro') ** 2)

            objectives.append(objective_value)
            trueLabels = y[r:, :]

            results = testDU.dot(self.p)
            for item in range(len(results)):
                results[item] = np.sign(results[item][0])

            report = classification_report(
                y_true=trueLabels,
                y_pred=results, output_dict=True)

            #rep = classification_report(trueLabels, results, zero_division=0, output_dict=True)
            f1 = report['weighted avg']['f1-score']
            precision = float(report['weighted avg']['precision'])
            accs.append(precision)
            writer.writerow((objective_value, f1 ))
            if _ > 1:
                print('-' * 50)
                print("epoch ", _, " done")
                oldval = objectives[-2]
                obval = objective_value
                diff = abs(oldval - obval)
                olddiff = abs(objectives[-3] - objectives[-2])
                if int(oldval) - int(obval) < 0:
                    obarrow = u"     ▲"
                elif int(oldval) - int(obval) == 0:
                    obarrow = u"     —"
                else:
                    obarrow = u" ▼"
                if diff - olddiff < 0:
                    arrow = u" ▲"
                else:
                    arrow = u" ▼"
                print("objective value: ", objective_value, obarrow)
                print("k -> k+1 difference: ", diff, arrow)
                print("precision: ", precision)

                if diff < mindiff:
                    mindiff = diff
                    mindiffEpoch = _
                    np.save('results/{}_diff_p.npy'.format(TOPIC), self.p)
                    np.save('results/{}_diff_V.npy'.format(TOPIC), self.V)
                    print(u'♤' * 55)
                if objective_value < minval:
                    minval = objective_value
                    minvalEpoch = _
                    np.save('results/{}_p.npy'.format(TOPIC), self.p)
                    np.save('results/{}_V.npy'.format(TOPIC), self.V)
                    print(u'✪' * 50)
                if f1 > maxf1:
                    maxf1 = f1
                    f1Epoch = _
                    np.save('results/{}_F1_p.npy'.format(TOPIC), self.p)
                    np.save('results/{}_F1_V.npy'.format(TOPIC), self.V)
                    print(u'&' * 50)

                if precision >= maxWeighted:
                    maxWeighted = precision
                    precisionEpoch = _
                    np.save('results/{}_prec_p.npy'.format(TOPIC), self.p)
                    np.save('results/{}_prec_V.npy'.format(TOPIC), self.V)
                    print('%' * 50)
                if report['macro avg']['f1-score'] >= maxMacro:
                    maxMacro = report['macro avg']['f1-score']
                    macroEpoch = _
                    np.save('results/{}_macro_p.npy'.format(TOPIC), self.p)
                    np.save('results/{}_macro_V.npy'.format(TOPIC), self.V)
                    print('*' * 50)




        fig = plt.figure()

        ax1 = fig.add_subplot(211)
        ax1.set_title("accuracy vs epoch number")
        ax1.plot(accs)
        ax2 = plt.subplot(212)
        ax2.set_title("objective value vs epoch number")
        ax2.plot(objectives)
        fig.tight_layout()
        plt.savefig('results/{}_accuracyandobjectivevals.png'.format(TOPIC))
        np.save('results/{}_DU.npy'.format(TOPIC), testDU)
        np.save('results/{}_yU.npy'.format(TOPIC), yU)
        np.save('results/{}_p.npy'.format(TOPIC), self.p)
        print("TriFN Model finished training...")
        print("LATENT SPACE DIMENSION: ", self.d)
        print("min val: ", minval, " - found at epoch: ", minvalEpoch, " with precision ", accs[minvalEpoch])
        print("min objective difference: ", mindiff, " - found at epoch : ", mindiffEpoch, " with precision ",
              accs[mindiffEpoch])
        print("maximum macro f1-score: ", maxMacro, " - found at epoch : ", macroEpoch, " with precision ",
              accs[macroEpoch])
        print("maximum weighted precision: ", maxWeighted, " - found at epoch : ", precisionEpoch)

    def predict(self, X_test):
        self.DU = X_test
        return self.__compute_yU()

    def getPosScore(self):
        return self.pos_score

    def getNegScore(self):
        return self.neg_score

    def getV(self):
        return self.V

    def getp(self):
        return self.p

    def setp(self, newp):
        self.p = newp

    def setV(self, newV):
        self.V = newV


def trainModel():
    # TOPIC = topicname
    test = TriFNClassify()
    test.fit()

    savedp = np.load('results/{}_p.npy'.format(TOPIC))
    savedV = np.load('results/{}_V.npy'.format(TOPIC))
    savedDU = np.load('results/{}_DU.npy'.format(TOPIC))

    results = savedDU.dot(savedp)
    for item in range(len(results)):
        results[item] = np.sign(results[item][0])

    print(results)
    report = classification_report(
        y_true=yU,
        y_pred=results, zero_division=0)

    print(report)
    dump(test, 'MODELS/{}_epochs_TriFN.joblib'.format(TOPIC))
    test.setp(savedp)
    test.setV(savedV)

    results = test.predict(DU)
    print(results)
    dump(test, 'MODELS/{}_TriFN.joblib'.format(TOPIC))
    # savedV = test.getV()
    # np.save('results/{}_V.npy'.format(TOPIC), savedV)
    report = classification_report(
        y_true=yU,
        y_pred=results)

    print(report)


'''
Given the ability to handle missing entries in NMF described in the above section and the powerful missing value 
imputation of NMF demonstrated in “Missing value imputation” section, we come up with a novel approach, akin to the 
well-known the training-validation split approach in supervised learning.

1.
Some portion (e.g., 30%) of entries are randomly deleted (selected to be missing) from A.

2.
The deleted entries are imputed by NMF with a set of different k’s.

3.
The imputed entries are compared to their observed values, and the k that gives the smallest error is selected.

The above approach can be argued by the assumption that only the correct k, if exists, has the right decomposition 
that can recover the missing entries. In contrast to the training-validation split in supervised learning, 
due to the typically big number of entries in A, we generally have a very large ‘sample size’. 
One can also easily adapt the idea of cross-validation to this approach. This idea should apply to any unsupervised 
learning method that handles missing values. 

'''

if __name__ == '__main__':
    trainModel()

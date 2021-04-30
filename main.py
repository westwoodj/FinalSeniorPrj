import argparse
import numpy as np
import sys
import warnings
from sklearn.metrics import classification_report, f1_score
import time
from joblib import dump, load
from sklearn.base import BaseEstimator, ClassifierMixin
from decimal import *



rng = np.random.default_rng(seed=42)


getcontext().prec=20


if not sys.warnoptions:
    warnings.simplefilter("ignore")

print('-'*50)
print("topics are as follows:\n[1] charliehebdo\n[2] ebola-essien\n[3] ferguson\n[4] germanwings-crash\n[5] ottawashooting\n"
      "[6] prince-toronto\n[7] putinmissing\n[8] sydneysiege\n[9] Covid\n[10] General (FNN)\n")
selection = input("What topic would you like to train?")

if selection == '1':
    TOPIC = 'charliehebdo'
    SPLIT = 0.4
elif selection == '2':
    TOPIC = 'ebola-essien'
elif selection == '3':
    TOPIC = 'ferguson'
elif selection == '4':
    TOPIC = 'germanwings-crash'
    SPLIT = 0.5
elif selection == '5':
    TOPIC = 'ottawashooting'
    SPLIT = 0.5
elif selection == '6':
    TOPIC = 'prince-toronto'
elif selection == '7':
    TOPIC = 'putinmissing'
elif selection == '8':
    TOPIC = 'sydneysiege'
    SPLIT = 0.3
elif selection == '9':
    TOPIC = 'COVID'
elif selection == '10':
    TOPIC = 'GENERAL'

print(TOPIC)

SEED = 42


d = 10  # num features

#
alpha, beta, gamma, lmbda, eta = -5, 1e-4, 10, 0.1, 1 #lmbda because lambda functions in Python




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

# get e
e = np.load('data/{}_results/e.npy'.format(TOPIC))
# determine sizes of matricies
X = X.T
np.random.seed(SEED)

#rng.random.shuffle(X)
'''
A = rng.random(A)
B = rng.random(B)
W = rng.random(W)
y = rng.random(y)
o = rng.random(o)
e = rng.random(e)

'''

np.random.shuffle(X)
np.random.shuffle(A)
np.random.shuffle(B)
np.random.shuffle(W)
np.random.shuffle(y)
np.random.shuffle(o)
np.random.shuffle(e)





#random shuffle of all of the data, seeded to the same value so any vector of vectors of size a, n is shuffled the same way


n, t = X.shape[0], X.shape[1]  # news, terms
m = A.shape[0]  # number of users
l = B.shape[0]  # number of publishers
r = int(X.shape[0] * SPLIT)
f = int(r + (X.shape[0] * (1-SPLIT))/2)
#f = int(r + ((X.shape[0] * (1-SPLIT))/2))
# labeled-unlabeled boundary #fix this to do random test, train split and then re format them to be labeled, 'unlabeled'

# reshape o, e, c, etc.
o = o - o.mean()
o = o / 35
o = np.reshape(o, newshape=(l, 1))
d = 10
# otf = tf.Tensor(o, dtype=np.int32)

# print(o.shape, o)
c = np.random.uniform(0, 1, [m, 1])  # credibility score
e = np.reshape(e, newshape=(l, 1))
# c = np.load(r'C:\Users\minim\PycharmProjects\SeniorPrj\results\c.npy')
# print(y.shape, y)
# etf = tf.Tensor(e, dtype=np.int32)

# get yL
y = np.reshape(y, newshape=(n, 1))
# ytf = tf.Tensor(y, dtype=np.int32)
Y = A.copy()

# print(y.shape, y)
# yL, yU = train_test_split(y, test_size=0.2, random_state=42)
yL = y[:r, :]  # labeled
yU = y[r:, :]  # unlabeled
#yT = y[f:, :] #testing set
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
V = np.random.uniform(0, 1, [t, d])  # word embedding
T = np.random.uniform(0, 1, [d, d])  # user-user correlation
D = np.random.uniform(0, 1, [n, d])  # news embedding
DL = D[:r, :]  # labeled
DU = D[r:, :]  # unlabeled
#DF = D[f:, :]
p = np.random.uniform(0, 1, [d, 1])  # mapper of labeled news embedding
q = np.random.uniform(0, 1, [d, 1])  # mapper for publisher embedding
'''
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

'''



class TriFNClassify(BaseEstimator):

    def __init__(self, d = 10, alpha = -5, beta = 1e-4, gamma = 10,lambda_ = 0.1, eta = 1, epochs = 100):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lmbda = lambda_
        self.eta = eta
        self.epochs = epochs
        self.r = r
        self.D = D
        self.DL = DL
        self.DU = DU
        self.p = p
        self.pos_score = 0
        self.neg_score = 0
        self.V = V

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
                    F[i, j] = G[i, j-m]
                if m + 1 <= i + 1 <= m + self.r and 1 <= j + 1 <= m:
                    F[i, j] = G[i-m, j]

        # compute S
        for i in range(m + self.r):
            S[i, i] = np.sum(F[i])
        # print("Here is L: ", S-F)

        return S - F

    def __update_D(self):
        # print("updating D")
        def pos(x):
            # print("pos: \n", x)
            return (np.abs(x) + x) * 0.5

        def neg(x):
            # print("neg: \n", x)
            return (np.abs(x) - x) * 0.5

        # print(B_bar.shape)
        # print(T.shape)
        # print(E.shape)
        # print(o.shape)
        # print(q.shape)
        # D_car_1 = B_bar.T.dot(E.T)
        Adc1 = np.dot(B_bar.T, E.T)
        Bdc1 = np.dot(Adc1, E)
        Cdc1 = np.dot(Bdc1, o)
        D_caret_1 = np.dot(Cdc1, q.T)
        # D_caret_1 = B_bar.T.dot(E.T).dot(E).dot(o).dot(q.T)
        Adc2 = np.dot(self.DL, self.p)
        Bdc2 = neg(np.dot(Adc2, self.p.T))
        Cdc2 = pos(np.dot(yL, self.p.T))
        Ddc2 = neg(np.dot(L21, U))
        Edc2 = neg(np.dot(L22, self.DL))

        D_caret_2 = self.eta * Bdc2 + self.eta * Cdc2 + self.beta * Ddc2 + self.beta * Edc2

        D_caret = X.dot(V) + \
                  self.gamma * pos(B_bar.T.dot(E.T).dot(E).dot(o).dot(q.T)) + \
                  self.gamma * neg(B_bar.T.dot(E.T).dot(E).dot(B_bar).dot(self.D).dot(q).dot(q.T)) + \
                  np.pad(D_caret_2, ((D_caret_1.shape[0] - D_caret_2.shape[0], 0), (0, 0)))

        D_tilde_1 = self.beta * pos(L21.dot(U)) + self.beta * pos(L22.dot(self.DL)) + self.eta * pos(self.DL.dot(self.p).dot(self.p.T)) + self.eta * neg(
            yL.dot(self.p.T))
        D_tilde = self.D.dot(V.T.dot(V)) + \
                  self.lmbda * self.D + \
                  self.gamma * pos(B.T.dot(E.T).dot(E).dot(B_bar).dot(self.D).dot(q).dot(q.T)) + \
                  self.gamma * neg(B_bar.T.dot(E.T).dot(E).dot(o).dot(q.T)) + \
                  np.pad(D_tilde_1, ((D.shape[0] - D_tilde_1.shape[0], 0), (0, 0))
                         )
        divres = np.sqrt((D_caret / D_tilde))
        # print(divres, sum(divres))
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
        # print(np.isfinite(U_caret).all(), np.isfinite(U_tilde).all())
        # print(np.isfinite(U).all())
        # print((U_caret >= 0).all(), (U_tilde >= 0).all())
        '''step = np.sqrt(U_caret/U_tilde)
        if np.isnan(step).any():
            print("nan error!")
            badItems = np.where(np.isnan(step))
            print("bad inputs at: "+str(badItems))
            print("bad  input values: (caret, tilde) ", str(U_caret[np.isnan(step)]), str(U_tilde[np.isnan(step)]))
            raise Exception("unexpected nan value in sqrt step!!")'''
        divres = np.divide(U_caret, U_tilde, out=np.zeros_like(U_caret), where=U_tilde != 0)
        return U * np.sqrt((divres))

    def __update_V(self):
        v1 = X.T.dot(self.D)
        v2 = (self.V.dot(self.D.T).dot(self.D) + self.lmbda * self.V)
        divres = np.divide(v1, v2)
        return self.V * np.sqrt(divres)

    def __update_T(self):
        T1 = self.alpha * U.T.dot(Y * A).dot(U)
        T2 = (self.alpha * U.T.dot(Y * (U.dot(T).dot(U.T))).dot(U) + self.lmbda * T)
        divres = np.divide(T1, T2)

        return T * np.sqrt(divres)

    def __update_p(self):
        return (self.eta / (self.eta * self.DL.T.dot(self.DL) + self.lmbda * I)).dot(self.DL.T).dot(yL)

    def __update_q(self):
        return (self.eta / (self.eta * self.D.T.dot(B_bar.T).dot(E).dot(B_bar).dot(self.D) + self.lmbda * I)).dot(self.D.T).dot(B_bar.T).dot(E).dot(o)

    def __compute_yU(self):
        # print(DU.dot(p))
        # print(sum(DU.dot(p)))
        return np.sign(self.DU.dot(self.p))  # , out=np.zeros_like(DU.dot(p)), where=abs(DU.dot(p))<1)

    def fit(self):
        #self.yU = y
        print("fitting...")
        print('computing L ...')

        L[:, :] = self.__compute_L()
        print('L done ...')
        print('-'*50)
        #print("[", end='')
        print('updating parameters ...')
        for _ in range(self.epochs):

            self.D[:, :] = self.__update_D()
            U[:, :] = self.__update_U()
            self.V[:, :] = self.__update_V()
            T[:, :] = self.__update_T()
            self.p[:, :] = self.__update_p()
            q[:, :] = self.__update_q()

            #np.linalg.norm(X - D.dot(V.T)) + (alpha * np.linalg.norm(np.multiply(Y, (A-U.dot(T.dot(U.T))))) ) + beta* np.trace()

            if _ == 0:
                results = self.__compute_yU()

                trueLabels = y[r:, :]
                # print("real labels:     ", trueLabels.T)
                # print("predicted labels:", results.T)
                maxMacro = f1_score(y_true=trueLabels, y_pred=results, average='macro', zero_division=0)


            else:
                '''
                #print("=", end='')
                print("epoch ", _, " done")
                print("-------------- SUMS --------------")
                print("D: ", sum(self.D))
                print("U: ", sum(self.U))
                print("V: ", sum(self.U))
                print("T: ", sum(self.U))
                print("p: ", sum(self.U))
                print("q: ", sum(self.U))
                print("DU: ", sum(self.DU))
                '''
                print('-'*50)
                print("epoch ", _, " done")

                results = self.__compute_yU()
                #print(sum(self.D), sum(U), sum(self.V), sum(T), sum(self.p), sum(q))
                print(sum(sum(self.D) + sum(U) + sum(self.V) + sum(T)+ sum(self.p)+ sum(q)))
                print(self.p)
                trueLabels = y[r:, :]
                #print("real labels:     ", trueLabels.T)
                #print("predicted labels:", results.T)
                report = classification_report(
                    y_true=trueLabels,
                    y_pred=results, output_dict=True, zero_division=0)
                #print("weighted avg: ", report['weighted avg']['precision']*100, " fake news avg: ", report['1.0']['precision']*100, " true news avg: ", report['-1.0']['precision']*100)
                #print("fake news sensitivity: ", report['1.0']['recall'], " F1 score: ", report['weighted avg']['f1-score'])
                printable_report = classification_report(
                    y_true=trueLabels,
                    y_pred=results, zero_division=0)
                print(printable_report)
                micro = f1_score(y_true=trueLabels, y_pred = results, average='micro', zero_division=0)
                macro = f1_score(y_true=trueLabels, y_pred = results, average='macro', zero_division=0)
                print('-'*50)
                print('micro: \n', micro)
                print('macro: \n', macro)
                # if the macro average f1 is >= 70% and the micro average is less than 1.3X the macro - the model is optimized
                if macro >= .50 and micro < 1.4 * macro:
                    print("still better!")
                    self.pos_score = report['1.0']['precision']
                    self.neg_score = report['-1.0']['precision']
                    self.bestV = self.V
                    self.bestp = self.p
                    self.bestD = self.D
                    self.bestDU = self.DU

                    maxMacro = macro
                    np.save("data/ottawashooting_results/{}}_V.npy", self.V)
                    break


                    #
                '''
                elif macro < maxMacro:
                    print("oops, I am less than last generation")
                    self.p = self.bestp
                    self.V = self.bestV
                    self.DU = self.bestDU

                    self.D = self.D

                    np.save('results/{}_V.npy'.format(TOPIC), self.V)

                    break
                '''
                #print(report.keys())
                #print("predicted news labels:\n", np.reshape(results, newshape=(len(yU),)))
                #print(report)

        #print("]")

        print("TriFN Model finished training...")

    '''
    
    https://stats.stackexchange.com/questions/156923/should-i-make-decisions-based-on-micro-averaged-or-macro-averaged-evaluation-mea
    
    https://www.sciencedirect.com/science/article/pii/S0306457309000259
    
    
    If you think there are labels with more instances than others and if you want to bias your metric toward the least 
    populated ones (or at least you don't want to bias toward the most populated ones), use macromedia.
    
    If the micromedia result is significantly lower than the macromedia one, it means that you have some gross 
    misclassification in the most populated labels, whereas your smaller labels are probably correctly classified. 
    
    If the macromedia result is significantly lower than the micromedia one, it means your smaller labels are poorly classified,
    whereas your larger ones are probably correctly classified.
    
    
    i.e. we want a high macro average because more news is real than fake, ... but if:
    
        micro << macro, true news is grossly misclassified
        
        micro >> macro fake news is misclassified but real news is correctly classified
    
    '''



    def predict(self, X_test):
        self.DU = X_test
        return self.__compute_yU()

    def getPosScore(self):
        return self.pos_score
    def getNegScore(self):
        return self.neg_score
    def getV(self):
        return self.V

def trainModel():
    #TOPIC = topicname
    test = TriFNClassify()
    test.fit()
    results = test.predict(DU)
    print(results)
    dump(test, 'MODELS/{}_TriFN.joblib'.format(TOPIC))


    report = classification_report(
                        y_true=y[r:, :],
                        y_pred=results)

    print(report)


if __name__ == '__main__':
    trainModel()



import numpy as np
import csv
from sklearn.model_selection import train_test_split
#import tensorflow_io as tfio
d = 10  # num features

#
alpha, beta, gamma, lmbda, eta = -5, 1e-4, 10, 0.1, 1 #lmbda because lambda functions in Python

'''
alpha and beta control social relationship and user-news engagements

gamma controls publisher-partisian contribution

eta controls the input of the linear classifier

'''


#get X







#yLtf = tf.Tensor(yL, dtype=np.int32)
#yUtf = tf.Tensor(yU, dtype=np.int32)


#print(yL.shape, yL)
#print(yU.shape, yU)

#get Y
#Ytf = tf.Tensor(Y, dtype=np.int32)




#Ftf = tf.Tensor(F, dtype=np.int32)
#Stf = tf.Tensor(S, dtype=np.int32)
#Ltf = tf.Tensor(L, dtype=np.int32)




#Gtf = tf.Tensor(G, dtype=np.int32)





#L11tf = tf.Tensor(L11, dtype=np.int32)
#L12tf = tf.Tensor(L12, dtype=np.int32)
#L21tf = tf.Tensor(L21, dtype=np.int32)
#L22tf = tf.Tensor(L22, dtype=np.int32)

#Etf = tf.Tensor(E, dtype=np.int32)
#Itf = tf.Tensor(I, dtype=np.int32)

#randomly initialize U, V, T, D, p, q

X = np.zeros((10, 10))
A = np.zeros((10, 10))
B = np.zeros((10, 10))
B.fill(1)
W = np.zeros((10, 10))
y = np.zeros((10, 1))
o = np.zeros((10, 1))
e = np.zeros((10, 1))




n, t = X.shape[0], X.shape[1]  # news, terms
m = A.shape[0]  # numer of users
l = B.shape[0]  # number of publishers
print(B.shape[0])
r = int(X.shape[0])
# labeled-unlabeled boundary #fix this to do random test, train split and then re format them to be labeled, 'unlabeled'

# reshape o, e, c, etc.


# otf = tf.Tensor(o, dtype=np.int32)

# print(o.shape, o)
c = np.random.uniform(0, 1, [m, 1])  # credibility score
# c = np.load(r'C:\Users\minim\PycharmProjects\SeniorPrj\results\c.npy')
# print(y.shape, y)
# etf = tf.Tensor(e, dtype=np.int32)

# get yL
# ytf = tf.Tensor(y, dtype=np.int32)
Y = A.copy()

# print(y.shape, y)
# yL, yU = train_test_split(y, test_size=0.2, random_state=42)
yL = y[:r, :]  # labeled
yU = y[r:, :]  # unlabeled
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
p = np.random.uniform(0, 1, [d, 1])  # mapper of labeled news embedding
q = np.random.uniform(0, 1, [d, 1])  # mapper for publisher embedding
#print("q: ", q.shape, q)

#Utf = tf.Tensor(U, dtype=np.int32)
#Vtf = tf.Tensor(V, dtype=np.int32)
#Ttf = tf.Tensor(T, dtype=np.int32)
#Dtf = tf.Tensor(D, dtype=np.int32)
#DLtf = tf.Tensor(DL, dtype=np.int32)
#DUtf = tf.Tensor(DU, dtype=np.int32)
#ptf = tf.Tensor(p, dtype=np.int32)
#qtf = tf.Tensor(q, dtype=np.int32)

'''

'''


pass
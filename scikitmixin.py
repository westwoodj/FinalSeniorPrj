from params import *
import sys
import warnings
from sklearn.metrics import classification_report
import time
from joblib import dump, load
from sklearn.base import BaseEstimator, ClassifierMixin
from decimal import *

getcontext().prec=20

if not sys.warnoptions:
    warnings.simplefilter("ignore")




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

            if _ % 2 == 0:
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
                print("epoch ", _, " done")

                results = self.__compute_yU()

                trueLabels = y[r:, :]
                print(trueLabels)
                report = classification_report(
                    y_true= trueLabels,
                    y_pred=results, output_dict=True)
                print(report)
                if report['weighted avg']['precision'] > 0.70:
                    self.pos_score = report['1']['precision']
                    self.neg_score = report['-1']['precision']
                    np.save('C:\\Users\\minim\\PycharmProjects\\SeniorPrj\\results\\ottawashooting_V.npy', self.V)

                    break
                #print(report.keys())
                #print("predicted news labels:\n", np.reshape(results, newshape=(len(yU),)))
                #print(report)

        #print("]")

        print("TriFN Model finished training...")


    def predict(self, X_test):
        self.DU = X_test
        return self.__compute_yU()

    def getPosScore(self):
        return self.pos_score
    def getNegScore(self):
        return self.neg_score
    def getV(self):
        return self.V



def trainModel(topic):

    test = TriFNClassify(topic=topic)
    test.fit()
    results = test.predict(DU)
    print(results)
    dump(test, 'MODELS/{}_TriFN.joblib'.format(topic))


    report = classification_report(
                        y_true=y[r:, :],
                        y_pred=results)

    print(report)


if __name__ == '__main__':
    trainModel("ottawashooting")
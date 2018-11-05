import pandas as pd
import numpy as np
import scipy.special as dg
import math
e0 = f0 = 1
a0 = b0 = 1e-16
import matplotlib.pyplot as plt
class VI:
    def __init__(self, ds, y, z):
        [X, Y, Z] = self.load_data(ds, y, z)
        self.ds = X
        self.y = Y
        self.z = Z
        self.N = self.ds.shape[0]
        self.d = self.ds.shape[1]
        self.te = np.dot(self.ds.T, self.y).reshape(self.d, 1)
        self.tp = np.zeros((self.d, self.d))
        for i in range(self.N):
            self.tp += np.dot(self.ds[i].reshape(self.d, 1), self.ds[i].reshape(1, self.d))
        self._lambda_a = self.N / 2 + e0
        self._lambda_b = f0
        self._alpha_a = np.ones((self.d, 1)) * a0 + 1/2
        self._alpha_b = np.ones((self.d, 1)) * b0
        self._w_mu = np.zeros((self.d, 1))
        self._w_sigma = np.diagflat(self._alpha_a / self._alpha_b)

    def load_data(self, ds, y, z):
        self.ds = pd.read_csv(ds).values
        self.y = pd.read_csv(y).values
        self.z = pd.read_csv(z).values
        return self.ds, self.y, self.z

    def _likelihood(self):
        sign, det = np.linalg.slogdet(self._w_sigma)
        return - self._lambda_a * math.log(self._lambda_b) - np.sum(np.dot(self._alpha_a.T, np.log(self._alpha_b))) + sign * det / 2

    def _update_alpha(self):
        self._alpha_b = np.diag(np.outer(self._w_mu, self._w_mu) + self._w_sigma).reshape(self.d, 1) / 2 + b0

    def _update_lambda(self):
        th = np.sum(np.diag(np.dot(np.dot(self.ds, self._w_sigma), self.ds.T)))
        tp = np.sum((self.y - np.dot(self.ds, self._w_mu).reshape(self.N, 1)) ** 2) + th
        self._lambda_b = f0 + tp / 2

    def _update_w(self):
        self._w_sigma = np.linalg.inv(np.diagflat(self._alpha_a / self._alpha_b) + self.tp * self._lambda_a / self._lambda_b)
        self._w_mu = np.dot(self._w_sigma, self._lambda_a / self._lambda_b * self.te).reshape(self.d, 1)


if __name__ == '__main__':
    for i in range(3):
        iterator = VI('./data_csv/X_set' + str(i+1) + '.csv', './data_csv/y_set' + str(i+1) + '.csv', './data_csv/z_set' + str(i+1) + '.csv')
        objective = []
        epoch = []
        for j in range(500):
            iterator._update_lambda()
            iterator._update_alpha()
            iterator._update_w()
            objective.append(iterator._likelihood())
            epoch.append(j)
            if j == 499:
                plt.plot(np.arange(iterator.d), iterator._alpha_b / iterator._alpha_a)
                plt.xlabel('k')
                plt.ylabel('1/E_q[alpha]')
                plt.title('Dataset' + str(i+1) + ': 1/Eq[alpha]')
                plt.show()
                print("1/Eq[lambda]: dataset"+ str(i+1)+"\n")
                print(iterator._lambda_b / iterator._lambda_a)
        plt.plot(epoch, objective)
        plt.xlabel('epoch')
        plt.ylabel('Objective function')
        plt.title('Dataset' + str(i+1) + ': Variational Objective Function')
        plt.show()
        y = np.dot(iterator.ds, iterator._w_mu)
        plt.plot(iterator.z, 10 * np.sinc(iterator.z), '#ebb329', linewidth=3.0)
        plt.plot(iterator.z, y, '#c76e5d', linewidth=3.0)
        plt.scatter(iterator.z, iterator.y)
        plt.xlabel('z')
        plt.ylabel('y')
        plt.legend(['ground truth', 'predicted', 'data'])
        plt.title('Dataset' + str(i+1) + ': Regression Results')
        plt.show()

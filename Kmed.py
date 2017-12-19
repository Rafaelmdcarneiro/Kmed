from copy import deepcopy
import numpy as npy
import pandas as pd
from matplotlib import pyplot as plt


class Kmed:

    def __init__(self, k, file_name):
        self.k = k
        self.file_name = file_name
        self.dataset = None

    def load_dataset(self):
        self.dataset = pd.read_csv(self.file_name)

    @staticmethod
    def distance(a, b, ax=1):
        return npy.linalg.norm(a - b, axis=ax)

    def run(self):

        f1 = self.dataset['V1'].values
        f2 = self.dataset['V2'].values
        X = npy.array(list(zip(f1, f2)))
        plt.scatter(f1, f2, c='black', s=7)

        cx = npy.random.randint(0, npy.max(X)-20, size=self.k)
        cy = npy.random.randint(0, npy.max(X)-20, size=self.k)
        c = npy.array(list(zip(cx, cy)), dtype=npy.float32)

        plt.scatter(cx, cy, marker='*', s=200, c='g')

        c_old = npy.zeros(c.shape)
        clusters = npy.zeros(len(X))
        error = self.distance(c, c_old, None)

        while error != 0:

            for i in range(len(X)): # Closest cluster
                distances = self.distance(X[i], c)
                cluster = npy.argmin(distances)
                clusters[i] = cluster

            c_old = deepcopy(c)

            for i in range(self.k):
                points = [X[j] for j in range(len(X)) if clusters[j] == i]
                c[i] = npy.mean(points, axis=0)
            error = self.distance(c, c_old, None)

        colors = ['b', 'r', 'g', 'y', 'c', 'm']
        fig, ax = plt.subplots()
        for i in range(self.k):
                points = npy.array([X[j] for j in range(len(X)) if clusters[j] == i])
                ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax.scatter(c[:, 0], c[:, 1], marker='*', s=200, c='#050505')

        plt.show()


k_med = Kmed(k=3, file_name='Kmed.csv')
k_med.load_dataset()
k_med.run()
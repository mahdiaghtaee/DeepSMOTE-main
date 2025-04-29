import numpy as np
from sklearn.neighbors import NearestNeighbors

class SMOTEGenerator:
    @staticmethod
    def biased_get_class(dec_x, dec_y, c):
        x_class = dec_x[dec_y == c]
        y_class = dec_y[dec_y == c]
        return x_class, y_class

    @staticmethod
    def generate(X, label, n_to_sample, fake_label):
        n_neigh = 6  # include itself
        nn_model = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
        nn_model.fit(X)
        dist, ind = nn_model.kneighbors(X)
        base_idx = np.random.choice(len(X), n_to_sample)
        neigh_idx = np.random.choice(range(1, n_neigh), n_to_sample)
        X_base = X[base_idx]
        X_neigh = X[ind[base_idx, neigh_idx]]
        samples = X_base + np.random.rand(n_to_sample, 1) * (X_neigh - X_base)
        return samples, [fake_label] * n_to_sample

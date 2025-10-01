import numpy as np

class MiniBatchKMeans:
    def __init__(self, n_clusters=3, max_iters=1000, batch_size=256, epsilon=1e-6):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.eps = epsilon
        self.centroids = None
        self.counts = None  # track how many points per cluster
        self.labels = None

    def _init_centroids(self, X):
        m = X.shape[0]
        random_i = np.random.choice(m, self.n_clusters, replace=False)
        self.centroids = X[random_i]
        self.counts = np.zeros(self.n_clusters)
        self.labels = []


    def fit(self, X):
        

        if self.centroids is None:
            self._init_centroids(X)

        for iteration in range(self.max_iters):
            self.labels = []
            for x in X:
                # Distance to centroids
                dists = np.sum((self.centroids - x) ** 2, axis=1)
                k = np.argmin(dists)

                # Incremental update rule
                self.counts[k] += 1
                eta = 1 / self.counts[k]
                self.centroids[k] = (1 - eta) * self.centroids[k] + eta * x
                # assign the cluster k to x in labels
                self.labels.append(k)
            #test
        return self.centroids ,np.array(self.labels)
    

    def partial_fit(self, X):
        self.labels = []
        #Update centroids with a batch X (incremental learning) 
        if self.centroids is None:
            self._init_centroids(X)

        for x in X:
            # Distance to centroids
            dists = np.sum((self.centroids - x) ** 2, axis=1)
            k = np.argmin(dists)

            # Incremental update rule
            self.counts[k] += 1
            eta = 1 / self.counts[k]
            self.centroids[k] = (1 - eta) * self.centroids[k] + eta * x
            # assign the cluster k to x in labels
            self.labels.append(k)

        return self.centroids ,np.array(self.labels)
            



    def predict_failure(self, X, centroid):
        # Predict if new data point X is close to failure centroid
        dists = np.sum((centroid - X) ** 2)
        prob_failure = np.exp(-dists) # decay function to get probability , bigger d means smaller prob
        return prob_failure 

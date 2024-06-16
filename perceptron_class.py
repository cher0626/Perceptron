import numpy as np
class Perceptron:
    def __init__(self, eta = 0.1, n_iter = 10, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.01, size=1 + x.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, actual in zip(x,y):
                update = self.eta * (actual - self.prediction(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                #add 1 to errors if update!=0
                errors += int(update != 0)
            self.errors_.append(errors)
        return self
    
    def weighted_sum(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def prediction(self, x):
        return np.where(self.weighted_sum(x) >= 0, 1, -1)



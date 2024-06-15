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


#train iris dataset (setosa and versicolor)
#plot all the data out and also the decision boundary for 1-6 iteration

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

colorList = ['blue', 'red']
markerList = ['o', 'x']
labelList = ["Setosa", "Versicolor"]

x = df.iloc[0:100, [0,2]].values
y = df.iloc[0:100, 4].values
y = np.where(y=="Setosa", -1, 1)

for j in range (1,7):

    plt.subplot(2, 3, j)
    plt.title("iteration = " + str(j))
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')

    iris_train = Perceptron(eta=0.1, n_iter=j)
    iris_train.fit(x,y)
    
    slope = -iris_train.w_[1]/iris_train.w_[2]
    y_intercept = -iris_train.w_[0]/iris_train.w_[2]
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl, 0], y=x[y==cl, 1], c=colorList[idx], marker=markerList[idx], label=labelList[idx])
    
    plt.plot(x, (slope*x + y_intercept))
    plt.legend(loc='upper left')
    
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
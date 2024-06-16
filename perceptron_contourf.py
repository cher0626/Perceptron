from perceptron_class import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#create a graph with decision region
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    colors = ['red', 'blue', 'lightgreen']
    markers = ['o', 'x']
    labelList = ["Setosa", "Versicolor"]
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #finding the best graph size and plate
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min() -1, x[:, 1].max()+1
    
    #np.arange(a, b, c) creates a array with min=a, max=b, interval=c
    #np.meshgrid(x,y) creates a matrix
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    #finding the predicted area which is 1 or -1
    z = classifier.prediction(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    #create the plot
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl, 0], y=x[y==cl, 1], c=colors[idx], marker=markers[idx], label=labelList[idx])
    

df = pd.read_csv('iris.csv')
x = df.iloc[0:100, [0,2]].values
y = df.iloc[0:100, 4].values
y = np.where(y=="Setosa", -1, 1)

iris_train = Perceptron(eta=0.1, n_iter=50).fit(x,y)
plot_decision_regions(x, y, classifier = iris_train)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
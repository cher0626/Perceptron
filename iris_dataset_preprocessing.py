import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

y = df.iloc[0:100, 4].values
y = np.where(y == 'Setosa', -1, 1)
x = df.iloc[0:100,[0,2]].values

colorList = ['blue', 'red']
markerList = ['o', 'x']
labelList = ["Setosa", "Versicolor"]

for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=x[y==cl, 0], y=x[y==cl, 1], c=colorList[idx], marker=markerList[idx], label=labelList[idx])

'''for i in range (100):
    if y[i] == -1:
        plotColor = "blue"
        plotMarker = "o"
        plotLabel = "Setosa"
    else: 
        plotColor = "red"
        plotMarker = "x"
        plotLable = "Versicolor"

    plt.scatter(x[i][0], x[i][1], color = plotColor, marker = plotMarker)'''

plt.legend(loc='upper left')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()
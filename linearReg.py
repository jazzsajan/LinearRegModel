import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

cpu_data = pd.read_csv("https://raw.githubusercontent.com/alpeca/LinearRegModel/main/machine.data.csv", sep = ',')

print("\nPreview of CPU Data")

cpu_data.head()

# analyzing data for preprocessing
print("------------------------------------------")
print("CPU Data Information\n")
print(cpu_data.info()) # each col has 209 non null values -> no empty values, no null values


print("------------------------------------------")
print("Reporting Duplicates\n")
print(cpu_data.duplicated()) # all 209 values show false for duplicate

cpu_data = cpu_data.drop(['vendor name','model name', 'erp'], axis = 1)

print("------------------------------------------")
print("\nDescription of CPU Data Frame")
print(cpu_data.describe())

# impact of each attribute on prp(Y)
print("------------------------------------------")
print("\nImpact of Attributes from Highest to Lowest")
print(abs(cpu_data.corr())['prp'].sort_values(ascending = False))
print("------------------------------------------")

# correlation matrix
print("\nCorrelation Matrix")
corr = cpu_data.corr()
sns.heatmap(corr, annot = True)

# splitting data into training and testing sets
train_ratio = 0.8
train_size = int(train_ratio * len(cpu_data))
train_data = cpu_data[:train_size]
test_data = cpu_data[train_size:]

# dropping parameters that do not significantly affect prp(Y)
train_data = train_data.drop(['myct'], axis = 1)
test_data = test_data.drop(['myct'], axis = 1)

# seperate attributes(X) and prp(Y)

X = train_data.values
Y = train_data['prp'].values.reshape(167, 1)

X_test = test_data.values
Y_test = test_data['prp'].values.reshape(42, 1)

# adding placeholder for x0 in matrix

X = np.vstack((np.ones((X.shape[0], )), X.T)).T
X_test = np.vstack((np.ones((X_test.shape[0], )), X_test.T)).T

# linear regression model

def model(X, Y, learning_rate, iteration):
    m = Y.size
    w = np.zeros((X.shape[1], 1))
    cost_doc = []

    for i in range(iteration):

        y_pred = np.dot(X, w)

        # cost function
        cost = (1/(2*m))*np.sum(np.square(y_pred - Y))

        # gradient descent using ssr
        d_w = (1/m)*np.dot(X.T, y_pred - Y)
        w = w - learning_rate*d_w

        cost_doc.append(cost)

        if(i%(iteration/10) == 0):
          print("cost:", cost)

    return w, cost_doc

iteration = 10000000
learning_rate = 0.000000001
w, cost_doc = model(X, Y, learning_rate = learning_rate, iteration = iteration)

y_pred = np.dot(X_test, w)
test_error = (1/X_test.shape[0])*np.sum(np.abs(y_pred - Y_test))

y_trainPred = np.dot(X, w)
train_error = (1/X.shape[0])*np.sum(np.abs(y_trainPred - Y))

# test statistics
mse = (mean_squared_error(Y, y_trainPred))
r2 = r2_score(Y, y_trainPred)

mse_test = (mean_squared_error(Y_test, y_pred))
r2_test = r2_score(Y_test, y_pred)


formattedLR=f'{learning_rate:.9f}'
print("Learning Rate:", formattedLR)
print("Iterations:", iteration)
print("Test Error:", round((test_error), 2))
print("Test Accuracy:", round((1- test_error), 2))
print("Test MSE: ", mse_test)
print("Test R-Squared: ", r2_test)
print("Train Error:", round((train_error), 2))
print("Train Accuracy:", round((1- train_error), 2))
print("Training MSE: ", mse)
print("Training R-Squared: ", r2)
print("Weights: ", w)

test_mse = [0.017236679458482618, 0.007275644315950057, 0.01738249751898345, 0.006914606153330225, 0.006271056991611179, 0.005867715644692224, 0.005867715644692224, 0.049288881564934404, 0.04696803371459169, 0.05083220454724993]
iterations = [10000000, 15000000, 10000000, 15000000, 20000000, 25000000,10000000, 15000000, 15000000, 10000000]

plt.scatter(iterations, test_mse)

plt.xlabel("Iterations")
plt.ylabel("Test MSE")
plt.title("Test MSE vs. Iterations")
plt.show()

training_mse = [0.002760609737583745, 0.007275644315950057, 0.0028865572143944307, 0.0005258685293344188, 0.0004184817404708458, 0.005867715644692224, 0.003517148979608694, 0.005946461976050696, 0.003472309167237979, 0.003761150906443901]
iterations = [10000000, 15000000, 10000000, 15000000, 20000000, 25000000,10000000, 15000000, 15000000, 10000000]

plt.scatter(iterations, training_mse)

plt.xlabel("Iterations")
plt.ylabel("Training MSE")
plt.title("Training vs. Iterations")
plt.show()

rng = np.arange(0, iteration)
plt.plot(rng, cost_doc)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs. Iterations")
plt.show()

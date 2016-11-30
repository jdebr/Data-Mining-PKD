import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

dataFile = "new.data"

dataSet = list()
# labels are really just the feature we are interested in imputing
labels = list()

with open(dataFile,'r') as f:
  rawData = f.read().splitlines()
  for line in rawData:
    dat = line.split(',')
    skip = False
    # currently 6 features
    for i in range(1,7):
      if dat[i].strip() == 'M' or dat[i].strip() == '':
        #print("skipped")
        skip = True
    # if there are no missing values, add to dataset  
    if not skip:
      # add features: 1-watertemp, 2-discharge, 3-conductance, 5-airTemp
      newDat = [float(dat[2]), float(dat[3]), float(dat[5])]
      dataSet.append(newDat)
      # imputing watertemp
      labels.append(float(dat[1]))
      
print(len(dataSet))
print(len(labels))

# Split data for testing imputation
train = dataSet[::2]
trainLabels = labels[::2]
test = dataSet[1::2]
testLabels = labels[1::2]

trainData = np.array(train)
testData = np.array(test)

# Set up KNNI using sklearn.neighbors.KNeighborsRegressor
neigh = neighbors.KNeighborsRegressor(n_neighbors=5,weights='distance')
neigh.fit(trainData, trainLabels)
guess = neigh.predict(testData)
print(guess[45])
print(testLabels[45])
print(guess[200])
print(testLabels[200])
print("MSE: " + str(mean_squared_error(guess, testLabels)))


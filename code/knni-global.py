import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import os
import csv

trainPath = '../data/cleaned data/Local/'
imputePath = '../data/cleaned data/Global/'
writePath = '../data/imputed data/Global/'


dataSet = list()
labels = list()

# Train model from datasets with temp values 
for fn in os.listdir(trainPath):
  dataFile = trainPath + fn
  with open(dataFile,'r') as f:
    rawData = f.read().splitlines()
    for ind, line in enumerate(rawData):
      dat = line.split(',')
      skip = False
      # currently 6 features
      for i in [1,2,5]:
        if dat[i].strip() == 'M' or dat[i].strip() == '' or dat[i].strip() == 'Ssn' or dat[i].strip() == 'Ice':
          #print("skipped")
          skip = True
      # if there are no missing values, add to dataset  
      if not skip:
        # add features: 1-watertemp, 2-discharge, 3-conductance, 4-gage height, 5-airTemp, 6-hourly precip
        newDat = [float(dat[2]), float(dat[5])]
        dataSet.append(newDat)
        # imputing watertemp
        labels.append(float(dat[1]))
        
# Set up KNNI using sklearn.neighbors.KNeighborsRegressor
trainData = np.array(dataSet)
neigh = neighbors.KNeighborsRegressor(n_neighbors=5,weights='distance')
neigh.fit(trainData, labels)

# Impute in each file 
for fn in os.listdir(imputePath):
  dataFile = imputePath + fn
  newFile = writePath + fn + "-imputed"
  
  imputationSet = list()
  impIndex = list()
  originalData = list()

  # build imputation set 
  with open(dataFile,'r') as f:
    rawData = f.read().splitlines()
    for ind, line in enumerate(rawData):
      dat = line.split(',')
      originalData.append(dat)
      skip = False
      # if dat[2] == ' Ssn' or dat[5] == ' Ssn':
        # print("RAT!")
        # print(fn)
        # print(ind)
      # if water temp is missing but discharge and airtemp are there, add to imputationSet 
      if (dat[1].strip() == '' or dat[1].strip() == 'Ssn') and not dat[2].strip() == '' and not dat[2].strip() == 'Ssn' and not dat[2].strip() == 'Eqp' and not dat[2].strip() == 'Ice' and not dat[2].strip() == 'Rat' and not dat[5].strip() == 'M':
        newDat = [float(dat[2]), float(dat[5])]
        imputationSet.append(newDat)
        impIndex.append(ind)
        
  
  # This section actually imputes unknown data, writing to a separate file
  testData = np.array(imputationSet)
  imputedValues = neigh.predict(testData)
  
  # Add imputed values to original data 
  valCount = 0
  for i in impIndex:
    originalData[i][1] = round(imputedValues[valCount], 1)
    valCount += 1
    
  # Write all data to new file 
  with open(newFile, 'w') as w:
    writer = csv.writer(w, lineterminator='\n')
    writer.writerows(originalData)

  ''' 
  # This section runs an experiment demonstrating efficacy of imputation 
  
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
  
  '''


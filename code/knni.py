import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import os
import csv

dataPath = '../data/cleaned data/'
writePath = '../data/imputed data/'

#for fn in os.listdir(dataPath):
for i in range(1):
  fn = '06037500-cleaned'

  dataFile = dataPath + fn
  newFile = writePath + fn + "-imputed"
  originalData = list()
  dataSet = list()
  imputationSet = list()
  impIndex = list()
  # labels are really just the feature we are interested in imputing, in this case water temp
  labels = list()

  with open(dataFile,'r') as f:
    rawData = f.read().splitlines()
    for ind, line in enumerate(rawData):
      dat = line.split(',')
      originalData.append(dat)
      skip = False
      # currently 6 features
      for i in range(1,7):
        if dat[i].strip() == 'M' or dat[i].strip() == '':
          #print("skipped")
          skip = True
      # if there are no missing values, add to dataset  
      if not skip:
        # add features: 1-watertemp, 2-discharge, 3-conductance, 4-gage height, 5-airTemp, 6-hourly precip
        newDat = [float(dat[2]), float(dat[5])]
        dataSet.append(newDat)
        # imputing watertemp
        labels.append(float(dat[1]))
      # if water temp is missing but discharge and airtemp are there, add to imputationSet 
      elif dat[1].strip() == '' and not dat[2].strip() == '' and not dat[5].strip() == 'M':
        newDat = [float(dat[2]), float(dat[5])]
        imputationSet.append(newDat)
        impIndex.append(ind)
        
  #print(len(dataSet))
  #print(len(labels))
  
  # This section actually imputes unknown data, writing to a separate file
  trainData = np.array(dataSet)
  testData = np.array(imputationSet)
  
  # Set up KNNI using sklearn.neighbors.KNeighborsRegressor
  neigh = neighbors.KNeighborsRegressor(n_neighbors=5,weights='distance')
  neigh.fit(trainData, labels)
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


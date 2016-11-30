import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
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
      newDat = [float(dat[2]), float(dat[5])]
      dataSet.append(newDat)
      # imputing watertemp
      labels.append(float(dat[1]))
      
#print(len(dataSet))
#print(len(labels))

# Split data for testing imputation
train = dataSet[::2]
trainLabels = labels[::2]
test = dataSet[1::2]
testLabels = labels[1::2]

trainData = np.array(train)
testData = np.array(test)

# Plot to visualize
'''
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
xpts, ypts = np.hsplit(np.array(dataSet), 2)
plt.plot(xpts, ypts, '.')
plt.show()
'''

'''
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


# Set up FKMI using skfuzzy.cluster.cmeans
fuzz_train = np.transpose(trainData) # skfuzzy takes horizontal array
#ncenters = 5 # 2 clusters = best FPC value after tuning
for ncenters in [2,3,4,5,6,7,8,9]:
  print("###### {0} CLUSTERS ########".format(ncenters))
  
  # Clustering 
  cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    fuzz_train, ncenters, 2, error=0.005, maxiter=1000, init=None )

  # Calculate water temps of centroids 
  centroid_temps = list()
  for i in range(ncenters):
    num = 0.0
    den = 0.0
    for j in range(len(trainLabels)):
      memb = u[i][j]
      num += (memb * trainLabels[j])
      den += (memb)
    temp = num/den 
    centroid_temps.append(temp)
      

  #print(u)
  #print(cluster_membership)
  #print(cntr)
  #print(fpc)

  # Graphing cluster centroids 
  # xpts, ypts = np.hsplit(cntr, 2)
  # plt.plot(xpts, ypts, '.')
  # plt.show()

  # Predict cluster membership of test set 
  fuzz_test = np.transpose(testData)
  u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    fuzz_test, cntr, 2, error=0.005, maxiter=1000)
    
  # Perform imputation of missing water temps 
  cluster_membership = np.argmax(u, axis=0)
  #print("Centroids: " + str(cntr))
  #print("Centroid temperatures: " + str(centroid_temps))
  #print("Membership: " + str(cluster_membership))
  #print(u)

  # Imputation as sum of membership functions times watertemp of centroids 
  imputed_temps = list()
  for i in range(len(testData)):
    imputed_temp = 0.0
    for j in range(ncenters):
      memb = u[j][i]
      imputed_temp += (memb * centroid_temps[j])
    imputed_temps.append(imputed_temp)
    
  #print(testLabels)
  #print(imputed_temps)
  
  # Experiment 
  print("MSE: " + str(mean_squared_error(imputed_temps, testLabels)))
    

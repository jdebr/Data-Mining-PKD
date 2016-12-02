# This script will strip unwanted characters from the combined data files
import re
import os

p = re.compile(r"[]|\\[']")
c = ','
dataPath = '../data/Combined Files/Combined Files/'
writePath = '../data/cleaned data/'

#for fn in os.listdir(dataPath):
fn = 
myFile = dataPath + fn
newFile = writePath + fn.split()[0] + "-cleaned"
with open(myFile, 'r') as f:
  with open(newFile, 'w') as w:
    #w.write("#Date WaterTemp(C) Discharge(CFS) Conductance GageHeight(ft) AirTemp(F) HourlyPrecip(inches)\n")
    for x in f:
      x = p.sub('',x)
      d = x.split(',')
      # Select the features we want to keep...these are numbered from left to right in the combined data file
      # Also note you must use the slice notation for the sum to work, even if only selecting one column
      n = sum([d[2:3], d[4:5], d[6:7], d[8:9], d[10:11], d[15:16], d[20:21]],[])
      newline = c.join(n) + '\n'
      w.write(newline)
    
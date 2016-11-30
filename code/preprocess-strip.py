import re

p = re.compile(r"[]|\\['n]")
c = ','

myFile = "./yellowstone.txt.txt"

with open(myFile, 'r') as f:
  with open("new.data", 'w') as w:
    w.write("#Date WaterTemp(C) Discharge(CFS) Conductance GageHeight(ft) AirTemp(F) HourlyPrecip(inches)\n")
    for x in f:
      x = p.sub('',x)
      d = x.split(',')
      n = sum([d[2:3], d[4:5], d[6:7], d[8:9], d[10:11], d[15:16], d[20:21]],[])
      #n = str(d[2]) + str(d[4]) + str(d[5]) + str(d[6]) + str(d[8]) + str(d[9]) + str(d[10]) + str(d[16]) + str(d[19]) + str(d[20]) + str(d[21]) + str(d[22]) + str(d[24]) + str(d[26]) + str(d[30])
      newline = c.join(n) + '\n'
      w.write(newline)
    
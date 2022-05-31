"""
Check correlation between the mass variables I have created and diphoton mass
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
import pandas as pd
import numpy as np

df = pd.read_parquet(sys.argv[1])

#df = df[df.category!=8]

plt.hist(df[df.process_id==33].sublead_lepton_pt, bins=50, range=(-10,100), density=True, alpha=0.5, label="signal") #mx=300
plt.hist(df[df.process_id==0].sublead_lepton_pt, bins=50, range=(-10,100), density=True, alpha=0.5, label="data")
plt.legend()
plt.savefig("sublead_lepton.png")
plt.clf()

for cat in range(1, 8):
  df2 = df[df.category==cat]
  plt.hist(df2[df2.process_id==33].sublead_lepton_pt, bins=50, range=(-10,100), density=True, alpha=0.5, label="signal") #mx=300
  plt.hist(df2[df2.process_id==0].sublead_lepton_pt, bins=50, range=(-10,100), density=True, alpha=0.5, label="data")
  plt.legend()
  plt.savefig("sublead_lepton_%d.png"%cat)
  plt.clf()

for id in np.unique(df.sublead_lepton_id):
  print(id)
  print(df[(df.sublead_lepton_id == id)&(df.process_id==0)].sublead_lepton_pt.min())
  print(df[(df.sublead_lepton_id == id)&(df.process_id!=0)].sublead_lepton_pt.min())




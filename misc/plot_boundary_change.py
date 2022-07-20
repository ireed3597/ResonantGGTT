import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import common

nominal_masses = [260,270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900,1000]

with open(sys.argv[1], "r") as f:
  optim_results = json.load(f)

masses = []
bounds = []
for entry in optim_results:
  m = int(entry["sig_proc"].split("M")[1])
  masses.append(m)
  bounds.append(entry["category_boundaries"])

masses = np.array(masses)
bounds = np.array(bounds)

bounds = bounds[np.argsort(masses)]
masses = masses[np.argsort(masses)]

#delete 0 boundary
bounds = bounds[:,1:]

for i in range(len(bounds[0])):
  plt.plot(masses, bounds[:,i])
  plt.plot(nominal_masses, bounds[np.isin(masses, nominal_masses)][:,i], 'r.')

plt.savefig("bound_change.png")
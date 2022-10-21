import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
#plt.rcParams['figure.constrained_layout.use'] = True
import numpy as np

import sys
import json 

with open(sys.argv[1], "r") as f:
  model = json.load(f)

mx = []
names = ["Signal Efficiency", r"$\bar{m}_{\gamma\gamma}$", r"$\sigma$", r"$\beta_l$", r"$m_l$", r"$\beta_r$", r"$m_r$"]
parameters = [[] for each in names]

for key in model.keys():
  mx.append(int(key))

  parameters[0].append(model[key][0])
  for i in range(1, 7):
    parameters[i].append(model[key][1][i])

for i in range(len(names)):
  plt.scatter(mx, parameters[i])


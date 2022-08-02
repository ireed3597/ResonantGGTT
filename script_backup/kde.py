import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import json

if __name__=="__main__":
  df = pd.read_parquet(sys.argv[1])
  with open(sys.argv[2], "r") as f:
    summary = json.load(f)
  proc_dict = summary["sample_id_map"]

  bkgs_ids = [proc_dict[proc] for proc in common.bkg_procs["Diphoton"]]
  bkgs_ids = [proc_dict[proc] for proc in common.bkg_procs["all"]]
  df = df[np.isin(df.process_id, bkgs_ids)]

  plt.hist(df.Diphoton_mass, bins=50, range=(100,150), weights=df.weight_central, density=True)
  plt.savefig("kde_test.png")

  x = df.Diphoton_mass
  w = df.weight_central
  
  x = x[w>0]
  w = w[w>0]

  #bandwidth = 0.9 * x.std() * np.power(len(x), -0.2)
  bandwidth = 5
  print(bandwidth)
  kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
  print("Training kde")
  kde.fit(x[:, np.newaxis], sample_weight=w)
  N = 100000
  print("Making samples")
  samples = np.sort(kde.sample(N).T[0])
  print(samples)
  print("Made")

  plt.hist(samples, bins=50, range=(100,150), histtype='step', density=True)
  plt.savefig("kde_test2.png")
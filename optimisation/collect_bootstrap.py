import pandas as pd
import numpy as np
import json
import os
import sys
import common
import matplotlib.pyplot as plt

def AMS(s, b):
  s, b = np.array(s), np.array(b)
  AMS2 = 2 * ( (s+b)*np.log(1+s/b) -s ) #calculate squared AMS for each category
  return np.sqrt(AMS2)

bootstraps = []
for directory in sorted(os.listdir(sys.argv[1])):
  print(directory)
  if "bootstrap" in directory:
    with open(os.path.join(sys.argv[1], directory, "optim_results.json"), "r") as f:
      bootstraps.append(json.load(f))

sig_procs = [each["sig_proc"] for each in bootstraps[0] if "optimal_limit" in each.keys()]
print(sig_procs)

nsigs = {sig_proc:[] for sig_proc in sig_procs}
nbkgs = {sig_proc:[] for sig_proc in sig_procs}
limits = {sig_proc:[] for sig_proc in sig_procs}
for bootstrap in bootstraps:
  for entry in bootstrap:
    if "optimal_limit" in entry.keys():
      nsigs[entry["sig_proc"]].append(entry["nsigs"])
      nbkgs[entry["sig_proc"]].append(entry["nbkgs"])
      limits[entry["sig_proc"]].append(entry["optimal_limit"])

for sig_proc in sig_procs:
  #if sig_proc != "XToHHggTauTau_M1000": continue
  print(sig_proc)
  nCats = np.array([len(each) for each in nsigs[sig_proc]])
  print(nCats)
  assert (nCats[0] == nCats).all()
  nCats = nCats[0]

  nsigs_p = np.array(nsigs[sig_proc])
  means = nsigs_p.mean(axis=0)
  error = nsigs_p.std(axis=0)
  plt.errorbar(np.arange(len(means)), means[::-1], error[::-1], fmt='o')
  plt.xlabel("CAT")
  plt.ylabel("Signal efficiency")
  plt.title(sig_proc)
  plt.savefig(os.path.join(sys.argv[1], "%s_sig.png"%sig_proc))
  plt.clf()

  nbkgs_p = np.array(nbkgs[sig_proc])
  means = nbkgs_p.mean(axis=0)
  error = nbkgs_p.std(axis=0)
  plt.errorbar(np.arange(len(means)), means[::-1], error[::-1], fmt='o')
  plt.xlabel("CAT")
  plt.ylabel("bkg efficiency")
  plt.title(sig_proc)
  plt.savefig(os.path.join(sys.argv[1], "%s_bkg.png"%sig_proc))
  plt.clf()

  ams_p = AMS(nsigs_p, nbkgs_p)
  means = ams_p.mean(axis=0)
  error = ams_p.std(axis=0)
  plt.errorbar(np.arange(len(means)), means[::-1], error[::-1], fmt='o')
  plt.xlabel("CAT")
  plt.ylabel("ams")
  plt.title(sig_proc)
  plt.savefig(os.path.join(sys.argv[1], "%s_ams.png"%sig_proc))
  plt.clf()

  ams_p = AMS(nsigs_p, nbkgs_p.mean(axis=0))
  means = ams_p.mean(axis=0)
  error = ams_p.std(axis=0)
  print(error[::-1][0])
  plt.errorbar(np.arange(len(means)), means[::-1], error[::-1], fmt='o')
  plt.xlabel("CAT")
  plt.ylabel("ams")
  plt.title(sig_proc)
  plt.savefig(os.path.join(sys.argv[1], "%s_ams_bkg_const.png"%sig_proc))
  plt.clf()

limits_p = np.array([limits[sig_proc] for sig_proc in sig_procs])
means = limits_p.mean(axis=1)
error = limits_p.std(axis=1)
plt.errorbar(np.arange(len(means)), means, error, fmt='o')
plt.xlabel("Sig proc")
plt.ylabel("limit")
plt.savefig(os.path.join(sys.argv[1], "limits.png"))
plt.clf()

limits_p = np.array([limits[sig_proc] for sig_proc in sig_procs]).T
limits_p = limits_p / limits_p.mean(axis=0)
means = limits_p.mean(axis=0)
error = limits_p.std(axis=0)
plt.errorbar(np.arange(len(means)), means, error, fmt='o')
plt.xlabel("Sig proc")
plt.ylabel("limit")
plt.savefig(os.path.join(sys.argv[1], "limits_normed.png"))
plt.clf()
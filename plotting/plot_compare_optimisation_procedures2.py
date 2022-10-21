import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (20,16)

import sys
import json
import os
import common
import numpy as np

masses = [260, 270, 280, 290, 300,320,350,400,450,500,550,600,650,700,750,800,900,1000]

grid_search_dir = sys.argv[1]
new_optim_results = sys.argv[2]
savedir = sys.argv[3]

grid_search_limits = []
for m in masses:
  with open(os.path.join(grid_search_dir, "XToHHggTauTau_M%d"%m, "optimisation_results.json"), "r") as f:
    results = json.load(f)
  grid_search_limits.append(results["optimal_limit"])

new_procedure_limits = []
new_procedure_masses = []
with open(new_optim_results, "r") as f:
  results = json.load(f)
  for entry in results:
    if "optimal_limit" not in entry.keys(): continue
    #print(entry)
    new_procedure_masses.append(common.get_MX_MY(entry["sig_proc"])[0])
    new_procedure_limits.append(entry["optimal_limit"])

grid_search_limits = np.array(grid_search_limits)

new_procedure_limits = np.array(new_procedure_limits)
new_procedure_masses = np.array(new_procedure_masses)
new_procedure_limits = new_procedure_limits[np.argsort(new_procedure_masses)]

f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

axs[0].plot(masses, grid_search_limits, label="Grid search")
axs[0].plot(masses, new_procedure_limits, label="New procedure")
axs[0].legend()
axs[0].set_ylabel("Approximate 95% CL limit")

axs[1].plot(masses, grid_search_limits/new_procedure_limits)
axs[1].set_ylabel("Grid / new")
axs[1].set_xlabel(r"$m_X$")

savename = os.path.join(grid_search_dir, new_optim_results).replace("/", "_")
savename = savename.replace("_optim_results.json", "")
savename = savename.replace("Outputs_GridSearch_vs_New_Graviton", "")
savename += ".png"

f.savefig(os.path.join(savedir, savename))
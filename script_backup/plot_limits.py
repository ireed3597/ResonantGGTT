import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import numpy as np
import sys
import tabulate
import pandas as pd
import os

BR_H_GG = 2.27e-3
BR_H_TT = 6.27e-2
BR_H_BB = 5.84e-1

BR_HH_GGTT = 2 * BR_H_GG * BR_H_TT
BR_HH_GGBB = 2 * BR_H_GG * BR_H_BB

nominal_masses = [260,270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900,1000]

# def getLimits(results_path):
#   with open(results_path, "r") as f:
#     results = f.readlines()
 
#   mx = []
#   for i, line in enumerate(results):
#     if "MX" not in line:
#       expected_starting_idx = i
#       break
#     else:
#       mx.append(int(line.split("=")[1][:-1]))
  
#   limits = [[] for i in range(5)]

#   for i in range(expected_starting_idx, len(results), 5):
#     for j in range(5):
#       limits[j].append(float(results[i+j].split("<")[1][:-1]))

#   mx = np.array(mx)
#   limits = np.array(limits)
#   idx = np.argsort(mx)
#   mx = mx[idx]
#   limits = limits[:,idx]

#   #convert from pb to fb
#   #limits = limits / 1000

#   return mx, limits

def getLimits(results_path):
  with open(results_path, "r") as f:
    results = f.readlines()

  mx = []
  for line in results:
    m = int(line.split(".")[0].split("_")[-1])
    mx.append(m)
  mx = np.array(sorted(set(mx)))
  
  limits = np.zeros((5, len(mx)))
  limits_no_sys = np.zeros((5, len(mx)))

  for line in results:
    m = int(line.split(".")[0].split("_")[-1])
    idx1 = np.where(mx == m)[0][0]
    if "2.5%" in line:
      idx2=0
    elif "16.0%" in line:
      idx2=1
    elif "50.0%" in line:
      idx2=2
    elif "84.0%" in line:
      idx2=3
    elif "97.5%" in line:
      idx2=4
    
    limit = float(line.split("r < ")[1])

    if "no_sys" in line:
      limits_no_sys[idx2][idx1] = limit
    else:
      limits[idx2][idx1] = limit

  return mx, limits, limits_no_sys
    
def plotLimits(mX, limits, ylabel, savename=None):
  plt.scatter(mX, limits[2], zorder=3, facecolors="none", edgecolors="blue")
  plt.scatter(mx[np.isin(mx, nominal_masses)], limits[2][np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  plt.plot(mX, limits[2], 'b--', zorder=3, label="Expected 95% CL limit")
  plt.fill_between(mX, limits[1], limits[3], zorder=2, facecolor="green", label=r"$\pm$ $1\sigma$")
  plt.fill_between(mX, limits[0], limits[4], zorder=1, facecolor="yellow", label=r"$\pm$ $2\sigma$")
  plt.xlabel(r"$m_X$")
  plt.ylabel(ylabel)
  
  plt.legend()
  bottom, top = plt.ylim()
  #plt.ylim(bottom, top*10)
  
  mplhep.cms.label(llabel="Work in Progress", data=True, lumi=137.2, loc=0)

  if savename!=None:
    plt.savefig(savename+".png")
    plt.savefig(savename+".pdf")
    plt.yscale("log")
    plt.savefig(savename+"_log.png")
    plt.savefig(savename+"_log.pdf")
    plt.clf()

def plotSystematicComparison(mx, limits, limits_no_sys, savename):
  ratio = limits[2]/limits_no_sys[2]
  plt.plot(mx, ratio)
  plt.scatter(mx[np.isin(mx, nominal_masses)], ratio[np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  plt.xlabel(r"$m_X$")
  plt.ylabel("Exp. limit w / wo systematics")

  plt.legend()

  plt.savefig(savename+".png")
  plt.savefig(savename+".pdf")
  plt.clf()

def plotSystematicComparison2(mx, limits, limits_no_sys, ylabel, savename):
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  ratio = limits[2]/limits_no_sys[2]
  
  #axs[0].scatter(mx, limits[2], zorder=3, facecolors="none", edgecolors="blue")
  #axs[0].scatter(mx[np.isin(mx, nominal_masses)], limits[2][np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  #axs[0].plot(mx, limits[2], 'b--', zorder=3, label="Expected 95% CL limit")
  axs[0].plot(mx, limits[2], zorder=3, label="Expected 95% CL limit")

  #axs[0].scatter(mx, limits_no_sys[2], zorder=3, facecolors="none", edgecolors="blue")
  #axs[0].scatter(mx[np.isin(mx, nominal_masses)], limits_no_sys[2][np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red", label="Nominal masses")
  #axs[0].plot(mx, limits_no_sys[2], 'b--', zorder=3, label="Expected 95% CL limit (no sys.)")
  axs[0].plot(mx, limits_no_sys[2], zorder=3, label="Expected 95% CL limit (no sys)")

  axs[0].set_ylabel(ylabel)
  axs[0].legend()
  axs[0].set_yscale("log")
  
  axs[1].plot(mx, ratio)
  axs[1].scatter(mx[np.isin(mx, nominal_masses)], ratio[np.isin(mx, nominal_masses)], zorder=4, facecolors="none", edgecolors="red")
  axs[1].set_ylabel("Ratio")
  axs[1].set_xlabel(r"$m_X$")

  plt.savefig(savename+".png")
  plt.savefig(savename+".pdf")
  plt.clf()


def tabulateLimits(mx, limits, path):
  df = pd.DataFrame({"MX": mx, "Expected 95% CL Limit [fb]": limits[2]})

  table = tabulate.tabulate(df, headers='keys', floatfmt=".4f")
  
  with open(os.path.join(path, "param_test_results.txt"), "w") as f:
    f.write(table)
  with open(os.path.join(path, "param_test_results.tex"), "w") as f:
    f.write(df.to_latex(float_format="%.4f"))
  df.to_csv(os.path.join(path, "param_test_results.csv"), float_format="%.4f")

mx, limits, limits_no_sys = getLimits(sys.argv[1])
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_br"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_br_no_sys"), exist_ok=True)
os.makedirs(os.path.join(sys.argv[2], "Limits_xs_no_sys"), exist_ok=True)

ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow HH \rightarrow \gamma\gamma\tau\tau)$ [fb]"
plotLimits(mx, limits, ylabel, os.path.join(sys.argv[2], "Limits_xs_br", "limits"))
plotLimits(mx, limits_no_sys, ylabel, os.path.join(sys.argv[2], "Limits_xs_br_no_sys", "limits_no_sys"))

ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow HH)$ [fb]"
plotLimits(mx, limits / BR_HH_GGTT, ylabel, os.path.join(sys.argv[2], "Limits_xs", "limits"))
plotLimits(mx, limits_no_sys / BR_HH_GGTT, ylabel, os.path.join(sys.argv[2], "Limits_xs_no_sys", "limits_no_sys"))

tabulateLimits(mx, limits, os.path.join(sys.argv[2], "Limits_xs_br"))
tabulateLimits(mx, limits / BR_HH_GGTT, os.path.join(sys.argv[2], "Limits_xs"))

tabulateLimits(mx, limits_no_sys, os.path.join(sys.argv[2], "Limits_xs_br_no_sys"))
tabulateLimits(mx, limits_no_sys / BR_HH_GGTT, os.path.join(sys.argv[2], "Limits_xs_no_sys"))

plotSystematicComparison(mx, limits, limits_no_sys, os.path.join(sys.argv[2], "limits_systematic_comparison"))
ylabel = r"$\sigma(pp \rightarrow X) B(X \rightarrow HH \rightarrow \gamma\gamma\tau\tau)$ [fb]"
plotSystematicComparison2(mx, limits, limits_no_sys, ylabel, os.path.join(sys.argv[2], "limits_systematic_comparison2"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

def plotDiff(x, y1, y2, y1_label, y2_label, xlabel, ylabel, savepath):
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  axs[0].plot(x, y1, label=y1_label)
  axs[0].plot(x, y2, label=y2_label)

  axs[0].set_ylabel(ylabel)

  axs[0].legend()

  axs[1].plot(x, y1-y2)
  axs[1].set_xlabel(xlabel)
  axs[1].set_ylabel("%s - %s"%(y1_label, y2_label))

  f.savefig(savepath+".png")
  f.savefig(savepath+".pdf")

if __name__=="__main__":
  import pandas as pd
  import sys
  import os
  results = pd.read_csv(os.path.join(sys.argv[1], "param_test_results.csv"), index_col=0)
  print(results)

  mx = []
  for each in results.index:
    mx.append(int(each.split("_")[1]))

  mx=mx[1:]
  results.drop("MX_260_MY_125", inplace=True)

  plotDiff(mx, results["all"], results["only"], "All", "Only", r"$m_X$", "Signal Efficiency", os.path.join(sys.argv[1], "all_only"))
  plotDiff(mx, results["all"], results["skip"], "All", "Skip", r"$m_X$", "Signal Efficiency", os.path.join(sys.argv[1], "all_skip"))



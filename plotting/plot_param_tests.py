from ctypes import alignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (20,10)

def plotDiff(x, y1, y2, y1_label, y2_label, xlabel, ylabel, savepath):
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  axs[0].plot(x, y1, label=y1_label)
  axs[0].plot(x, y2, label=y2_label)

  axs[0].set_ylabel(ylabel)

  axs[0].legend()

  axs[1].plot(x, 100*(y1-y2))
  axs[1].set_xlabel(xlabel)
  axs[1].set_ylabel("%s - %s (%%)"%(y1_label, y2_label))

  f.savefig(savepath+".png")
  f.savefig(savepath+".pdf")

def plot2DDiff(x1_x2, y1, y2, y1_label, y2_label, xlabel, ylabel, savepath):
  x = np.arange(0, len(x1_x2))

  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  axs[0].set_ylabel(ylabel)
  axs[1].set_xlabel(xlabel)
  axs[1].set_xticks(x)
  axs[1].set_xticklabels([xi[1] for xi in x1_x2])
  axs[1].set_ylabel("%s - %s (%%)"%(y1_label, y2_label))

  for mx in np.unique(x1_x2[:,0]):
    idx = np.where(x1_x2[:,0]==mx)[0]
    axs[0].plot(x[idx], y1[idx], label=y1_label, color="tab:blue")
    axs[0].plot(x[idx], y2[idx], label=y2_label, color="tab:orange")
    axs[1].plot(x[idx], 100*(y1[idx]-y2[idx]), color="tab:blue")
    y1_label, y2_label = None, None

  axs[0].legend()

  axs[0].set_ylim(axs[0].get_ylim()) #force y lim to stay the same
  axs[1].set_ylim(axs[1].get_ylim())
  text_yet = False
  for i in range(1, len(x1_x2)):
    if not text_yet:
      text_y_loc = axs[0].get_ylim()[0] + 0.05*(axs[0].get_ylim()[1]-axs[0].get_ylim()[0])
      axs[0].text(i-1, text_y_loc, r"$m_X=%d$"%x1_x2[i][0], verticalalignment="bottom", horizontalalignment="left")
      text_yet = True
    if x1_x2[i][0] != x1_x2[i-1][0]:
      axs[0].plot([(2*i-1)/2, (2*i-1)/2], axs[0].get_ylim(), '--', color='0.8')
      axs[1].plot([(2*i-1)/2, (2*i-1)/2], axs[1].get_ylim(), '--', color='0.8')
      text_yet = False


  f.savefig(savepath+".png")
  f.savefig(savepath+".pdf")


if __name__=="__main__":
  import pandas as pd
  import sys
  import os
  import numpy as np
  results = pd.read_csv(os.path.join(sys.argv[1], "param_test_results.csv"), index_col=0)
  print(results)

  mx_my = []
  for each in results.index:
    s = each.split("_")
    mx_my.append([int(s[1]), int(s[3])])

  mx_my = np.array(mx_my)

  if (mx_my[:,1] == 125.0).all():
    mx = mx_my[:,0]

    mx=mx[1:]
    results.drop("MX_260_MY_125", inplace=True)

    plotDiff(mx, results["all"], results["only"], "All", "Only", r"$m_X$", "Signal Efficiency", os.path.join(sys.argv[1], "all_only"))
    plotDiff(mx, results["all"], results["skip"], "All", "Skip", r"$m_X$", "Signal Efficiency", os.path.join(sys.argv[1], "all_skip"))
  else:
    plot2DDiff(mx_my, results["all"], results["only"], "All", "Only", r"$m_Y$", "Signal Efficiency", os.path.join(sys.argv[1], "all_only"))
    plot2DDiff(mx_my, results["all"], results["skip"], "All", "Skip", r"$m_Y$", "Signal Efficiency", os.path.join(sys.argv[1], "all_skip"))



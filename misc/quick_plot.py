import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

def getFWHM(bins, hist):
  bc = (bins[:-1]+bins[1:])/2  

  lhist = hist[:np.argmax(hist)]
  lbc = bc[:np.argmax(hist)]
  rhist = hist[np.argmax(hist):]
  rbc = bc[np.argmax(hist):]

  l_half_idx = np.argmin(abs(lhist-max(hist)/2))
  r_half_idx = np.argmin(abs(rhist-max(hist)/2))

  return rbc[r_half_idx] - lbc[l_half_idx]


parser = argparse.ArgumentParser()
parser.add_argument('--parquet-input', '-i', type=str, required=True)
parser.add_argument('--figure-output', '-o', type=str, default="quick_plot.png")
parser.add_argument('--column', '-c', type=str, required=True)
parser.add_argument('--weight-column', '-w', type=str, default="weight")
parser.add_argument('--range', '-r', type=float, nargs=2, default=None)
parser.add_argument('--nbins', '-n', type=int, default=None)
parser.add_argument('--selection', '-s', type=str, nargs="+", default=None)
parser.add_argument('--log', action="store_true")

args = parser.parse_args()
df = pd.read_parquet(args.parquet_input)

if args.column not in df.columns:
  print("\n".join(df.columns))
  raise Exception(f"{args.column} does not exist in the dataframe")

if args.selection != None:
  selection = "&".join(["(df.%s)"%condition for condition in args.selection])
  df = df[eval(selection)]
else:
  selection = "All Events"

var = df[args.column]
s = (var>=args.range[0])&(var<=args.range[1])
var = var[s]
w = df[args.weight_column][s]

hist, bin_edges = np.histogram(var, bins=args.nbins, range=args.range, weights=w)

details = "Mean = %.2f\nStd = %.2f\nFWHM = %.2f"%(var.mean(), var.std(), getFWHM(bin_edges, hist))

plt.hist(bin_edges[:-1], bin_edges, weights=hist, label=details)
plt.legend()
plt.title(selection)
plt.xlabel(args.column)
if args.log:
  plt.yscale("log")
plt.savefig(args.figure_output)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import common

from plotting.plot_input_features import *
from scipy.optimize import curve_fit

from tqdm import tqdm

# class ExpFunc():
#   def __init__(self, N, norm, l, l_up, l_down, c):
#     self.l = l
#     self.l_up = l
#     self.l_down = l

#     self.N = N
#     self.N_up = N + np.sqrt(N)
#     self.N_down = N - np.sqrt(N)
    
#     self.c = c

#     self.norm = norm
  
#   def __call__(self, m):
#     return (self.N / self.norm(self.l, self.c)) * np.exp(-self.l*m + self.c)

#   def getNEventsInSR(self, sr):
#     nominal = intExp(sr[0], sr[1], self.l, self.N/self.norm(self.l, self.c), self.c)
#     return nominal    

# def intExp(a, b, l, N=1, c=0):
#   return (N/l) * (np.exp(-l*a+c) - np.exp(-l*b+c))

# class ExpFunc():
#   def __init__(self, N, norm, l, l_up, l_down):
#     self.l = l
#     self.l_up = l
#     self.l_down = l

#     self.N = N
#     self.N_up = N + np.sqrt(N)
#     self.N_down = N - np.sqrt(N)

#     self.norm = norm
  
#   def __call__(self, m):
#     return (self.N / self.norm(self.l)) * np.exp(-self.l*m)

#   def getNEventsInSR(self, sr):
#     nominal = intExp(sr[0], sr[1], self.l, self.N/self.norm(self.l))
#     return nominal    

# def intExp(a, b, l, N=1):
#   return (N/l) * (np.exp(-l*a) - np.exp(-l*b))

class ExpFunc():
  def __init__(self, N, norm, l, l_up, l_down):
    self.l = l
    self.l_up = l
    self.l_down = l

    self.N = N
    self.N_up = N + np.sqrt(N)
    self.N_down = N - np.sqrt(N)

    self.norm = norm
  
  def __call__(self, m):
    return (self.N / self.norm(self.l)) * np.power(m, -self.l)

  def getNEventsInSR(self, sr):
    nominal = intExp(sr[0], sr[1], self.l, self.N/self.norm(self.l))
    return nominal    

def intExp(a, b, l, N=1):
  return N/(l-1) * (np.power(a, -l+1) - np.power(b, -l+1))

def plotBkg(df,savepath,mgg_range=(100,180),sr=(120,130), dont_save=False):
  bkg = df[(df.mass>mgg_range[0])&(df.mass<mgg_range[1])]
  bkg_sr = bkg[(bkg.mass>sr[0])&(bkg.mass<sr[1])]
  bkg_sidebands = bkg[~((bkg.mass>sr[0])&(bkg.mass<sr[1]))]

  if (len(bkg_sr)==0) | (len(bkg_sidebands)==0):
    plt.hist(bkg.mass, bins=32, range=mgg_range)
    if not dont_save: plt.savefig(savepath)
    plt.clf()
    return 10.

  bin_width = (sr[1]-sr[0]) / 8
  assert (mgg_range[0]-sr[0])/bin_width == (mgg_range[0]-sr[0])//bin_width, print((mgg_range[0]-sr[0])/bin_width, (mgg_range[0]-sr[0])//bin_width)
  assert (mgg_range[1]-sr[1])/bin_width == (mgg_range[1]-sr[1])//bin_width, print((mgg_range[1]-sr[1])/bin_width, (mgg_range[1]-sr[1])//bin_width)
  nbins = int((mgg_range[1] - mgg_range[0]) / bin_width)

  hist, bin_edges = np.histogram(bkg_sidebands.mass, bins=nbins, range=mgg_range, weights=bkg_sidebands.weight)
  hist_w2, bin_edges = np.histogram(bkg_sidebands.mass, bins=nbins, range=mgg_range, weights=bkg_sidebands.weight**2)
  N, bin_edges = np.histogram(bkg_sidebands.mass, bins=nbins, range=mgg_range)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  
  s = (bin_centers>sr[0]) & (bin_centers<sr[1])
  bin_centers = bin_centers[~s]
  hist = hist[~s]
  hist_w2 = hist_w2[~s]
  N = N[~s]
  
  error = np.sqrt(hist_w2)
  error[N==0] = 1

  plt.errorbar(bin_centers, hist, error, fmt='o')

  norm = lambda l,: (intExp(mgg_range[0], sr[0], l) + intExp(sr[1], mgg_range[1], l))
  exp = lambda x, l: np.power(x, -l) / norm(l)
  popt, pcov = curve_fit(exp, bin_centers, hist/sum(hist), [3.], sigma=error/sum(hist), bounds=([0.], [10.0]))

  bkg_func = ExpFunc(sum(hist), norm, popt[0], popt[0], popt[0])
  m = np.linspace(mgg_range[0], mgg_range[1], 100)
  plt.plot(m, bin_width*bkg_func(m), label="l=%f"%popt[0])
  plt.legend()



  nbins = 8
  hist, bin_edges = np.histogram(bkg_sr.mass, bins=nbins, range=sr, weights=bkg_sr.weight)
  hist_w2, bin_edges = np.histogram(bkg_sr.mass, bins=nbins, range=sr, weights=bkg_sr.weight**2)
  N, bin_edges = np.histogram(bkg_sr.mass, bins=nbins, range=sr)
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  
  error = np.sqrt(hist_w2)
  error[N==0] = 1

  plt.errorbar(bin_centers, hist, error, fmt='o')

  n_sr_pred = bkg_func.getNEventsInSR(sr)
  n_sr_pred_error = n_sr_pred / np.sqrt(len(bkg_sidebands))
  n_sr_true = bkg_sr.weight.sum() 
  n_sr_true_error = n_sr_true / np.sqrt(len(bkg_sr))
  pull = (n_sr_true - n_sr_pred) / np.sqrt(n_sr_pred_error**2 + n_sr_true_error**2)

  plt.text(plt.xlim()[0], plt.ylim()[0],"pull = %f"%pull)

  if not dont_save: plt.savefig(savepath)
  plt.clf()

  return pull

def do(df, outdir):
  os.makedirs(outdir, exist_ok=True)

  pulls = []
  for i in tqdm(range(100)):
    df_boot = df.sample(frac=1.0, replace=True)
    mgg_range = (100,180)
    sr = (115,135)
    pulls.append(plotBkg(df_boot, os.path.join(outdir, "boot.png"), mgg_range=mgg_range, sr=sr, dont_save=True))
  plotBkg(df_boot, os.path.join(outdir, "boot.png"), mgg_range=mgg_range, sr=sr, dont_save=False)
  pulls=np.array(pulls)
  plt.hist(pulls, label="mean=%f\nwidth=%f"%(pulls.mean(), pulls.std()))
  plt.legend()
  plt.savefig(os.path.join(outdir, "boot_pulls.png"))
  plt.clf()

  pull_bias = pulls.mean()
  pull_sf = pulls.std()

  pulls_low = []
  pulls_high = []
  mxs = []
  mys = []
  for column in df.columns:
    if "intermediate" in column:
      sig_proc = column.split("score_")[1]
      mx, my = common.get_MX_MY(sig_proc)

      #mx = int(column.split("MX")[1].split("_")[1])
      #my = int(column.split("MY")[1].split("_")[1])
      
      if not args.Y_gg:
        mgg_range = (100,180)
        sr = (115,135)
      else:
        mgg_range = [my-25*(my/125.), my+55*(my/125.)]
        sr = [my-10*(my/125.), my+10*(my/125.)]
              
      print(mx, my)
      print(mgg_range, sr)

      df_pres = df[(df.mass>mgg_range[0])&(df.mass<mgg_range[1])]
      df_pres = df_pres.sort_values(column)
      if len(df_pres) < 400:
        print("Continuing")
        continue

      cdf = np.cumsum(df_pres.weight) / df_pres.weight.sum()
      #low_idx = np.argmin(abs(cdf-0.10))+1
      #high_idx = np.argmin(abs(cdf-0.90))-1
      
      low_idx = 200
      high_idx = len(df_pres)-200
      print(low_idx, high_idx, len(df_pres))
      
      # if (low_idx<100) or ((len(df_pres)-high_idx) < 100):
      #   print("Continuing")
      #   plt.hist(df_pres.iloc[high_idx:].mass, bins=32, range=mgg_range)
      #   plt.savefig(os.path.join(outdir, "mx_%d_my%d_high.png"%(mx,my)))
      #   plt.clf()
      #   continue      
      #assert (low_idx>100) and ((len(df_pres)-high_idx) > 100)

      #low_idx = int(len(df)/2 - 1)
      #high_idx = int(len(df)/2 + 1)

      df_low = df_pres.iloc[:low_idx]
      df_high = df_pres.iloc[high_idx:]

      mxs.append(mx)
      mys.append(my)
      pulls_low.append(plotBkg(df_low, os.path.join(outdir, "mx_%d_my%d_low.png"%(mx,my)), mgg_range=mgg_range, sr=sr))
      pulls_high.append(plotBkg(df_high, os.path.join(outdir, "mx_%d_my%d_high.png"%(mx,my)), mgg_range=mgg_range, sr=sr))

  #pulls_low=np.array(pulls_low-pull_bias) / pull_sf
  #pulls_high=np.array(pulls_high-pull_bias) / pull_sf
  pulls_low=np.array(pulls_low) / pull_sf
  pulls_high=np.array(pulls_high) / pull_sf

  if len(pulls_low) > 0:
    plt.hist(pulls_low, bins=8, range=(-4, 4), alpha=0.5, label="Bottom 200: mean=%.2f"%pulls_low.mean())
    plt.hist(pulls_high, bins=8, range=(-4, 4), alpha=0.5, label="Top 200: mean=%.2f"%pulls_high.mean())

    plt.xlabel("(MC yield - predicted yield) / uncertainty")
    plt.legend()
    plt.savefig(os.path.join(outdir, "pulls.png"))
    plt.clf()

    pulls_diff = pulls_high - pulls_low
    plt.hist(pulls_diff, bins=12, histtype='step', label="diff")
    plt.legend()
    plt.savefig(os.path.join(outdir, "pulls_diff.png"))

    lim = max([abs(min(pulls_diff)), abs(max(pulls_diff))])
    plt.hist2d(mxs, mys, bins=50, weights=pulls_diff, cmap="RdBu", vmin=-lim, vmax=+lim)
    plt.xlabel(r"$m_X$")
    plt.ylabel(r"$m_Y$")
    plt.colorbar()
    plt.savefig(os.path.join(outdir, "pulls_hist2d.png"))

def main(args):
  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  columns = list(filter(lambda x: "intermediate" in x, common.getColumns(args.parquet_input)))
  columns += ["process_id", "category", "Diphoton_mass", "weight"]
  df = pd.read_parquet(args.parquet_input, columns=columns)

  bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["Diphoton"]]
  df = df[df.process_id.isin(bkg_ids)]

  df.rename({"Diphoton_mass":"mass"}, axis=1, inplace=True)
  df = df.reindex(sorted(df.columns), axis=1)

  do(df, os.path.join(args.outdir, "all"))

  # for cat in df.category.unique():
  #   print("-"*50)
  #   print(common.category_map[cat])
  #   if sum(df.category==cat) > 0:
  #     do(df[df.category==cat], os.path.join(args.outdir, common.category_map[cat].replace("/", "_")))

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--Y-gg', action="store_true")

  args = parser.parse_args()

  df = main(args)
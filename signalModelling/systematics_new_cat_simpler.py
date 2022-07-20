import sys
from matplotlib.pyplot import polar
import pandas as pd
import numpy as np

import scipy.interpolate as spi
import scipy.optimize as spo

import argparse
import os
import json
import common

from signalModelling.interpolate_new_cat_simpler import tagSignals

def getEffSigma(mgg, w):
  assert sum(w) > 0
  w_normed = w / sum(w)

  l = mgg.quantile(0.05)
  h = mgg.quantile(0.95)

  hist, bin_edges = np.histogram(mgg, bins=1000, range=(l,h), weights=w_normed)
  
  min_width = h-l
  for i in range(len(hist)):
    sumw = 0
    for j in range(i, len(hist)):
      sumw += hist[j]
      if sumw >= 0.683:
        width = bin_edges[j]-bin_edges[i]
        if width < min_width:
          min_width = width
        break

  assert min_width != h-l
  
  return min_width / 2

def getMean(mgg, w):
  return np.average(mgg, weights=w)

def testEffSigma():
  mgg1 = np.random.normal(loc=125, scale=0.5, size=10000)
  mgg2 = np.random.normal(loc=125, scale=0.5, size=10000)
  mgg = np.concatenate([mgg1, mgg2])
  #w = np.random.random(size=len(mgg))
  w = np.ones_like(mgg)

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  plt.hist(mgg, bins=100, range=(120,130), weights=w)
  plt.savefig("eff_sigma_test.png")

  effSigma = getEffSigma(mgg, w)
  print(effSigma)

def rename_btag_systematics(df):
  btag_columns = filter(lambda x: ("btag" in x) and ("central" not in x), df.columns)
  mapper = {}
  for col in btag_columns:
    col_split = col.split("_")
    col_split[-1], col_split[-2] = col_split[-2], col_split[-1] #swap last two bits
    mapper[col] = "_".join(col_split)

  df.rename(mapper, axis=1, inplace=True)

  central_btag_col = list(filter(lambda x: ("btag" in x) and ("central" in x), df.columns))[0]

  btag_columns = filter(lambda x: ("btag" in x) and ("up" in x), df.columns)
  for col in btag_columns:
    df["%s_central"%"_".join(col.split("_")[:-1])] = df[central_btag_col]
  df.drop(central_btag_col, axis=1, inplace=True)

def deriveYieldSystematic(df, systematic, nominal_masses, masses):
  print(">> Doing yields")
  centrals = np.array([df.loc[(df.MX==m), "weight_%s_central"%systematic].sum() for m in nominal_masses])
  ups = np.array([df.loc[(df.MX==m), "weight_%s_up"%systematic].sum() for m in nominal_masses])
  downs = np.array([df.loc[(df.MX==m), "weight_%s_down"%systematic].sum() for m in nominal_masses])

  variations_up = 1 + abs(1 - ups/centrals)
  variations_down = 1 + abs(1 - downs/centrals)
  variations_mean = (variations_up + variations_down)/2

  variations_spline = spi.interp1d(nominal_masses, variations_mean, kind='linear')
  
  return [variations_spline(m) for m in masses]

def getYieldSystematicsNames(original_df):
  rename_btag_systematics(original_df)
  yield_systematics = []
  for col in original_df.columns:
    if "central" in col and ("weight_central" not in col):
      yield_systematics.append("_".join(col.split("_")[1:-1]))

  return yield_systematics

def dumpVariations(variations, masses, original_outdir, year, SR):
  outdir = os.path.join(original_outdir, str(year), str(SR))
  os.makedirs(outdir, exist_ok=True)

  models = {}
  for i, m in enumerate(masses):
    models[m] = {systematic:float(variations[systematic][i]) for systematic in variations.keys()}

  with open(os.path.join(outdir, "systematics.json"), "w") as f:
    json.dump(models, f, indent=4, sort_keys=True)

def deriveInterpolationSystematic(df, nominal_masses, masses):
  print(">> Doing interpolation systematic")
  #norms for all nominal masses
  norms_nominal = [df.loc[(df.MX==m), "weight"].sum() for m in nominal_masses]
  print(norms_nominal)
  #create a spline for each nominal mass (except edges) where you remove the nominal mass from spline
  nom_m = list(nominal_masses)
  splines = [spi.interp1d(nom_m[:i]+nom_m[i+1:], norms_nominal[:i]+norms_nominal[i+1:], kind='cubic') for i in range(1, len(nominal_masses)-1)]
  
  norms_nominal = np.array(norms_nominal) 
  norms_interpolated = np.array([splines[i](m) for i, m in enumerate(nominal_masses[1:-1])])

  variations1 = 1 + abs(norms_nominal[1:-1]-norms_interpolated)/norms_nominal[1:-1]
  variations2 = 1 + abs(norms_nominal[1:-1]-norms_interpolated)/norms_interpolated
  variations = np.min(np.array([variations1, variations2]), axis=0)
  variations = np.concatenate([[variations[0]], variations, [variations[-1]]]) #give edge masses same systematic as closest one in

  variations_spline = spi.interp1d(nominal_masses, variations, kind='linear')

  variations = [variations_spline(m) for m in masses]
  for i, m in enumerate(masses):
    if m in nominal_masses:
      variations[i] = 1.0

  return variations

def deriveParquetYieldSystematic(dfs, systematic, nominal_masses, masses, year, SR):
  print(">> Doing parquet yields")
  df_up = dfs["%s_up"%systematic][(dfs["%s_up"%systematic].year==year) & (dfs["%s_up"%systematic].SR==SR)]
  df_down = dfs["%s_down"%systematic][(dfs["%s_down"%systematic].year==year) & (dfs["%s_down"%systematic].SR==SR)]

  yields = lambda df: np.array([df.loc[(df.MX==m), "weight"].sum() for m in nominal_masses])

  ups = yields(df_up)
  downs = yields(df_down)

  variations_mean = 1 + abs(ups - downs) / (ups + downs)

  variations_spline = spi.interp1d(nominal_masses, variations_mean, kind='linear')
  
  return [variations_spline(m) for m in masses]

def getParquetShapeVariations(f, df_up, df_down, nominal_masses, masses):
  up = f(df_up)
  down = f(df_down)

  variations_mean = abs(up - down) / (up + down)

  variations_spline = spi.interp1d(nominal_masses, variations_mean, kind='linear')

  return [variations_spline(m) for m in masses]

def deriveParquetShapeSystematics(dfs, systematic, nominal_masses, masses, year, SR):
  print(">> Doing parquet shape yields")
  df_up = dfs["%s_up"%systematic][(dfs["%s_up"%systematic].year==year) & (dfs["%s_up"%systematic].SR==SR)]
  df_down = dfs["%s_down"%systematic][(dfs["%s_down"%systematic].year==year) & (dfs["%s_down"%systematic].SR==SR)]

  yields = lambda df: np.array([df.loc[(df.MX==m), "weight"].sum() for m in nominal_masses])
  sigmas = lambda df: np.array([getEffSigma(df[df.MX==m].Diphoton_mass, df[df.MX==m].weight) for m in nominal_masses])
  means = lambda df: np.array([getMean(df[df.MX==m].Diphoton_mass, df[df.MX==m].weight) for m in nominal_masses])

  consts = {}
  consts["const_rate_%s"%systematic] = getParquetShapeVariations(yields, df_up, df_down, nominal_masses, masses)
  consts["const_sigma_%s"%systematic] = getParquetShapeVariations(sigmas, df_up, df_down, nominal_masses, masses)
  consts["const_mean_%s"%systematic] = getParquetShapeVariations(means, df_up, df_down, nominal_masses, masses)

  return consts

def deriveSystematics(dfs, nominal_masses, masses, original_outdir):
  original_df = dfs["nominal"]

  yield_systematics = getYieldSystematicsNames(original_df)
  parquet_yield_systematics = ["JER", "JES", "MET_JER", "MET_JES", "MET_Unclustered", "Muon_pt", "Tau_pt"]
  parquet_shape_systematics = ["fnuf", "material", "scale", "smear"]

  for year in np.unique(original_df.year):
    for SR in np.sort(np.unique(original_df.SR)):
      print(year, SR)
      df = original_df[(original_df.year==year)&(original_df.SR==SR)]

      variations = {}

      #for systematic in yield_systematics:
      #  variations[systematic] = deriveYieldSystematic(df, systematic, nominal_masses, masses)
      #for systematic in parquet_yield_systematics:
      #  variations[systematic] = deriveParquetYieldSystematic(dfs, systematic, nominal_masses, masses, year, SR)
      #for systematic in parquet_shape_systematics:
      #  variations.update(deriveParquetShapeSystematics(dfs, systematic, nominal_masses, masses, year, SR))

      variations["interpolation"] = deriveInterpolationSystematic(df, nominal_masses, masses)

      dumpVariations(variations, masses, original_outdir, year, SR)

def addMigrationSystematics(outdir):
  years = sorted(list(filter(lambda x: os.path.isdir(os.path.join(outdir, x)), os.listdir(outdir))))
  SRs = sorted(list(filter(lambda x: os.path.isdir(os.path.join(outdir, years[0], x)), os.listdir(os.path.join(outdir, years[0])))))

  for year in years:
    sig_models = {}
    systematics = {}
    for SR in SRs:
      with open(os.path.join(outdir, year, SR, "model.json"), "r") as f:
        sig_models[SR] = json.load(f)
      with open(os.path.join(outdir, year, SR, "systematics.json"), "r") as f:
        systematics[SR] = json.load(f)

    for SR in SRs[:-1]:
      masses = sig_models[SR].keys()
      SRp1 = str(int(SR)+1)
      for m in masses:
        yield1 = sig_models[SR][m][0]
        yield2 = sig_models[SRp1][m][0]

        interpolation_uncert1 = systematics[SR][m]["interpolation"]
        interpolation_uncert2 = systematics[SRp1][m]["interpolation"]

        sys_name = "Interpolation_migration_%s_%s"%(SR, SRp1)

        up1 = 1 + ((interpolation_uncert2-1)*yield2) / yield1
        down1 = 1 / interpolation_uncert1

        up2 = 1 + ((interpolation_uncert1-1)*yield1) / yield2
        down2 = 1 / interpolation_uncert2

        for j in SRs:
          if j == SR:
            systematics[j][m][sys_name+"_left"] = up1
            systematics[j][m][sys_name+"_right"] = down1
          elif j == SRp1:
            systematics[j][m][sys_name+"_left"] = down2
            systematics[j][m][sys_name+"_right"] = up2
          else:
            systematics[j][m][sys_name+"_left"] = 1.0
            systematics[j][m][sys_name+"_right"] = 1.0

    for SR in SRs:
      with open(os.path.join(outdir, year, SR, "systematics.json"), "w") as f:
        json.dump(systematics[SR], f, indent=4)

def loadDataFrame(path, proc_dict, optim_dir, columns=None, batch_size=None):
  if batch_size is None:
    df = pd.read_parquet(path, columns=columns)
  else:
    from pyarrow.parquet import ParquetFile
    import pyarrow as pa
    pf = ParquetFile(path) 
    iter = pf.iter_batches(batch_size=batch_size, columns=columns)
    first_rows = next(iter) 
    df = pa.Table.from_batches([first_rows]).to_pandas()

  df = df[df.y==1]
  common.add_MX_MY(df, proc_dict)
  tagSignals(df, optim_dir, proc_dict)
  return df

def loadDataFrames(args):
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']
  
  dfs = {}
  
  #load nominal dataframe
  df = loadDataFrame(os.path.join(args.parquet_input, "merged_nominal.parquet"), proc_dict, args.optim_dir, batch_size=None)
  systematic_columns = list(filter(lambda x: ("intermediate_transformed_score" in x), df.columns)) + ["Diphoton_mass", "process_id", "weight", "y", "year"]
  dfs["nominal"] = df

  # for path in os.listdir(args.parquet_input):
  #   if (".parquet" in path) and ("nominal" not in path):
  #     print(path)
  #     df = loadDataFrame(os.path.join(args.parquet_input, path), proc_dict, optim_dir, columns=systematic_columns, batch_size=None)
  #     name = "_".join(path.split(".parquet")[0].split("_")[1:])
  #     dfs[name] = df
  return dfs

def main(args):
  dfs = loadDataFrames(args)

  nominal_masses = np.sort(np.unique(dfs["nominal"].MX))
  masses = np.arange(nominal_masses[0], nominal_masses[-1]+args.step, args.step)
  
  deriveSystematics(dfs, nominal_masses, masses, args.outdir)
  addMigrationSystematics(args.outdir)
  return dfs

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-dir', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--step', type=float, default=10.0)
  args = parser.parse_args()

  os.makedirs(args.outdir, exist_ok=True)

  dfs = main(args)
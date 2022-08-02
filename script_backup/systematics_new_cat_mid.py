import pandas as pd
import numpy as np

import scipy.interpolate as spi

import argparse
import os
import json
import common
import sys

import tracemalloc

from numba import jit

@jit(nopython=True)
def go_fast(hist, bin_edges, l, h):
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
    if sumw < 0.683: #no more 68% windows
      break
  return min_width

def getEffSigma(mgg, w):
  assert len(w) >= 100
  assert sum(w) > 0

  s = (mgg >= 115) & (mgg <= 135)
  w_normed = w[s] / sum(w[s])

  #l = mgg.quantile(0.05)
  #h = mgg.quantile(0.95)
  l = 115
  h = 135

  hist, bin_edges = np.histogram(mgg[s], bins=2000, range=(l,h), weights=w_normed)
  
  min_width = go_fast(hist, bin_edges, l, h)
  assert min_width != h-l
  
  return min_width / 2

def getMean(mgg, w):
  s = (mgg >= 115) & (mgg <= 135)
  #return np.median(mgg[s])
  return np.average(mgg[s], weights=w[s])

# def getMean(mgg, w):
#   assert len(w) >= 100
#   assert sum(w) > 0
#   w_normed = w / sum(w)

#   #l = mgg.quantile(0.05)
#   #h = mgg.quantile(0.95)
#   l = 115
#   h = 135

#   hist, bin_edges = np.histogram(mgg, bins=2000, range=(l,h), weights=w_normed)

#   idx = np.argmax(hist)
#   mean = (bin_edges[idx] + bin_edges[idx+1])/2

#   return mean

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

  #variations_mean = abs(up - down) / (up + down)
  variations_mean = (up - down) / (up + down)

  variations_spline = spi.interp1d(nominal_masses, variations_mean, kind='linear')

  return [variations_spline(m) for m in masses]

def deriveParquetShapeSystematics(dfs, systematic, nominal_masses, masses, year, SR):
  print(">> Doing parquet shape yields")
  df_up = dfs["%s_up"%systematic][(dfs["%s_up"%systematic].year==year) & (dfs["%s_up"%systematic].SR==SR)]
  df_down = dfs["%s_down"%systematic][(dfs["%s_down"%systematic].year==year) & (dfs["%s_down"%systematic].SR==SR)]

  yields = lambda df: np.array([df.loc[(df.MX==m), "weight"].sum() if sum(df.MX==m) >= 100 else 1.0 for m in nominal_masses])
  sigmas = lambda df: np.array([getEffSigma(df[df.MX==m].Diphoton_mass, df[df.MX==m].weight) if sum(df.MX==m) >= 100 else 1.0 for m in nominal_masses])
  means = lambda df: np.array([getMean(df[df.MX==m].Diphoton_mass, df[df.MX==m].weight) if sum(df.MX==m) >= 100 else 1.0 for m in nominal_masses])

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
  #parquet_yield_systematics = []
  #parquet_shape_systematics = ["fnuf"]  

  systematics = {}
  with open(os.path.join(original_outdir, "model.json"), "r") as f:
    sig_model = json.load(f)

  for year in sig_model.keys():
    systematics[year] = {}
    for SR in sig_model[year].keys():
      print(year, SR)
      systematics[year][SR] = {}

      df = original_df[(original_df.year==int(year))&(original_df.SR==int(SR))]

      variations = {}

      for systematic in yield_systematics:
        variations[systematic] = deriveYieldSystematic(df, systematic, nominal_masses, masses)
      for systematic in parquet_yield_systematics:
        variations[systematic] = deriveParquetYieldSystematic(dfs, systematic, nominal_masses, masses, int(year), int(SR))
      for systematic in parquet_shape_systematics:
        variations.update(deriveParquetShapeSystematics(dfs, systematic, nominal_masses, masses, int(year), int(SR)))

      variations["interpolation"] = [sig_model[year][SR][str(m)]["this mass"]["norm_systematic"] if m not in nominal_masses else 1.0 for m in masses]

      models = {}
      for i, m in enumerate(masses):
        models[str(m)] = {sys_name:float(variations[sys_name][i]) for sys_name in variations.keys()}
      systematics[year][SR] = models

  with open(os.path.join(original_outdir, "systematics.json"), "w") as f:
    json.dump(systematics, f, indent=4)

def addMigrationSystematics(outdir):
  with open(os.path.join(outdir, "model.json"), "r") as f:
    sig_model = json.load(f)
  with open(os.path.join(outdir, "systematics.json"), "r") as f:
    systematics = json.load(f)

  years = sorted(list(sig_model.keys()))
  SRs = sorted(list(sig_model[years[0]].keys()))
  masses = sorted(list(sig_model[years[0]][SRs[0]].keys()))

  print(years)
  print(SRs)

  for year in years:
    for SR in SRs[:-1]:
      masses = sig_model[year][SR].keys()
      SRp1 = str(int(SR)+1)
      for m in masses:
        yield1 = sig_model[year][SR][m]["this mass"]["norm"]
        yield2 = sig_model[year][SRp1][m]["this mass"]["norm"]

        if (yield1 == 0) or (yield2 == 0):
          up1 = up2 = down1 = down2 = 1.0
        else:
          interpolation_uncert1 = systematics[year][SR][m]["interpolation"]
          interpolation_uncert2 = systematics[year][SRp1][m]["interpolation"]

          if interpolation_uncert1 >= 1.5 or interpolation_uncert2 >= 1.5:
            print(year, SR, m, interpolation_uncert1, interpolation_uncert2)

          sys_name = "Interpolation_migration_%s_%s"%(SR, SRp1)

          up1 = 1 + ((interpolation_uncert2-1)*yield2) / yield1
          down1 = 1 - (interpolation_uncert1 - 1)
          assert down1 > 0

          up2 = 1 + ((interpolation_uncert1-1)*yield1) / yield2
          down2 = 1 - (interpolation_uncert2 - 1)
          assert down2 > 0

        for j in SRs:
          if j == SR:
            systematics[year][j][m][sys_name+"_left"] = up1
            systematics[year][j][m][sys_name+"_right"] = down1
          elif j == SRp1:
            systematics[year][j][m][sys_name+"_left"] = down2
            systematics[year][j][m][sys_name+"_right"] = up2
          else:
            systematics[year][j][m][sys_name+"_left"] = 1.0
            systematics[year][j][m][sys_name+"_right"] = 1.0

  with open(os.path.join(outdir, "systematics.json"), "w") as f:
    systematics = json.dump(systematics, f, indent=4)

def tagSignals(df, optim_dir, proc_dict):
  df["SR"] = -1

  with open(os.path.join(optim_dir, "optim_results.json"), "r") as f:
    optim_results = json.load(f)

  for proc in proc_dict.keys():
    if proc_dict[proc] in np.unique(df.process_id):
      score_name = "intermediate_transformed_score_%s"%proc

      for entry in optim_results:
        if score_name == entry["score"]:
          boundaries = entry["category_boundaries"][::-1]
          break

      for i in range(len(boundaries)-1):
        selection = (df[score_name] <= boundaries[i]) & (df[score_name] > boundaries[i+1]) & (df.process_id == proc_dict[proc])
        df.loc[selection, "SR"] = i

  return df[df.SR!=-1]

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

  columns_to_keep = list(filter(lambda x: x[:5]!="score", df.columns)) #remove raw score
  df = df[df.y==1]
  df = df[columns_to_keep]
  
  common.add_MX_MY(df, proc_dict)
  tagSignals(df, optim_dir, proc_dict)
  return df

def loadDataFrames(args):
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']
  
  dfs = {}
  
  batch_size = None

  #load nominal dataframe
  df = loadDataFrame(os.path.join(args.parquet_input, "merged_nominal.parquet"), proc_dict, args.optim_dir, batch_size=batch_size)
  systematic_columns = list(filter(lambda x: ("intermediate_transformed_score" in x), df.columns)) + ["Diphoton_mass", "process_id", "weight", "y", "year"]
  dfs["nominal"] = df

  tracemalloc.start()

  for path in os.listdir(args.parquet_input):
    if (".parquet" in path) and ("nominal" not in path):
    #if "fnuf" in path:
      print(path)
      print(np.array(tracemalloc.get_traced_memory())/(1024*1024*1024))
      df = loadDataFrame(os.path.join(args.parquet_input, path), proc_dict, args.optim_dir, columns=systematic_columns, batch_size=batch_size)
      name = "_".join(path.split(".parquet")[0].split("_")[1:])
      dfs[name] = df

  tracemalloc.stop()
  print(dfs)
  return dfs

def main(args):
  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=True)
    return True

  dfs = loadDataFrames(args)

  nominal_masses = np.sort(np.unique(dfs["nominal"].MX))
  masses = np.arange(nominal_masses[0], nominal_masses[-1]+args.step, args.step, dtype=int)
  
  deriveSystematics(dfs, nominal_masses, masses, args.outdir)
  addMigrationSystematics(args.outdir)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-dir', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--step', type=float, default=10.0)
  parser.add_argument('--batch', action="store_true")
  args = parser.parse_args()

  os.makedirs(args.outdir, exist_ok=True)

  main(args)
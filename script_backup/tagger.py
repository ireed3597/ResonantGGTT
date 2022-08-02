import pandas as pd
import numpy as np
import json 
import argparse
import os

def loadDataFrame(args):
  df = pd.read_parquet(args.parquet_input)
  with open(args.optim_results) as f:
    optim_results = json.load(f)

  return df, optim_results

def assignSignalRegions(df, optim_results):
  df["SR"] = -1
  
  boundaries = optim_results["category_boundaries"][::-1] #so that cat0 is most pure
  for i in range(len(boundaries)-1):
    selection = (df[optim_results["score"]] <= boundaries[i]) & (df[optim_results["score"]] > boundaries[i+1])
    #selection = (df.score <= boundaries[i]) & (df.score > boundaries[i+1])
    df.loc[selection, "SR"] = i
  return df[df.SR!=-1]

def main(args):
  df, optim_results = loadDataFrame(args)

  df = assignSignalRegions(df, optim_results)
  df.to_parquet(os.path.join(args.outdir, "output.parquet"))

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str)
  #parser.add_argument('--sig-proc', '-p', type=str, required=True)
  args = parser.parse_args()

  if args.outdir == None:
    args.outdir = os.path.join("tagging_output")
  
  os.makedirs(args.outdir, exist_ok=True)

  main(args)

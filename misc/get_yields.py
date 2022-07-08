import pandas as pd
import sys
import json

def assignSignalRegions(df, optim_results, score_name):
  df["SR"] = -1
  
  boundaries = optim_results["category_boundaries"][::-1] #so that cat0 is most pure
  for i in range(len(boundaries)-1):
    selection = (df[score_name] <= boundaries[i]) & (df[score_name] > boundaries[i+1])
    df.loc[selection, "SR"] = i
  return df[df.SR!=-1]

df = pd.read_parquet(sys.argv[1])
with open(sys.argv[1], "r") as f:
  optim_results = json.load(sys.argv[2])

mx = []
#data = []
sig = []
bkg = []


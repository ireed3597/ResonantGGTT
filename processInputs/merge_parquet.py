import pandas as pd
import sys
from tqdm import tqdm

dfs = []
for f in tqdm(sys.argv[2:]):
  try:
    dfs.append(pd.read_parquet(f))
  except:
    print("Could not read %s"%f)
df = pd.concat(dfs)

#df = pd.concat([pd.read_parquet(f) for f in sys.argv[2:]])
df.to_parquet(sys.argv[1])

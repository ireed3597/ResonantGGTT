import json
import os
import argparse
import pandas as pd
import tabulate
import common
import numpy as np
import warnings

import sys

mxs = [int(path.split("M")[1]) for path in os.listdir(sys.argv[1]) if "XToHH" in path]

test = []
train = []

for mx in mxs:
  with open(os.path.join(sys.argv[1], "XToHHggTauTau_M%d"%mx, "ROC_skimmed.json"), "r") as f:
    roc = json.load(f)
    test.append(roc["test_auc"])
    train.append(roc["train_auc"])

df = pd.DataFrame({"mx":mxs, "train_auc":train, "test_auc":test})
df.sort_values("mx", inplace=True)
df.set_index("mx", inplace=True)
print(df.to_latex(float_format="%.5f"))
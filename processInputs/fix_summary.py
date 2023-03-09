"""
Edit the summary file so the GJets processes are treated as one
"""

import json
import sys

with open(sys.argv[1], "r") as f:
  summary = json.load(f)
  
gjet_id = None
to_drop_keys = []
for key, item in summary["sample_id_map"].items():
  if "GJets" in key:
    to_drop_keys.append(key)
    if gjet_id is not None:
      assert item == gjet_id
    else:
      gjet_id = item

for key in to_drop_keys:
  del summary["sample_id_map"][key]

summary["sample_id_map"]["GJets"] = gjet_id

with open(sys.argv[2], "w") as f:
  json.dump(summary, f, indent=2)

import argparse
import os
import common

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-dir', '-i', type=str, required=True)
  parser.add_argument('--output-dir', '-o', type=str, required=True)
  parser.add_argument('--folders', type=str, nargs="+", required=True)
  parser.add_argument('--keep-features', '-f', type=str, default=None)
  parser.add_argument('--sig-procs', '-p', type=str, nargs="+", default=None)

  args = parser.parse_args()

  for folder in args.folders:
    os.makedirs(os.path.join(args.output_dir, folder), exist_ok=True)
    files = os.listdir(os.path.join(args.input_dir, folder))
    for f in files:
      if "merged_nominal" not in f: continue
      options = "-i %s/%s/%s -o %s/%s/%s -s %s/summary.json -f %s --batch"%(args.input_dir,folder,f,  args.output_dir,folder,f,  args.input_dir,  args.keep_features)
      if args.sig_procs is not None: options += " -p %s"%" ".join(args.sig_procs)
      common.submitToBatch(["processInputs/process_HiggsDNA_Inputs.py"] + options.split(" "), extra_memory=8)

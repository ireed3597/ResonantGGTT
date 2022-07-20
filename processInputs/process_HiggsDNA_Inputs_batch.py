import argparse
import os
import common

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-dir', '-i', type=str, required=True)
  parser.add_argument('--output-dir', '-o', type=str, required=True)
  parser.add_argument('--folders', type=str, nargs="+", required=True)
  parser.add_argument('--keep-features', '-f', type=str, default=None)

  args = parser.parse_args()

  for folder in args.folders:
    os.makedirs(os.path.join(args.output_dir, folder), exist_ok=True)
    files = os.listdir(os.path.join(args.input_dir, folder))
    for f in files:
      options = "-i %s/%s/%s -o %s/%s/%s -s %s/summary.json -f %s --batch"%(args.input_dir,folder,f,  args.output_dir,folder,f,  args.input_dir,  args.keep_features)
      common.submitToBatch(["processInputs/process_HiggsDNA_Inputs.py"] + options.split(" "))
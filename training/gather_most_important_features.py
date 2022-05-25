import json
import os
import argparse

def main(args):
  sig_procs = filter(lambda x: os.path.isdir(os.path.join(args.input_dir, x)), os.listdir(args.input_dir))

  most_important_features = []
  for sig_proc in sig_procs:
    with open(os.path.join(args.input_dir, sig_proc, "feature_importances.json"), "r") as f:
      feature_importances = json.load(f)[1]
    most_important_features += feature_importances[:args.top_n_features]
  most_important_features = list(set(most_important_features))

  print(">> Found %d features"%len(most_important_features))
  print("\n".join(most_important_features))

  with open(args.output_json, "w") as f:
    json.dump(most_important_features, f) 

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-dir', '-i', type=str, required=True)
  parser.add_argument('--top-n-features', '-n', type=int, default=20)
  parser.add_argument('--output-json', '-o', default=None)

  args = parser.parse_args()

  if args.output_json == None:
    args.output_json = os.path.join(args.input_dir, "most_important_features.json")

  main(args)
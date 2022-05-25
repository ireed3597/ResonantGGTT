#!/usr/bin/env bash

cd /home/hep/mdk16/PhD/ggtt/ResonantGGTT
source env/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/hep/mdk16/PhD/ggtt/ResonantGGTT"

python processInputs/process_HiggsDNA_Inputs.py -i /vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/May22/merged_nominal.parquet -o test.parquet -s /vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/May22/summary.json
#!/bin/sh

cd /home/hep/mdk16/PhD/ggtt/ResonantGGTT
source setup.sh
file=$1
python processInputs/merge_parquet.py Inputs_NMSSM_Y_gg_reco_MX_mgg/$file Inputs_NMSSM_Y_gg_reco_MX_mgg/*/$file
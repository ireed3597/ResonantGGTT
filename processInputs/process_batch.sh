#!/bin/sh

cd /home/hep/mdk16/PhD/ggtt/ResonantGGTT
source setup.sh
file=$1

pushd /vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/Jun22
  folders=$(echo NMSSM_XYH_Y_gg_H_tautau_MX_*)
  folders=($folders)
  folder=${folders[$((SGE_TASK_ID-1))]}
popd

mkdir -p /vols/cms/mdk16/ggtt/Inputs/HiggsDNA/post/Jun22/NMSSM_Y_gg_reco_MX_mgg/${folder}
python processInputs/process_HiggsDNA_Inputs.py -i /vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/Jun22/${folder}/${file} -o /vols/cms/mdk16/ggtt/Inputs/HiggsDNA/post/Jun22/NMSSM_Y_gg_reco_MX_mgg/${folder}/${file} -s /vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/Jun22/summary.json --keep-features important_17_corr

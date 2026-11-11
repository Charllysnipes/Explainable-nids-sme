#!/bin/bash
# Downloads the three public datasets into ./data in the layout data_prep.py expects.
# Run from the repository root:  bash scripts/get_data.sh
set -e
mkdir -p data/probe data/unsw && cd data
[ -d NSL_KDD ] || git clone --depth 1 https://github.com/defcom17/NSL_KDD NSL_KDD
cd probe
[ -d Intrusion-Detection-System-Using-Machine-Learning ] || \
  git clone --depth 1 https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning
if [ ! -d datasets ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/comsyssec/datasets
  (cd datasets && git sparse-checkout set unsw-nb15)
fi
cd ..
if [ ! -f unsw/UNSW_NB15_training-set.csv ]; then
  [ -d unsw_repo ] || git clone --depth 1 https://github.com/InitRoot/UNSW_NB15 unsw_repo
  python3 -c "import zipfile; zipfile.ZipFile('unsw_repo/UNSW_NB15.zip').extractall('unsw_tmp')"
  find unsw_tmp -name 'UNSW_NB15_t*-set.csv' -exec cp {} unsw/ \;
fi
echo "Datasets ready in $(pwd)"

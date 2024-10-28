#!/bin/bash
#SBATCH --job-name=gen_test
#SBATCH --output=logs/gen_test.out 
#SBATCH --partition=ml-all-gpu 
#SBATCH --ntasks=1
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=13G
#SBATCH --gres=gpu:0
#SBATCH --nice=1000
# module load anaconda
# module load cuda/10.2

# DATASETS=('CiteSeer' 'Cora' 'Photo' 'dblp' 'usa' 'brazil')
# DATASETS=('Cora')
# for dataset in "${DATASETS[@]}"
# do
#     python pipeline_gen_coherence_test.py \
#     --base-folder ./data \
#     --config-file ./config/"$dataset"_coherence_test.json \
#     --task exp 
# done

# DATASETS=('CiteSeer' 'Cora' 'imdb' 'dblp' 'usa' 'brazil')
DATASETS=('Cora')
for dataset in "${DATASETS[@]}"
do
    python pipeline_gen_coherence_test.py \
    --base-folder ./data \
    --config-file ./config/"$dataset"_coherence_test.json \
    --task nc 
done


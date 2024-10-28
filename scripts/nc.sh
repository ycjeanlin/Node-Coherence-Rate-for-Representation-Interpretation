#!/bin/bash


# exp
DATASETS=('Cora')
MODELS=('gcn')
for dataset in "${DATASETS[@]}"
do
    for model in "${MODELS[@]}"
    do
        python pipeline_"$model".py \
            --base-folder ./data \
            --model-config ./config/"$dataset"_baseline_"$model".json \
            --test-config ./config/"$dataset"_coherence_test.json \
            --with-feat \
            --task nc

        python pipeline_nc.py \
            --base-folder ./data \
            --model-config ./config/"$dataset"_baseline_"$model".json \
            --test-config ./config/"$dataset"_coherence_test.json
    done
done




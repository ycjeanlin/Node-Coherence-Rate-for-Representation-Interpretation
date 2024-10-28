#!/bin/bash
#
DATASETS=('Cora')
MODELS=('gcn')
for dataset in "${DATASETS[@]}"
do
    for model in "${MODELS[@]}"
    do
        python pipeline_"$model".py \
            --base-folder ./data \
            --model-suffix baseline \
            --model-config ./config/"$dataset"_baseline_"$model".json \
            --test-config ./config/"$dataset"_coherence_test.json \
            --with-feat \
            --task link

        python pipeline_link.py \
            --base-folder ./data \
            --model-suffix baseline \
            --model-config ./config/"$dataset"_baseline_"$model".json \
            --test-config ./config/"$dataset"_coherence_test.json
    done
done



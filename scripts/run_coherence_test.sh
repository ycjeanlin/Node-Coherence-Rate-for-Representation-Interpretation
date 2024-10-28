#!/bin/bash

DATASETS=('Cora')
MODELS=('random', 'gcn')
for dataset in "${DATASETS[@]}"
do
    for model in "${MODELS[@]}"
    do
        python pipeline_run_coherence_test.py \
            --base-folder ./data \
            --test-config-file ./config/"$dataset"_coherence_test.json \
            --model-config-file ./config/"$dataset"_baseline_"$model".json \
            --model-suffix baseline \
            --task nc 
    done
done

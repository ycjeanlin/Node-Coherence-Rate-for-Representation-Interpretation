#!/bin/bash

DATASETS=('Cora')
for dataset in "${DATASETS[@]}"
do
    python pipeline_svd.py \
        --base-folder ./data \
        --embed-dim  512 \
        --config-file ./config/"$dataset"_coherence_test.json


    RELATIONS=('label_dist' 'pagerank' 'g_dist' 'degree_dist' 'has_link' 'attr' 'attr_dist')
    for suffix in "${RELATIONS[@]}"
    do
        python pipeline_run_coherence_test.py \
            --base-folder ./data \
            --test-config-file ./config/"$dataset"_coherence_test.json \
            --model-config-file ./config/"$dataset"_baseline_svd.json \
            --model-suffix "$suffix"_512 \
            --task exp 
    done

done


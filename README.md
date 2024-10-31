# Introduction 
This repository is based on the paper "Rethinking Node Representation Interpretation through Relation Coherence." The paper can be found in this [link](https://purdue0-my.sharepoint.com/:b:/g/personal/lin915_purdue_edu/EeYQpgEhE11Dno8ht6fX0gMBhSsFL-OjY5Am63wy-0rR4Q?e=4i7AGM).


# Node Representation Interpretation
1. Install all the required packages listed in the `requirements.txt`
2. Generate coherence test data: `bash ./scripts/gen_coherence_test.sh`
3. Generate node representations for models: `bash ./scripts/nc.sh`
4. Run Node Coherence Rate for Representation Interpretation (NCI): `bash ./scripts/run_coherence_test.sh`
5. Review the interpretation results by using the jupyter notebook `./notebooks/interpretation_result.ipynb`


# Interpretation Method Evaluation Process
1. Install all the required packages listed in the `requirements.txt`
2. Generate coherence test data: `bash ./scripts/gen_coherence_test.sh`
3. Run the IME process: `bash ./scripts/run_ime_process.sh`
4. Review the evalaution results by using the jupyter notebook: `./notebooks/IME_result.ipynb` 


# Note
* You only need to run `bash ./scripts/gen_coherence_test.sh` once for each graph and each downstream task.


# prefixReward
do prefix tuning on bert-based reward for downstream gemma-7b tuning

## How to use
```shell
python prefixReward_v11_train.py --cfg-path projects/prefixReward_v11/train.yaml
```
## requirements
```shell
conda create --name prefixReward python=3.8
conda activate prefixReward

pip install transformers==4.40.0 torch==2.4.1 pandas==2.0.3 openpyxl==3.1.5
```

## Model info
### prefixReward v11
#### data
The following data format input is required
```
df cols: {
    `sample_idx`(int), `data_idx`(int), `data_idx`(int), 
    `assistant1`(str), `user1`(str), 
    `assistant2`(str), `user2`(str), 
    `assistant3`(str), `user3`(str), 
    `original_answer`(str), `gemma-7b_response`(str), 
    `Fit the context`(int), `Localization`(int), `Concision`(int), `Truthfulness`(int), `Harmfulness`(int), `Overall satisfaction`(int), `ranking(int)`
    }
```

#### inference
<div align="center">
  <img src="https://github.com/yasaisen/prefixReward/blob/main/doc/prefixReward_v11/prefixReward_v11_inference.png" alt="inference" width="300">
</div>

#### model structure
<div align="center">
  <img src="https://github.com/yasaisen/prefixReward/blob/main/doc/prefixReward_v11/prefixReward_v11_model.png" alt="model structure" width="400">
</div>

#### training
<div align="center">
  <img src="https://github.com/yasaisen/prefixReward/blob/main/doc/prefixReward_v11/prefixReward_v11_training.png" alt="training" width="600">
</div>

#### training
<div align="center">
  <img src="https://github.com/yasaisen/prefixReward/blob/main/doc/prefixReward_v11/reward_diff_plot.png" alt="training" width="600">
</div>

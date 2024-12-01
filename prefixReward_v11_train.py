"""
2411252359
cr yasaisen
next add config to init
"""

import argparse
import yaml
from dataset.prefixReward_fromGemma7b.ppoGemma7b_datasets import ComparativeBuilder
from models.prefixReward_v11.prefixReward import ComparativeRewardModel
from tasks.prefixReward_v11.prefixReward_task import RewardModelTask
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True)
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    print('>>> Congif:', cfg)
    return cfg


#######################################################################################################
import os
import pandas as pd

def get_df():
    r"""
    df cols: {
        `sample_idx`, `data_idx`, `data_idx`, 
        `assistant1`, `user1`, 
        `assistant2`, `user2`, 
        `assistant3`, `user3`, 
        `original_answer`, `gemma-7b_response`, 
        `Fit the context`, `Localization`, `Concision`, `Truthfulness`, `Harmfulness`, `Overall satisfaction`, `ranking`
        }
    """
    # ROOT_PATH = os.getcwd()
    ROOT_PATH = '/home/yasaisen/Desktop/24_research/research_main/lab_10'
    DATA_PATH_1 = os.path.join(ROOT_PATH, 'new_gemma-7b-it_responses_t07_2411130709_rated&ranked.xlsx') # & -> -
    DATA_PATH_2 = os.path.join(ROOT_PATH, 'gemma-7b-it_responses_edit.csv')

    df_1 = pd.read_excel(DATA_PATH_1)
    df_2 = pd.read_csv(DATA_PATH_2)
    df_2['ranking'] = df_2['Overall Satisfaction']
    df_2.loc[327-249-1, 'ranking'] = 3.0
    df = pd.concat([df_1, df_2], ignore_index=True)
    return df
#######################################################################################################


df = get_df()


cfg = get_cfg()
train_loader, val_loader = ComparativeBuilder.get_dataloaders(cfg['dataset'], df)
model = ComparativeRewardModel.from_config(cfg['model'])
trainer = RewardModelTask(model, train_loader, val_loader, cfg)

trainer.train_model()




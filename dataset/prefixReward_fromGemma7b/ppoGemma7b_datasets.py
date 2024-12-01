"""
2411252359
cr yasaisen
next add config to init
"""


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import random


class ComparativeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.contexts = [item[0] for item in data]
        self.better_responses = [item[1] for item in data]
        self.worse_responses = [item[2] for item in data]
        self.explanations = [item[3] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.contexts)
    
    def truncate_from_beginning(self, text):
        tokens = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        
        if tokens['input_ids'].shape[1] > self.max_length:
            start_idx = tokens['input_ids'].shape[1] - self.max_length + 1
            truncated_input_ids = torch.cat([
                tokens['input_ids'][:, :1], 
                tokens['input_ids'][:, start_idx:]
            ], dim=1)
            truncated_attention_mask = torch.cat([
                tokens['attention_mask'][:, :1],
                tokens['attention_mask'][:, start_idx:]
            ], dim=1)
            
            return {
                'input_ids': truncated_input_ids,
                'attention_mask': truncated_attention_mask
            }
        else:
            return self.tokenizer(
                text,
                truncation=False,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
    
    def __getitem__(self, idx):
        context = self.contexts[idx]
        better_response = self.better_responses[idx]
        worse_response = self.worse_responses[idx]
        
        context_encoding = self.truncate_from_beginning(context)
        better_encoding = self.truncate_from_beginning(better_response)
        worse_encoding = self.truncate_from_beginning(worse_response)
        
        return {
            'context_ids': context_encoding['input_ids'].squeeze(),
            'context_mask': context_encoding['attention_mask'].squeeze(),
            'better_ids': better_encoding['input_ids'].squeeze(),
            'better_mask': better_encoding['attention_mask'].squeeze(),
            'worse_ids': worse_encoding['input_ids'].squeeze(),
            'worse_mask': worse_encoding['attention_mask'].squeeze(),
            'contexts': context, 
            'better_responses': better_response, 
            'worse_responses': worse_response, 
        }

class ComparativeBuilder():
    def get_data_list_from_df(df: pd.DataFrame) -> list:
        role_list = ['assistant', 'user']
        data_list = []
        ranking_list = []

        data_idx_temp = -1
        sample_idx_temp = -1
        test_idx_temp = -1

        max_idx = 0
        for idx in range(len(df)):
            if str(df.iloc[idx]['ranking']) == 'nan':
                continue
            max_idx = idx
        idx = 0
        # print(idx, max_idx)
        while True:
            if int(df.iloc[idx]['data_idx']) == data_idx_temp + 1:
                sample_idx_temp = int(df.iloc[idx]['sample_idx'])
                test_idx_temp = int(df.iloc[idx]['data_idx'])

                context_messages = []
                for step in range(6):
                    if role_list[step % 2] == 'assistant' and str(df.iloc[idx][role_list[step % 2] + str(step // 2 + 1)]) == 'nan':
                        break
                    context_messages += [{
                        'role': role_list[step % 2], 
                        'content': df.iloc[idx][role_list[step % 2] + str(step // 2 + 1)].replace('\n', '')
                    }]
                data_idx_temp = data_idx_temp + 1

            ranking_temp_list = []
            response_list = []
            for running_test_idx in range(10):
                if int(df.iloc[idx + running_test_idx]['sample_idx']) == sample_idx_temp and int(df.iloc[idx + running_test_idx]['data_idx']) == data_idx_temp and int(df.iloc[idx + running_test_idx]['test_idx']) == running_test_idx:
                    data_idx_mark = 'S' + str(df.iloc[idx + running_test_idx]['sample_idx']) + 'D' + str(df.iloc[idx + running_test_idx]['data_idx']) + 'T' + str(df.iloc[idx + running_test_idx]['test_idx']) + 'V1'

                    ranking_order = int(df.iloc[idx + running_test_idx]['ranking'])
                    response = df.iloc[idx + running_test_idx]['gemma-7b_response']

                    response_list += [{'data_idx_mark': data_idx_mark, 
                                    'ranking_order': ranking_order, 
                                    'response': response
                                    }]
                    ranking_temp_list += [ranking_order]

            data_list += [{'context_messages': context_messages, 
                        'response_list': response_list
                        }]
            ranking_list += [ranking_temp_list]
            
            idx = idx + running_test_idx + 1
            if idx >= max_idx:
                break

        # print(len(data_list))
        return data_list


    def paired_data_sampler(data_list: list, pairs_sample_num: int=5) -> list:
        paired_data_list = []
        for idx in range(len(data_list)):
            context_messages = str(data_list[idx]['context_messages'])
            response_list = data_list[idx]['response_list']

            sorted_responses = sorted(response_list, key=lambda x: x['ranking_order'])

            pairs = []
            worse_idx_temp = []
            idx_pair_temp = []
            while True:
                bypass = False
                if [sorted_responses[0]['ranking_order'], sorted_responses[1]['ranking_order'], sorted_responses[2]['ranking_order']] == [1, 10, 10]:
                    better_idx = 0
                    bypass = True
                else:
                    better_idx = random.randrange(0, 3) # len(sorted_responses) - 2)

                if sorted_responses[better_idx]['ranking_order'] == 10:
                    continue

                if [sorted_responses[7]['ranking_order'], sorted_responses[8]['ranking_order'], sorted_responses[9]['ranking_order']] == [1, 1, 10]:
                    worse_idx = 9
                    bypass = True
                else:
                    worse_idx = random.randrange(better_idx + 1, len(sorted_responses) - 1)

                if (better_idx, worse_idx) in idx_pair_temp and not bypass: # worse_idx in worse_idx_temp or 
                    continue
                if sorted_responses[better_idx]['ranking_order'] == sorted_responses[worse_idx]['ranking_order']:
                    continue

                better = sorted_responses[better_idx]['response']
                worse = sorted_responses[worse_idx]['response']
                pair_idx = sorted_responses[better_idx]['data_idx_mark'] + '(' + str(sorted_responses[better_idx]['ranking_order']) + '-' + str(better_idx) + ')-' + sorted_responses[worse_idx]['data_idx_mark'] + '(' + str(sorted_responses[worse_idx]['ranking_order']) + '-' + str(worse_idx) + ')'

                paired_data_list += [(context_messages, better, worse, pair_idx)]

                pairs += [(better_idx, worse_idx)]
                worse_idx_temp += [worse_idx]
                idx_pair_temp += [(better_idx, worse_idx)]

                if len(pairs) == pairs_sample_num:
                    break

        print('>>> Total data num:', len(paired_data_list))
        return paired_data_list
    
    @classmethod
    def get_dataloaders(self, cfg: dict, df: pd.DataFrame):
        bert_name = cfg.get("bert_name", "bert-base-chinese")
        pairs_sample_num = int(cfg.get("pairs_sample_num", 5))
        tokenizer_max_length = int(cfg.get("tokenizer_max_length", 512-20))
        batch_size_train = int(cfg.get("batch_size_train"))
        batch_size_eval = int(cfg.get("batch_size_eval"))
        data_split = float(cfg.get("data_split", 0.8))

        tokenizer = BertTokenizer.from_pretrained(bert_name)

        data_list = self.get_data_list_from_df(df)
        paired_data_list = self.paired_data_sampler(data_list, pairs_sample_num=pairs_sample_num)

        dataset = ComparativeDataset(paired_data_list, tokenizer, max_length=tokenizer_max_length)

        train_size = int(data_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_eval)

        return train_loader, val_loader
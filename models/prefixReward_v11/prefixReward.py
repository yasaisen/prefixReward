"""
2411252359
cr yasaisen
next add config to init
"""


import torch
import torch.nn as nn
from transformers import BertModel


class ComparativeRewardModel(nn.Module):
    def __init__(self, prefix_length=20, bert_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.prefix_length = prefix_length
        self.hidden_size = self.bert.config.hidden_size
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, self.hidden_size)
        )
        
        self.context_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.response_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def _get_text_embedding(self, input_ids, attention_mask, text_type):
        batch_size = input_ids.shape[0]
        
        prefix_embeds = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        word_embeds = self.bert.embeddings(input_ids)
        
        inputs_embeds = torch.cat([prefix_embeds, word_embeds], dim=1)
        
        prefix_attention_mask = torch.ones(
            batch_size, self.prefix_length, 
            device=attention_mask.device
        )
        attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        cls_output = outputs.last_hidden_state[:, self.prefix_length]
        
        if text_type == "context":
            return self.context_proj(cls_output)
        else:
            return self.response_proj(cls_output)
    
    def get_reward(self, context_ids, context_mask, response_ids, response_mask):
        context_embeds = self._get_text_embedding(context_ids, context_mask, "context")
        response_embeds = self._get_text_embedding(response_ids, response_mask, "response")
        
        interaction = context_embeds * response_embeds
        
        combined = torch.cat([context_embeds, response_embeds, interaction], dim=-1)
        
        reward = self.reward_head(combined)
        return reward
    
    def forward(self, context_ids, context_mask, 
                response1_ids, response1_mask,
                response2_ids, response2_mask):

        reward1 = self.get_reward(context_ids, context_mask, 
                                response1_ids, response1_mask)
        reward2 = self.get_reward(context_ids, context_mask, 
                                response2_ids, response2_mask)
        
        return reward1, reward2
    
    @classmethod
    def from_config(cls, cfg):
        prefix_length = int(cfg.get("prefix_length"))
        bert_name = cfg.get("bert_name")
        load_pretrained = cfg.get("load_pretrained")
        weight_path = cfg.get("weight_path")

        model = cls(
            prefix_length=prefix_length,
            bert_name=bert_name,
        )

        if load_pretrained:
            print('>>> Loading weight from:', weight_path)
            msg = model.load_state_dict(torch.load(weight_path)['model_state_dict'])
            print(msg)
        else:
            print('>>> Train from scratch')

        print(f">>> Total parameters:     {sum(p.numel() for p in model.parameters())}")
        print(f">>> Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        return model
import torch
import numpy as np

def print_model_sparsity(network):
    for name, layer in network.named_modules():
        if type(layer) == torch.nn.Linear:
            print(name, 1 - np.mean(np.abs(layer.weight.data.cpu().numpy()) > 1e-10))

from transformers import (
   AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)

model_name = "/hdd1/jianwei/workspace/robust_ticket_soups/dense/tmp/finetune_glue-sst2_lr2e-05_epochs5_seed42_time1682642736786/epoch4"
config = AutoConfig.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

print_model_sparsity(model)

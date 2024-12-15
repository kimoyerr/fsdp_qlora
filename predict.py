import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModel
from safetensors import safe_open
from safetensors.torch import load_file
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoConfig
import pandas as pd
from transformers import EsmForMaskedLM, EsmTokenizer
from confit.data_utils import Mutation_Set

from confit.train import evaluate
from confit.stat_utils import spearman, compute_score, BT_loss, KLloss

# Path to the safetensors file
safetensors_file = "/home/ubuntu/Krishna-Llama-south-1/train_outputs/CAPSD_Ogden_2019/model_state_dict_epoch_0.safetensors"
lora_config = LoraConfig.from_pretrained("/home/ubuntu/Krishna-Llama-south-1/train_outputs/CAPSD_Ogden_2019")

# Load the state dict from safetensors 
state_dict = load_file(safetensors_file)

torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
# Load the LoraConfig 
base_model = AutoModelForMaskedLM.from_pretrained(
                    lora_config.base_model_name_or_path,
                    use_cache=False,
                    torch_dtype=torch_dtype,
                    _attn_implementation="eager"
                )

peft_config = lora_config

# Wrap the model with PEFT 
peft_model = get_peft_model(base_model, peft_config)
missing_keys, unexpected_keys =  peft_model.load_state_dict(state_dict, strict=False)

# Dataloader
test_csv = pd.read_csv("/home/ubuntu/Krishna-Llama-south-1/fsdp_qlora/data/CAPSD_Ogden_2019/test_small.csv")
wt_path = "/home/ubuntu/Krishna-Llama-south-1/fsdp_qlora/data/CAPSD_Ogden_2019/wt.fasta"
    
tokenizer = EsmTokenizer.from_pretrained(lora_config.base_model_name_or_path)
dataset_name = "CAPSD_Ogden_2019"
testset = Mutation_Set(wt_path, data=test_csv, fname=dataset_name,  tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=12, collate_fn=testset.collate_fn)

# Use for predictions
peft_model.eval()
# Push to device
peft_model = peft_model.to("cuda:0")
print(f"testing_test")
seq_list = []
score_list = []
gscore_list = []
with torch.no_grad():
    for step, data in enumerate(testloader):
        print("step", step)
        seq, mask = data[0], data[1]
        wt, wt_mask = data[2], data[3]
        pos = data[4]
        pos = [p.to("cuda") for p in pos]
        golden_score = data[5]

        score, logits, _, _ = compute_score(peft_model, seq.cuda(), mask.cuda(), wt.cuda(), pos, tokenizer)
        score = score.cuda()
        golden_score = golden_score.cuda()
        score_list.extend(score.cpu())
        gscore_list.extend(golden_score.cpu())
score_list = np.asarray(score_list)
gscore_list = np.asarray(gscore_list)
sr = spearman(score_list, gscore_list)
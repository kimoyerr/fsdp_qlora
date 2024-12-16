import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModel
from safetensors import safe_open
from safetensors.torch import load_file
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, EsmTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoConfig
import pandas as pd
from confit.data_utils import Mutation_Set

from confit.train import evaluate
from confit.stat_utils import spearman, compute_score, BT_loss, KLloss

# Path to the safetensors file
data_dir = "/home/ubuntu/Krishna-Llama-south-1/fsdp_qlora/data/CAPSD_Ogden_2019"
ckpt_dir = "/home/ubuntu/Krishna-Llama-south-1/train_outputs/CAPSD_Ogden_2019"
dataset_name = "CAPSD_Ogden_2019"
epoch = 3
batch_size = 192
test_dataset_name = "test"

safetensors_file = os.path.join(ckpt_dir, f"model_state_dict_epoch_{epoch}.safetensors")
lora_config = LoraConfig.from_pretrained(ckpt_dir)
test_data_path = os.path.join(data_dir, f"{test_dataset_name}.csv")

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
test_df= pd.read_csv(os.path.join(data_dir, f"{test_dataset_name}.csv"))
wt_path = os.path.join(data_dir, "wt.fasta")
    
tokenizer = EsmTokenizer.from_pretrained(lora_config.base_model_name_or_path)
testset = Mutation_Set(wt_path, data=test_df, fname=dataset_name,  tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn, shuffle=False)

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
        
        # Decode sequences
        seq = tokenizer.batch_decode(seq, skip_special_tokens=True)
        # Remove ' ' from the sequence
        seq = [s.replace(" ", "") for s in seq]
        seq_list.extend(seq)
        
score_list = np.asarray(score_list)
gscore_list = np.asarray(gscore_list)
# Write to csv
pred_df = pd.DataFrame({"pred_score": score_list, "golden_score": gscore_list, "seq": seq_list})
pred_file = os.path.join(data_dir, f"{test_dataset_name}_pred.csv")
merged_pred_file = os.path.join(data_dir, f"{test_dataset_name}_merged_pred.csv")
# Merge with original data
merged_pred_df = pd.merge(pred_df, test_df, how="left", on="seq")  # There will be 
sr = spearman(score_list, gscore_list)
print(f"Spearman: {sr}")
pred_df.to_csv(pred_file, index=False)
merged_pred_df.to_csv(merged_pred_file, index=False)
print("all done")
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


def log_likelihood_batch(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean_and_sum",
) -> torch.Tensor:
    """
    Calculate the log likelihood for a batch of sequences.

    Args:
        logits (torch.Tensor): The predicted logits from the model, of shape (batch_size, seq_length, vocab_size).
        target (torch.Tensor): The target sequence indices, of shape (batch_size, seq_length).
        reduction (str, optional): Specifies the reduction type to apply to the output.
            Options are 'mean', 'sum', or 'mean_and_sum'. Defaults to 'mean_and_sum'.

    Returns:
        torch.Tensor or tuple: Depending on the reduction type, returns either a tensor
        of log likelihoods for each sequence or a tuple containing the mean and sum of
        log likelihoods for each sequence.
    """
    assert logits.shape[0] == target.shape[0]
    assert logits.shape[1] == target.shape[1]

    # Get number of sequences in the batch
    seq_length = logits.shape[1]
    num_seqs = logits.shape[0]

    # Reshape to batch_size*seq_length, vocab_size
    logits = logits.reshape(-1, logits.size(-1))
    target = target.reshape(-1)

    # Cross entropy loss
    losses = -torch.nn.functional.cross_entropy(
        logits, target, reduction="none", ignore_index=-5
    )  # ignore padding token which originally was 0 and now is -5
    # Get the log likelihood for each sequence in the batch by averaging the losses in chunks of seq_length
    if reduction == "sum":
        losses = losses.reshape(num_seqs, seq_length).sum(dim=1)
    elif reduction == "mean":
        losses = losses.reshape(num_seqs, seq_length).mean(dim=1)
    elif reduction == "mean_and_sum":
        losses = (
            losses.reshape(num_seqs, seq_length).mean(dim=1),
            losses.reshape(num_seqs, seq_length).sum(dim=1),
        )

    return losses

# Function to calculate the log likelihood from logits
def calc_likelihood_from_logits(model_inputs, logits):
    """
    Calculate the log likelihood for a batch of sequences from logits.

    Args:
        model_inputs (torch.Tensor): The input sequence indices, of shape (batch_size, seq_length).
        logits (torch.Tensor): The predicted logits from the model, of shape (batch_size, seq_length, vocab_size).

    Returns:
        tuple: A tuple containing the mean and sum of log likelihoods for each sequence.
    """
    
    assert len(logits.shape) == 3
    # shift by one token
    logits = logits[:, :-1, ...]
    target = model_inputs[:, 1:, ...]
    assert target.shape[1] == logits.shape[1]

    # remove terminals
    bos_token, eos_token = 3, 4
    if (target[:, -1] == bos_token).any(dim=0) or (
        target[:, -1] == eos_token
    ).any(dim=0):
        logits = logits[:, :-1, ...]
        target = target[:, :-1]

    assert (target == bos_token).sum() == 0
    assert (target == eos_token).sum() == 0
    assert target.shape[1] == logits.shape[1]

    # remove unused logits
    first_token, last_token = 5, 29
    logits = logits[:, :, first_token : (last_token + 1)]
    target = target - first_token

    assert logits.shape[2] == (last_token - first_token + 1)

    # calculate likelihood
    target = target.to(logits.device)
    lll = log_likelihood_batch(
        logits=logits, target=target, reduction="mean_and_sum"
    )
    # Separate mean and sum
    lll_mean = lll[0].cpu().numpy()
    lll_sum = lll[1].cpu().numpy()

    return lll_mean, lll_sum, 



# Path to the safetensors file
data_dir = "/home/jupyter/Capsid/data/AAV5_VR4"
ckpt_dir = "/home/jupyter/train_outputs/AAV5_VR4/"
dataset_name = "AV5-VR4-test"
epoch = 0
batch_size = 48
test_dataset_name = "test_small"

safetensors_file = os.path.join(ckpt_dir, f"model_state_dict_epoch_{epoch}.safetensors")
lora_config = LoraConfig.from_pretrained(ckpt_dir)
test_data_path = os.path.join(data_dir, f"{test_dataset_name}.csv")

# Load the state dict from safetensors 
state_dict = load_file(safetensors_file)

torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
# Load the LoraConfig 
if lora_config.task_type == TaskType.CAUSAL_LM:
    base_model = AutoModelForCausalLM.from_pretrained(
                    lora_config.base_model_name_or_path,
                    use_cache=False,
                    torch_dtype=torch_dtype,
                    _attn_implementation="eager",
                    trust_remote_code=True,
                )
    tokenizer = AutoTokenizer.from_pretrained(
        lora_config.base_model_name_or_path, trust_remote_code=True
    )
    padding_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    # To avoid progen2 adding endoftext token instead of padding token
    tokenizer.pad_token_id = padding_token_id
else:
    base_model = AutoModelForMaskedLM.from_pretrained(
                    lora_config.base_model_name_or_path,
                    use_cache=False,
                    torch_dtype=torch_dtype,
                    _attn_implementation="eager",
                    trust_remote_code=True,
                )
    tokenizer = EsmTokenizer.from_pretrained(lora_config.base_model_name_or_path)

peft_config = lora_config

# Wrap the model with PEFT 
peft_model = get_peft_model(base_model, peft_config)
missing_keys, unexpected_keys =  peft_model.load_state_dict(state_dict, strict=False)

# Dataloader
test_df= pd.read_csv(os.path.join(data_dir, f"{test_dataset_name}.csv"))
wt_path = os.path.join(data_dir, "wt.fasta")
testset = Mutation_Set(wt_path, data=test_df, fname=test_dataset_name,  tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn, shuffle=False)

# Use for predictions
peft_model.eval()
# Push to device
peft_model = peft_model.to("cuda:0")
print(f"testing_test")
seq_list = []
score_list = []
gscore_list = []
lll_mean_list = []
lll_sum_list = []
with torch.no_grad():
    for step, data in enumerate(testloader):
        print("step", step)
        seq, mask = data[0], data[1]
        wt, wt_mask = data[2], data[3]
        pos = data[4]
        pos = [p.to("cuda") for p in pos]
        golden_score = data[5]

        score, logits, _, _ = compute_score(peft_model, seq.cuda(), mask.cuda(), wt.cuda(), pos, tokenizer)
        # lll = calc_likelihood_from_logits(seq, logits)
        score = score.cuda()
        golden_score = golden_score.cuda()
        score_list.extend(score.cpu())
        gscore_list.extend(golden_score.cpu())
        # Check if task_type is CAUSAL_LM
        if lora_config.task_type == TaskType.CAUSAL_LM:
            lll = calc_likelihood_from_logits(seq, logits)
            lll_mean_list.extend(lll[0])
            lll_sum_list.extend(lll[1])
        
        # Decode sequences
        seq = tokenizer.batch_decode(seq, skip_special_tokens=True)
        # Remove ' ' from the sequence
        seq = [s.replace(" ", "") for s in seq]
        seq_list.extend(seq)
        
score_list = np.asarray(score_list)
gscore_list = np.asarray(gscore_list)
# Write to csv
if lora_config.task_type == TaskType.CAUSAL_LM:
    pred_df = pd.DataFrame({"pred_score": score_list, "golden_score": gscore_list, "lll_mean": lll_mean_list, "lll_sum": lll_sum_list, "seq": seq_list})
else:
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
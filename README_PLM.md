# fsdp_qlora for finetuning Protein Language Models (PLMs)

See the "README" file for general information about running FSDP_QLORA from Answer.AI. This README is specifically for finteuning PLMs 

## Data
All data used for training/finetuning is on S3 and versioned using [DVC](https://dvc.org/doc/start)

## Pre-process proteingym data
To pre-process proteingym data to be used for ConFit fintuning use the ```nbs/protein_data_preprocessing.ipynb``` notebook

### Useful DVC commands
```dvc pull``` to pull to the local repo folde, the latest version of the data from S3
```dvc add``` if you make any changes to the files
```dvc push``` to push the changed data back to S3

# Installation Instructions for Finetuning using LoRA and ConFit
```
pip3 install datasets
pip3 install safetensors
pip3 install transformers
pip3 install git+https://github.com/huggingface/accelerate
pip3 install git+https://github.com/huggingface/peft.git
pip3 install bitsandbytes
pip3 install fastcore
pip3 install biopython
pip3 install wandb
pip3 install seaborn
pip3 install debugpy
```

```
cd /home/ubuntu/Krishna-Llama/ConFit
pip install -e .
cd -

cd /home/ubuntu/Krishna-Llama/DisCo-CLIP
pip install -e .
cd -
```

# Conda install
```
conda env create --name pytorch24
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c huggingface datasets transformers safetensors accelerate peft
conda install -c conda-forge bitsandbytes wandb biopython debugpy seaborn
conda install -c fastai fastcore
```
```
python train.py --model_name hugohrban/progen2-base --precision bf16 --model_type causal --train_type lora --lora_rank 8 --lora_alpha 8 --lora_dropout 0.1 --lora_target_modules qkv_proj --dataset confit --batch_size 12 --log_to wandb --num_epochs 25 --lr 5e-4 --lr_scheduler constant --eval_interval 1 --model_save_interval 1 --gradient_accumulation_steps 1 --protein_dataset AAV5_VR4 --protein_trainset_path /home/jupyter/fsdp_qlora/data/AAV5_VR4/train_small.csv --protein_valset_path /home/jupyter/fsdp_qlora/data/AAV5_VR4/test_small.csv --protein_testset_path /home/jupyter/fsdp_qlora/data/AAV5_VR4/test_small.csv --wt_fasta_path /home/jupyter/fsdp_qlora/data/AAV5_VR4/wt.fasta --project_name fsdp_qlora_confit_fewshot_finetuning --save_model True --output_dir /home/jupyter/train_outputs/AAV_VR4
```
## Finetuning
To finetune, modify the train_PLM.sh file as needed and run it

## Experiment Tracking and Logging
[Weights and biases](https://wandb.ai/inscripta/) is very nice to log all the ML runs metadata and metrics. I've created a free Inscripta account that you can join and we will all be able to see the reports and runs there. Make sure to turn this on as an argument to train.py

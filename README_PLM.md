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

## Finetuning
To finetune, modify the train_PLM.sh file as needed and run it

## Experiment Tracking and Logging
[Weights and biases](https://wandb.ai/inscripta/) is very nice to log all the ML runs metadata and metrics. I've created a free Inscripta account that you can join and we will all be able to see the reports and runs there. Make sure to turn this on as an argument to train.py

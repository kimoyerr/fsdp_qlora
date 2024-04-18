# fsdp_qlora for finetuning Protein Language Models (PLMs)

See the "README" file for general information about running FSDP_QLORA from Answer.AI. This README is specifically for finteuning PLMs 

## Data
All data used for training/finetuning is on S3 and versioned using [DVC](https://dvc.org/doc/start)

### Useful DVC commands
```dvc pull``` to pull to the local repo folde, the latest version of the data from S3
```dvc add``` if you make any changes to the files
```dvc push``` to push the changed data back to S3


## Finetuning
To finetune, modify the train_PLM.sh file as needed and run it

## Experiment Tracking and Logging
[Weights and biases](https://wandb.ai/inscripta/) is very nice to log all the ML runs metadata and metrics. I've created a free Inscripta account that you can join and we will all be able to see the reports and runs there. Make sure to turn this on as an argument to train.py

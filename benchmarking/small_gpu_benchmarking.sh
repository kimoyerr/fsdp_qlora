python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name meta-llama/Llama-2-7b-hf --context_length 256 --train_type lora
python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name meta-llama/Llama-2-7b-hf --context_length 256 --train_type qlora
python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_cpu_offload --train_type lora
python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name meta-llama/Llama-2-7b-hf --context_length 256 --use_cpu_offload --train_type qlora
python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_cpu_offload --train_type lora
python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name meta-llama/Llama-2-13b-hf --context_length 256 --use_cpu_offload --train_type qlora
python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_cpu_offload --train_type lora
python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name codellama/CodeLlama-34b-hf --context_length 256 --use_cpu_offload --train_type qlora
python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_cpu_offload --train_type lora
python train.py --batch_size 128 --num_epochs 1 --dataset alpaca_sample --use_flash_attention --precision bf16 --mp_bf16_mode mixed --log_to wandb --model_name meta-llama/Llama-2-70b-hf --context_length 256 --use_cpu_offload --train_type qlora
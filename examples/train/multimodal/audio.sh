# pip install "transformers==4.48.*"
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --dataset 'speech_asr/speech_asr_aishell1_trainsets:validation#20000' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4

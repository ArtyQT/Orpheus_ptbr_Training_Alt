python train.py \
    --base_model_name canopylabs/3b-es_it-ft-research_release \
    --hf_token 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 \
    --dataset_name freds0/BRSpeech-TTS-Leni \
    --output_dir ./checkpoints_latest \
    --max_steps 2400 \
    --save_steps 1200 \
    --num_cpus 4 \
    --num_amostras -1

# --resume_from_checkpoint ./checkpoints_orpheus_ptbr/checkpoint-16000/

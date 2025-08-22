CUDA_VISILE_DEVICES=9 python src/evaluate_checkpoint.py \
--lora_ckpt /gpfs-flash/junlab/yexi24-postdoc/soc-fine-tuning-sd/checkpoints/AM-SVD-DAVIS-buffer/rm100.0_control1.0_smoothTrue_bufferTrue/best_val_reward_mean=0.3983_epoch=9.ckpt \
--output_dir ./eval_results/ #\
# --use_base_model

CUDA_VISILE_DEVICES=9 python eval_phyvid.py --master_port 12173
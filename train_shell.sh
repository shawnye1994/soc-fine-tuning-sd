# CUDA_VISIBLE_DEVICES=8,9 python src/train.py --config configs/multi_prompt.yaml
# CUDA_VISIBLE_DEVICES=8,9 python /gpfs-flash/junlab/yexi24-postdoc/eval_phyvid.py --master_port 12490

CUDA_VISIBLE_DEVICES=5 python src/video_train.py --config configs/svd_athestics_buffer.yaml
CUDA_VISIBLE_DEVICES=5 python /gpfs-flash/junlab/yexi24-postdoc/eval_phyvid.py --master_port 12490
CUDA_VISIBLE_DEVICES=0,1
GPU_NUM=2
export NCCL_MIN_NCHANNELS=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=$GPU_NUM generate_video.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B \
    --wav2vec_dir models/wav2vec2-base-960h \
    --model_type pro \
    --cond_image examples/girl.png \
    --audio_path examples/podcast_sichuan_16k.wav \
    --audio_encode_mode stream
CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python generate_video.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B \
    --wav2vec_dir models/wav2vec2-base-960h \
    --model_type lite \
    --cond_image examples/girl.png \
    --audio_path examples/podcast_sichuan_16k.wav \
    --audio_encode_mode stream
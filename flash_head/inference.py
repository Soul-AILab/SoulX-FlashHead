# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import yaml
import torch
import copy
from loguru import logger

from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline
from flash_head.src.distributed.usp_device import get_device, get_parallel_degree

with open("flash_head/configs/infer_params.yaml", "r") as f:
    infer_params = yaml.safe_load(f)

def get_pipeline(world_size, ckpt_dir, model_type, wav2vec_dir):
    global infer_params
    ulysses_degree, ring_degree = get_parallel_degree(world_size, infer_params['num_heads'])
    device = get_device(ulysses_degree, ring_degree)
    logger.info(f"ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, device: {device}")

    pipeline = FlashHeadPipeline(
        checkpoint_dir=ckpt_dir,
        model_type=model_type,
        wav2vec_dir=wav2vec_dir,
        device=device,
        use_usp=(world_size > 1),
    )

    # compute motion_frames_num
    motion_frames_latent_num = infer_params['motion_frames_latent_num']
    motion_frames_num = (motion_frames_latent_num - 1) * pipeline.config.vae_stride[0] + 1
    infer_params['motion_frames_num'] = motion_frames_num

    # TODO: move to args
    if model_type == "pretrained":
        infer_params['sample_steps'] = 20
    else:
        infer_params['sample_steps'] = 4
    return pipeline

def get_base_data(pipeline, cond_image_path_or_dir, base_seed, use_face_crop):
    pipeline.prepare_params(
        cond_image_path_or_dir=cond_image_path_or_dir,
        target_size=(infer_params['height'], infer_params['width']),
        frame_num=infer_params['frame_num'],
        motion_frames_num=infer_params['motion_frames_num'],
        sampling_steps=infer_params['sample_steps'],
        seed=base_seed,
        shift=infer_params['sample_shift'],
        color_correction_strength=infer_params['color_correction_strength'],
        use_face_crop=use_face_crop,
    )

def get_infer_params():
    global infer_params
    return copy.deepcopy(infer_params)

def get_audio_embedding(pipeline, audio_array, audio_start_idx=-1, audio_end_idx=-1):
    # audio_array = loudness_norm(audio_array, infer_params['sample_rate'])
    audio_embedding = pipeline.preprocess_audio(audio_array, sr=infer_params['sample_rate'], fps=infer_params['tgt_fps'])

    if audio_start_idx == -1 or audio_end_idx == -1:
        audio_start_idx = 0
        audio_end_idx = audio_embedding.shape[0]

    indices = (torch.arange(2 * 2 + 1) - 2) * 1

    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_end_idx-1)

    audio_embedding = audio_embedding[center_indices][None,...].contiguous()
    return audio_embedding

def run_pipeline(pipeline, audio_embedding):
    audio_embedding = audio_embedding.to(pipeline.device)
    sample = pipeline.generate(audio_embedding)
    sample_frames = (((sample+1)/2).permute(1,2,3,0).clip(0,1) * 255).contiguous()
    return sample_frames


"""
Gradio 流式视频生成：视频生成&视频保存异步进行，确保实时性
"""
import gradio as gr
import os
import torch
import numpy as np
import time
import wave
import imageio
import librosa
import subprocess
import queue
import threading
from datetime import datetime
from collections import deque
from loguru import logger

from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
    get_audio_embedding,
    run_pipeline,
)

# gr.Video 的 streaming=True 要求视频片段大于1s，实际需要接近3s才能不卡顿。
# 为了适配，每 3 个 chunk 合并为一段视频
CHUNKS_PER_SEGMENT = 3

pipeline = None
loaded_ckpt_dir = None
loaded_wav2vec_dir = None
loaded_model_type = None


def _write_frames_to_mp4(frames_list, video_path, fps):
    """将帧列表写入 MP4（仅视频轨）。"""
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    with imageio.get_writer(
        video_path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0"],
    ) as writer:
        for frames in frames_list:
            frames_np = frames.numpy().astype(np.uint8)
            for i in range(frames_np.shape[0]):
                writer.append_data(frames_np[i, :, :, :])
    return video_path


def save_video_with_audio(frames_list, video_path, audio_path, fps):
    """写入完整视频并混入完整音频（-shortest 保证音画同步，yuv420p + faststart 保证浏览器可播）。"""
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    _write_frames_to_mp4(frames_list, temp_path, fps)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            # "-shortest",
            video_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return video_path

def _save_chunk_audio_to_wav(audio_array, wav_path, sample_rate=16000):
    """将一段 float32 [-1,1] 的音频数组保存为 wav 文件。"""
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    samples = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return wav_path

def run_inference_streaming(
    ckpt_dir,
    wav2vec_dir,
    model_type,
    cond_image,
    audio_path,
    seed,
    use_face_crop,
    progress=gr.Progress(),
):
    """
    流式推理：主程序监控 res_queue，有 frames 就保存并 yield；
    推理在独立线程中执行，按 chunk 顺序 infer，结果放入 res_queue。
    """
    global pipeline, loaded_ckpt_dir, loaded_wav2vec_dir, loaded_model_type

    is_cold_start = pipeline is None
    if (
        pipeline is None
        or loaded_ckpt_dir != ckpt_dir
        or loaded_wav2vec_dir != wav2vec_dir
        or loaded_model_type != model_type
    ):
        is_cold_start = True
        if is_cold_start:
            progress(0.1, desc="Loading model weights (first run takes ~1-2 min for compilation)...")
        else:
            progress(0.1, desc="Loading Model...")
        logger.info(f"Loading pipeline with ckpt_dir={ckpt_dir}, wav2vec_dir={wav2vec_dir}")
        try:
            pipeline = get_pipeline(
                world_size=1,
                ckpt_dir=ckpt_dir,
                model_type=model_type,
                wav2vec_dir=wav2vec_dir,
            )
            loaded_ckpt_dir = ckpt_dir
            loaded_wav2vec_dir = wav2vec_dir
            loaded_model_type = model_type
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise gr.Error(f"Failed to load model: {e}")

    if is_cold_start:
        progress(0.4, desc="Encoding condition image (compiling VAE, may take ~20s)...")
    else:
        progress(0.4, desc="Preparing Data...")
    base_seed = int(seed) if seed >= 0 else 9999
    try:
        get_base_data(
            pipeline,
            cond_image_path_or_dir=cond_image,
            base_seed=base_seed,
            use_face_crop=use_face_crop,
        )
    except Exception as e:
        logger.error(f"Error in get_base_data: {e}")
        raise gr.Error(f"Error processing inputs: {e}")

    infer_params = get_infer_params()
    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num

    try:
        human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise gr.Error(f"Failed to load audio file: {e}")

    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps

    stream_dir = os.path.join("gradio_results", "stream_preview")
    os.makedirs(stream_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    accumulated = []

    # 默认使用 stream 模式：准备 chunk 切片
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    remainder = len(human_speech_array_all) % human_speech_array_slice_len
    if remainder > 0:
        pad_length = human_speech_array_slice_len - remainder
        human_speech_array_all = np.concatenate(
            [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
        )
    human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)
    total_chunks = len(human_speech_array_slices)
    if total_chunks == 0:
        raise gr.Error("Audio too short: no chunks to generate. Please use a longer audio.")

    # Data prepare：按每 k 个 chunk 合并为一段 wav 保存（时间戳+segment_id 命名）
    segment_audio_paths = {}
    num_segments = (total_chunks + CHUNKS_PER_SEGMENT - 1) // CHUNKS_PER_SEGMENT
    for segment_id in range(num_segments):
        start = segment_id * CHUNKS_PER_SEGMENT
        end = min(start + CHUNKS_PER_SEGMENT, total_chunks)
        audio_concat = np.concatenate(
            [human_speech_array_slices[i] for i in range(start, end)]
        )
        segment_audio_name = f"audio_{timestamp}_seg_{segment_id:04d}.wav"
        segment_audio_path = os.path.join(stream_dir, segment_audio_name)
        _save_chunk_audio_to_wav(
            audio_concat,
            segment_audio_path,
            sample_rate=sample_rate,
        )
        segment_audio_paths[segment_id] = segment_audio_path
    logger.info(
        f"Pre-saved {num_segments} segment audios (every {CHUNKS_PER_SEGMENT} chunks) under {stream_dir}"
    )

    # 结果队列：推理线程放入 (chunk_idx, chunk_frames_np)，主线程根据 chunk_id 取对应音频合并
    res_queue = queue.Queue()

    def inference_worker():
        """单独线程：按 chunk 顺序执行 infer，每生成一帧就放入 res_queue，立即继续下一 chunk。"""
        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
            torch.cuda.synchronize()
            start_time = time.time()
            video = run_pipeline(pipeline, audio_embedding)
            video = video[motion_frames_num:]
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            logger.info(f"Infer chunk-{chunk_idx} done, cost time: {elapsed:.2f}s")
            chunk_frames_np = video.cpu().numpy()
            res_queue.put((chunk_idx, chunk_frames_np))
        res_queue.put(None)

    if is_cold_start:
        progress(0.6, desc="Warming up model (compiling kernels, first chunk takes ~60s)...")
    else:
        progress(0.7, desc="Starting generation...")

    worker_thread = threading.Thread(target=inference_worker)
    worker_thread.start()
    logger.info("Inference worker thread started. Main will consume res_queue and yield video paths.")

    frame_buffer = []
    warmup_notified = False
    while True:
        while res_queue.empty():
            if not warmup_notified and is_cold_start:
                progress(0.7, desc="Compiling CUDA kernels (one-time, ~60s)... Subsequent runs will be fast.")
                warmup_notified = True
            time.sleep(0.5)
        item = res_queue.get()
        if item is None:
            break
        chunk_idx, chunk_frames_np = item
        chunk_frames = torch.from_numpy(chunk_frames_np)
        accumulated.append(chunk_frames)
        frame_buffer.append(chunk_frames)
        if len(frame_buffer) == CHUNKS_PER_SEGMENT:
            segment_id = (chunk_idx + 1 - CHUNKS_PER_SEGMENT) // CHUNKS_PER_SEGMENT
            segment_audio_path = segment_audio_paths[segment_id]
            segment_path = os.path.join(
                stream_dir, f"preview_{timestamp}_seg_{segment_id:04d}.mp4"
            )
            save_video_with_audio(
                frame_buffer,
                segment_path,
                segment_audio_path,
                fps=tgt_fps,
            )
            logger.info(
                f"Saved segment-{segment_id} (chunks {segment_id * CHUNKS_PER_SEGMENT}-{chunk_idx}) and yielding to frontend."
            )
            yield os.path.abspath(segment_path)
            frame_buffer = []

    # 不足 k 的剩余 chunk 合并为最后一段
    if frame_buffer:
        segment_id = num_segments - 1
        segment_audio_path = segment_audio_paths[segment_id]
        segment_path = os.path.join(
            stream_dir, f"preview_{timestamp}_seg_{segment_id:04d}.mp4"
        )
        save_video_with_audio(
            frame_buffer,
            segment_path,
            segment_audio_path,
            fps=tgt_fps,
        )
        logger.info(
            f"Saved final segment-{segment_id} ({len(frame_buffer)} chunks) and yielding to frontend."
        )
        yield os.path.abspath(segment_path)

    worker_thread.join()

    if not accumulated:
        raise gr.Error("No video frames generated. Please check inputs and try again.")

    output_dir = "gradio_results"
    os.makedirs(output_dir, exist_ok=True)
    final_filename = f"res_{timestamp}.mp4"
    final_path = os.path.join(output_dir, final_filename)
    save_video_with_audio(accumulated, final_path, audio_path, fps=tgt_fps)
    logger.info(f"Saved to {final_path}")


# ---------- Gradio UI ----------
with gr.Blocks(title="SoulX-FlashHead 流式视频生成", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚡ SoulX-FlashHead 流式视频生成")
    gr.Markdown("上传图片与音频，边生成边播放，音画同步。当前仅支持单GPU。")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎬 生成输入")
                with gr.Row():
                    cond_image_input = gr.Image(
                        label="Condition Image",
                        type="filepath",
                        value="examples/girl.png",
                        height=300,
                    )
                    audio_path_input = gr.Audio(
                        label="Audio Input",
                        type="filepath",
                        value="examples/podcast_sichuan_16k.wav",
                    )
            generate_btn = gr.Button("🚀 流式生成视频", variant="primary", size="lg")
            with gr.Accordion("⚙️ 高级设置", open=False):
                ckpt_dir_input = gr.Textbox(
                    label="FlashHead Checkpoint Directory",
                    value="models/SoulX-FlashHead-1_3B",
                )
                wav2vec_dir_input = gr.Textbox(
                    label="Wav2Vec Directory",
                    value="models/wav2vec2-base-960h",
                )
                model_type_input = gr.Dropdown(
                    label="Model Type",
                    choices=["pro", "lite"],
                    value="lite",
                )
                use_face_crop_input = gr.Checkbox(label="Use Face Crop", value=False)
                seed_input = gr.Number(label="Random Seed", value=9999, precision=0)
        with gr.Column(scale=1):
            gr.Markdown("### 📺 输出视频（流式更新）")
            video_output = gr.Video(
                label="Generated Video",
                height=512,
                format="mp4",
                streaming=True,
                autoplay=True,
            )

    generate_btn.click(
        fn=run_inference_streaming,
        inputs=[
            ckpt_dir_input,
            wav2vec_dir_input,
            model_type_input,
            cond_image_input,
            audio_path_input,
            seed_input,
            use_face_crop_input,
        ],
        outputs=video_output,
    )

if __name__ == "__main__":
    app.launch()
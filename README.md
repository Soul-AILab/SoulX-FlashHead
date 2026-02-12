<div align="center">


<h1>SoulX-FlashHead: Oracle-guided Generation of Infinite Real-time Streaming Talking Heads</h1>

[Tan Yu*](https://jiayoujiayoujiayoua.github.io/), [Qian Qiao*](https://qianqiaoai.github.io/)<sup>&#9993;</sup>, [Le Shen*](https://openreview.net/profile?id=%7ELe_Shen3), [Ke Zhou](https://github.com/jokerz0624), [Jincheng Hu](#), [Dian Sheng](#), [Bo Hu](#), [Haoming Qin](#), [Jun Gao](#), [Changhai Zhou](#), [Shunshun Yin](#), [Siyuan Liu](#) <sup>&#9993;</sup>


<sup>*</sup>Equal Contribution
<sup>&#9993;</sup>Corresponding Author


<a href='https://soul-ailab.github.io/soulx-flashhead/' target="_blank"><img src='https://img.shields.io/badge/Project-Page-green'></a> <a href='https://arxiv.org/pdf/2602.07449' target="_blank"><img src='https://img.shields.io/badge/Technical-Report-red'></a>
<a href='https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B' target="_blank"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href="https://huggingface.co/datasets/Soul-AILab/VividHead" target="_blank"><img src="https://img.shields.io/badge/🤗 Hugging Face-Dataset-blue" alt="Dataset"></a>&nbsp;
</div>

## ⚡ Highlights
- **Model_Lite** [Released](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B/tree/main/Model_Lite) get 96 FPS, or 3-concurrent real-time(25+ FPS) streaming on single RTX4090.
- **Model_Pro** [Released](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B/tree/main/Model_Pro) can generate high-quality videos with 10.8 FPS on single RTX4090, or real-time(25+ FPS) on two RTX5090.
- **Model_Pretrained** is coming soon, providing high-performance weights and experimental foundations for community research.

## 🔥 News
- **2026.02.12** - The [online demo](#online-experience-qr) is now available via the Soul App. Download it today to try it out. 
- **2026.02.12** - We have released the [inference code](https://github.com/Soul-AILab/SoulX-FlashHead), and the [model weights](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B).
- **2026.02.12** - We released **Project page** on [SoulX-FlashHead](https://soul-ailab.github.io/soulx-flashhead/).
- **2026.02.07** - We released [Dataset](https://huggingface.co/datasets/Soul-AILab/VividHead).
- **2026.02.07** - We released **SoulX-FlashHead Technical Report** on [Arxiv](https://arxiv.org/pdf/2602.07449) and [GitHub repository](./assets/SoulX_FlashHead.pdf).


## 📑 Todo List
- [x] Technical report 
- [x] Project Page
- [x] Inference code
- [x] Distilled Checkpoint of Pro-Model & Lite-Model release
- [ ] Pretrained Checkpoint release

## 🌰 Examples
More examples are available in the project.

<table>
  <tbody>
    <!-- Row 1: Videos 1-5 -->
    <tr>
      <td width="30%"><video src="https://private-user-images.githubusercontent.com/176391424/548713532-5d4800cf-d0dd-4aaf-a887-d9f202d3b4b6.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA4ODYxMTcsIm5iZiI6MTc3MDg4NTgxNywicGF0aCI6Ii8xNzYzOTE0MjQvNTQ4NzEzNTMyLTVkNDgwMGNmLWQwZGQtNGFhZi1hODg3LWQ5ZjIwMmQzYjRiNi5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMjEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDIxMlQwODQzMzdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xZWQzODMzYjYzZmE1ODc2ZjA0NDhkYzcyZGIxZDRiYzlmNTU0M2Y1ZGUxNjlmYzgzMjNhMTM1MTQ2MGNmMzM1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.e7zRf7beypjK6JnJpUnivJv-1_s937aK89TxTa2m_Sc" style="width:100%; aspect-ratio:512/512; object-fit:cover;" controls loop></video></td>
      <td width="30%"><video src="https://private-user-images.githubusercontent.com/176391424/548713758-051f5779-cd5d-4336-9326-3b2e55ccc77d.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA4ODYxNDAsIm5iZiI6MTc3MDg4NTg0MCwicGF0aCI6Ii8xNzYzOTE0MjQvNTQ4NzEzNzU4LTA1MWY1Nzc5LWNkNWQtNDMzNi05MzI2LTNiMmU1NWNjYzc3ZC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMjEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDIxMlQwODQ0MDBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mZDdlNTA4YjJjNmQzZGZlNWQwYjc0MmZkOTcyYWFhZjY2NzRjMjQyNjI3YWMyZDA5ZmI0ZGNiZDc0ODBhYTg2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.S5cJs9MLQRXKUQsC2lCZl74QQKI2orWxT-NcF4qHpr0" style="width:100%; aspect-ratio:512/512; object-fit:cover;" controls loop></video></td>
      <td width="30%"><video src="https://private-user-images.githubusercontent.com/176391424/548713661-8cdfd881-7782-403c-9dc0-e93930750dfe.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzA4ODYxMzEsIm5iZiI6MTc3MDg4NTgzMSwicGF0aCI6Ii8xNzYzOTE0MjQvNTQ4NzEzNjYxLThjZGZkODgxLTc3ODItNDAzYy05ZGMwLWU5MzkzMDc1MGRmZS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMjEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDIxMlQwODQzNTFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03ZjFkZWZiMGIxNzkyYjZjNWM1YzU5OWRlN2Q4Mzk2ZGQwZGE4OTc1NDMyYjg2YTU5OTE2MDVhYzE3ZGMyMmE3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.blrq-obdV5NBPojgWEdOaXCVriCAIZjsKQ_x_DDPQ5k" style="width:100%; aspect-ratio:512/512; object-fit:cover;" controls loop></video></td>
    </tr>

  </tbody>
</table>



## 📖 Quickstart
###  🔧 Installation
#### 1. Create a Conda environment
```bash
conda create -n flashhead python=3.10
conda activate flashhead
```
#### 2. Install PyTorch on CUDA
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```
#### 3. Install other dependencies
```bash
pip install -r requirements.txt
```
#### 4. FlashAttention installation:
```bash
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```

-- If it takes a long time, we recommend the way below.
1. download wheel file from [here](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2)
2. pip install xxx.whl

#### 5. SageAttention installation (Optional)
```bash
pip install sageattention==2.2.0 --no-build-isolation
```

#### 6. FFmpeg installation
```bash
# Ubuntu / Debian
apt-get install ffmpeg
# CentOS / RHEL
yum install ffmpeg ffmpeg-devel
```
or
```bash
# Conda (no root required) 
conda install -c conda-forge ffmpeg==7
```
### 🤗 Model download
| Model Component | Description | Link |
| :--- | :--- | :---: |
| `SoulX-FlashHead-1_3B` | Our 1.3B model| 🤗 [Huggingface](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| `wav2vec2-base-960h` | wav2vec2-base-960h | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h) |

```bash
# If you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
```
### 🚀 Inference
```bash
# Infer with [Pro-Model] on single GPU
bash inference_script_single_gpu_pro.sh


# Infer with [Pro-Model] on multy GPUs
bash inference_script_multi_gpu_pro.sh
# Real-time inference speed of Pro-Model can only be supported on two RTX-5090 with SageAttention.


# Infer with [Lite-Model] on single GPU
bash inference_script_single_gpu_lite.sh
# Real-time inference speed can be supported on single RTX-4090 (up to 3 concurrent).
```

### 👋 Online Experience 
For a real-time interactive experience, scan the QR code to enter the event link. [2026.2.12~2026.3.11]
<a id="online-experience-qr"></a>
<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/soul_event_link.png" width="200" alt="SoulApp event QR Code"/>
        <br />
        <strong>Real-time Online Experience<br>(SoulApp 实时在线体验)</strong>
      </td>
    </tr>
  </table>
</div>

## 📧 Contact Us
If you are interested in leaving a message to our work, feel free to email yutan@soulapp.cn or qiaoqian@soulapp.cn or le.shen@mail.dhu.edu.cn or zhouke@soulapp.cn or liusiyuan@soulapp.cn

We have opened a WeChat group. Additionally, we represent **SoulApp** and warmly welcome everyone to download the app and join our Soul group for further technical discussions and updates!

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/wechat_group.png" width="300" alt="WeChat Group QR Code"/>
        <br />
        <strong>Join WeChat Group<br>(加入微信技术群)</strong>
      </td>
      <td width="100"></td>
      <td align="center">
        <img src="assets/soul_group.png" width="300" alt="Soul App Group QR Code"/>
        <br />
        <strong>Download SoulApp & Join Group<br>(下载SoulApp加入群组)</strong>
      </td>
    </tr>
  </table>
</div>

 
## 📚 Citation

If you find our work useful in your research, please consider citing:

```
@misc{yu2026soulxflashheadoracleguidedgenerationinfinite,
      title={SoulX-FlashHead: Oracle-guided Generation of Infinite Real-time Streaming Talking Heads}, 
      author={Tan Yu and Qian Qiao and Le Shen and Ke Zhou and Jincheng Hu and Dian Sheng and Bo Hu and Haoming Qin and Jun Gao and Changhai Zhou and Shunshun Yin and Siyuan Liu},
      year={2026},
      eprint={2602.07449},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.07449}, 
}
```

## 🙇 Acknowledgement
- [Wan](https://github.com/Wan-Video/Wan2.1): the base model we built upon.
- [LTX-Video](https://github.com/Lightricks/LTX-Video): the VAE of our Lite-Model.
- [Self forcing](https://github.com/guandeh17/Self-Forcing): the codebase we built upon.
- [DMD](https://github.com/tianweiy/DMD2) and [Self forcing++](https://github.com/justincui03/Self-Forcing-Plus-Plus): the key distillation technique used by our method.
- [SoulX-FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk/) is another model developed by our team, featuring 14B parameters and real-time capabilities.
> [!TIP]
> If you find our work useful, please also consider starring the original repositories of these foundational methods.

## 💡 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Soul-AILab/SoulX-FlashHead&type=date&legend=top-left)](https://www.star-history.com/#Soul-AILab/SoulX-FlashHead&type=date&legend=top-left)

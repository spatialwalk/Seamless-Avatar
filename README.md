# 🎭 Seamless Avatar

A project for generating seamless dyadic interaction avatars.

## 🎬 Video Demo


https://github.com/user-attachments/assets/054f66d2-7aa9-4510-8353-f51edabf147b
> 🔊 **Note:** Click the speaker icon in the player to enable audio.


## 🛠️ Environment Setup


### Installation Steps

```bash
# Create Conda environment
conda create -n seamless-avatar python=3.10
conda activate seamless-avatar

# Install system dependencies
apt-get update && apt-get install -y ffmpeg
apt-get install ninja-build

# Install Python dependencies
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install -r requirements.txt

```

Set up Hugging Face mirror for faster model/dataset downloads if you are in a region with slow access to Hugging Face:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```


## 📁 Data Preparation

```bash
python -m src.data_preprocess.extract_tar_and_remove
python -m src.data_preprocess.compute_stats
python -m src.data_preprocess.generate_splits
```



## 🚀 Training

### Single GPU Training

```bash
python -m train_dit
```

### Multi-GPU Distributed Training

```bash
torchrun --nproc_per_node=6 -m train_dit
```


### 📉 Training Loss Curve ([Swanlab](https://swanlab.cn/))

DiT[Gesture] loss curve:
![Swanlab Chart](assets/image.png)


More Training Info:
| Model         | Training Duration      | GPUs         | Checkpoint         |  Full Training Log         |
|---------------|--------------|--------------|--------------------|--------------------|
| DiT[Gesture]  | 13 hours      | 6 RTX4090    | [epoch420](https://huggingface.co/xwshi/Seamless-Avatar-Smplx-Model/resolve/main/DiT_gesture.pt?download=true)          | 👉  [DiT[Gesture] Log](https://swanlab.cn/@gjj/Seamless-Avatar/runs/6t4cimbag950wmqmwip44/chart) |
| DiT[Expression]| 15 hours     | 6 RTX4090    | [epoch500](https://huggingface.co/xwshi/Seamless-Avatar-Smplx-Model/resolve/main/DiT_expression.pt?download=true)           | 👉  [DiT[Expression] Log](https://swanlab.cn/@gjj/Seamless-Avatar/runs/wjz3q1c942rw57e3h01yw/chart) | 


## 🧪 Inference

```bash
python -m src.motion_detokenizer.infer_dit
```



## 📊 Metrics

```bash
python -m src.metrics.emage_metric
```

### Example Results

```
====================================================================================================
Summary Table
====================================================================================================
                                     FGD ↓     L1div ↑       LVD ↓       MSE ↓
----------------------------------------------------------------------------------------------------
DiT_holistic_pred_0108_v1         9.47e-01    5.63e+00    3.91e-05    1.24e-06
====================================================================================================
```


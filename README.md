# 🎭 Seamless Avatar

A project for generating seamless dyadic interaction avatars.

## 🎬 Video Demo


https://github.com/user-attachments/assets/02828e1c-31f0-40ad-b9d0-35eab7519f42
> 🔊 **Note:** Click the speaker icon in the player to enable audio.


## 🛠️ Environment Setup


### Installation Steps

```shell
# Clone the repository
git clone https://github.com/spatialwalk/Seamless-Avatar.git
cd Seamless-Avatar/

# Create Conda environment
conda create -n seamless-avatar python=3.10 -y
conda activate seamless-avatar

# Install system dependencies
apt-get update && apt-get install -y ffmpeg
apt-get install ninja-build

# Install Python dependencies
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

```


## 📁 Download Dataset and Pre-trained Models
```shell
# Install aria2 for faster downloads
apt install aria2 -y
chmod a+x scripts/hfd.sh

# Set up Hugging Face mirror for faster model/dataset downloads if you are in a region with slow access to Hugging Face:
# export HF_ENDPOINT=https://hf-mirror.com

# Download the dataset
bash ./scripts/download_dataset.sh

# download pre-trained models
bash ./scripts/hfd.sh xwshi/Seamless-Avatar-Smplx-Model --local-dir models/pretrained_models

# download smplx model
bash ./scripts/download_smplx.sh
cp models/smplx/SMPLX_NEUTRAL_2020.npz ./src/metrics/emage_evaltools/smplx_models/smplx/ # this is for metric evaluation
```


## 🧪 Inference

```bash
python infer_dit.py # audio --> npz
python vis_infer.py # npz --> video
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
                                     FGD ↓     L1div ↑       LVD ↓       MSE ↓        BC ↓
----------------------------------------------------------------------------------------------------
DiT_holistic_pred_0108_v1         5.64e-01    8.74e+00    4.12e-05    1.30e-06    4.64e-01
====================================================================================================
```



## 🚀 Training

### Single GPU Training

```bash
python train_dit.py expression
python train_dit.py gesture
python train_dit.py hands
```

### Multi-GPU Distributed Training

```bash
torchrun --nproc_per_node=6 -m train_dit expression
torchrun --nproc_per_node=6 -m train_dit gesture
torchrun --nproc_per_node=6 -m train_dit hands
```


## 📉 Training Loss Curve ([Swanlab](https://swanlab.cn/))

DiT[Gesture] loss curve:
![Swanlab Chart](assets/image.png)


More Training Info:
| Model         | Training Duration      | GPUs         | Checkpoint         |  Training Log         |
|---------------|--------------|--------------|--------------------|--------------------|
| DiT[Gesture]  | 13 hours      | 6 RTX4090    | [epoch420](https://huggingface.co/xwshi/Seamless-Avatar-Smplx-Model/resolve/main/DiT_gesture.pt?download=true)          | 👉  [Full Log](https://swanlab.cn/@gjj/Seamless-Avatar/runs/6t4cimbag950wmqmwip44/chart) |
| DiT[Expression]| 15 hours     | 6 RTX4090    | [epoch500](https://huggingface.co/xwshi/Seamless-Avatar-Smplx-Model/resolve/main/DiT_expression.pt?download=true)           | 👉  [Full Log](https://swanlab.cn/@gjj/Seamless-Avatar/runs/wjz3q1c942rw57e3h01yw/chart) | 
| DiT[Hands]| 15 hours     | 6 RTX4090    | [epoch500](https://huggingface.co/xwshi/Seamless-Avatar-Smplx-Model/resolve/main/DiT_hands.pt?download=true)           | 👉  [Full Log](https://swanlab.cn/@gjj/Seamless-Avatar/runs/g5yme25i9ivn8o55po3di/chart) | 


## 🎬 More Video Demos

https://github.com/user-attachments/assets/65121399-e695-421d-8af6-4bbf8ce15e54
https://github.com/user-attachments/assets/7096fa08-0151-4e0c-9136-616a5def06b2
https://github.com/user-attachments/assets/2facf4be-39a2-4803-bd3e-6f38a6e1d94c
https://github.com/user-attachments/assets/62b8751c-5ae2-40a4-b0c9-48262ec70c50
https://github.com/user-attachments/assets/e345d855-a817-45b3-965e-a19c63758323
https://github.com/user-attachments/assets/22a3abb9-5064-47cd-9a49-474e7cecec4a
https://github.com/user-attachments/assets/922c99f6-2a0b-4b4e-95fb-a073ccc41d42
https://github.com/user-attachments/assets/5965bae0-4296-4cea-96f9-a0af1dda8d32
https://github.com/user-attachments/assets/ba89f31c-8977-4b66-b3be-2650ece3e40e
https://github.com/user-attachments/assets/99019257-bb85-4b03-b9f5-6d7c592dc1cf
https://github.com/user-attachments/assets/3e44a345-cb08-4ca0-b50b-5663ed2ff26a
https://github.com/user-attachments/assets/b3325294-72a1-466a-8b7d-ebd025f8166b






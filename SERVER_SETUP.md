# Server Setup & Execution Guide

This guide details how to clone the repository and run the `SelfRSSplat` model on a Linux server.

## 1. Prerequisites

Ensure your server has the following installed:
- **Git**: For cloning the repository.
- **Python 3.8+**: Recommended version.
- **CUDA Toolkit**: Required for GPU acceleration (verify with `nvcc --version`).

## 2. Clone the Repository

Clone the project from your GitHub repository:

```bash
git clone https://github.com/Merin-M/main_project.git
cd main_project
```

## 3. Environment Setup

It is highly recommended to use a virtual environment or Conda environment to manage dependencies.

### Option A: Using venv (Standard Python)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using Conda

```bash
# Create a conda environment
conda create -n selfrssplat python=3.8
conda activate selfrssplat

# Install dependencies
# Install dependencies
pip install -r requirements.txt

# IMPORTANT: Install CuPy manually based on your CUDA version
# Check your CUDA version with: nvcc --version
# Then install the matching package:
# For CUDA 11.x:
pip install cupy-cuda11x
# For CUDA 12.x:
# pip install cupy-cuda12x
```

> **Note**: If `cupy` fails to install, check your CUDA version and install the matching package (e.g., `cupy-cuda11x` for CUDA 11).

## 4. Download/Prepare Data & Models

Since large files are ignored by git, you need to manually transfer or download:
1.  **Datasets**: Place your Fastec or other datasets in a known directory (e.g., `/data/fastec_rs_train`).
2.  **Pretrained Models**: If you are resuming training or running inference, upload your pretrained weights to `Pretrain_models_SelfSoftsplat/` or the path specified in your scripts.

## 5. Running Training (Linux)

Use the provided shell scripts (e.g., `train_fastec.sh`) as a template.

1.  **Edit the script** to match your server paths:
    ```bash
    nano train_fastec.sh
    ```
    *Update `fastec_root_path_training_data`, `log_dir`, and `log_dir_pretrained_GS`.*

2.  **Make the script executable**:
    ```bash
    chmod +x train_fastec.sh
    ```

3.  **Run the training**:
    ```bash
    ./train_fastec.sh
    ```
    *Or running in background:*
    ```bash
    nohup ./train_fastec.sh > training_log.out 2>&1 &
    tail -f training_log.out
    ```

## 6. Running Inference/Demo

Similar to training, use `inference.sh` or `demo_video.sh`. Ensure paths to your trained models are correct.

```bash
chmod +x inference.sh
./inference.sh
```

## 7. Troubleshooting

### ImportError: libcudart.so.12: cannot open shared object file
This means `cupy` is installed but cannot find the CUDA runtime libraries.
1.  Verify where your CUDA is installed (usually `/usr/local/cuda-12.2/lib64`).
2.  Add it to your library path:
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.2/lib64
    ```
### ImportError: libGL.so.1: cannot open shared object file
This happens if `opencv-python` is installed on a server without GUI libraries.
**Fix:** Switch to the headless version:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```
(This is already handled in the updated `requirements.txt`).


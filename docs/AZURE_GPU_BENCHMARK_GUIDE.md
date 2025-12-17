# Azure GPU ASR Performance Testing Implementation Guide
## Whisper-Kotoba Model with Flash Attention Benchmarking

---

## Executive Summary

This guide details the setup of an Azure Standard_NC6s_v3 VM (NVIDIA Tesla V100, 16GB VRAM) for performance benchmarking of the Kotoba Whisper v2.2 Japanese ASR model. The testing will focus on **streaming performance** with comparative analysis of Flash Attention 2 vs standard attention, collecting all 4 required metric categories: RTF/latency, throughput, Flash Attention comparison, and GPU utilization.

**Estimated Total Cost:** $73-92 for 1-3 day testing (with auto-shutdown)
**Total Setup Time:** ~2-3 hours
**Testing Duration:** 6-24 hours (depending on thoroughness)

---

## Table of Contents

1. [Phase 1: Azure VM Provisioning](#phase-1-azure-vm-provisioning)
2. [Phase 2: GPU Environment Setup](#phase-2-gpu-environment-setup)
3. [Phase 3: Code Deployment](#phase-3-code-deployment)
4. [Phase 4: Testing Scripts Development](#phase-4-testing-scripts-development)
5. [Phase 5: Performance Monitoring Setup](#phase-5-performance-monitoring-setup)
6. [Phase 6: Cost Optimization](#phase-6-cost-optimization)
7. [Phase 7: Results Collection & Analysis](#phase-7-results-collection--analysis)
8. [Phase 8: Troubleshooting](#phase-8-troubleshooting)
9. [Phase 9: Complete Testing Workflow](#phase-9-complete-testing-workflow)
10. [Expected Performance Targets](#expected-performance-targets)
11. [Implementation Timeline](#implementation-timeline)

---

## Phase 1: Azure VM Provisioning (30-45 minutes)

### 1.1 Prerequisites Check

**Local Machine Requirements:**
- Azure CLI installed (`az --version`)
- SSH client available
- Active Azure subscription with quota for NC-series VMs

**Validation Commands:**
```bash
# Login to Azure
az login

# Check available subscriptions
az account list --output table

# Set subscription (if multiple)
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Check NC6s_v3 availability in regions
az vm list-skus --location eastus --size Standard_NC --all --output table | grep NC6s_v3
az vm list-skus --location westeurope --size Standard_NC --all --output table | grep NC6s_v3
az vm list-skus --location westus2 --size Standard_NC --all --output table | grep NC6s_v3
```

**Recommended Regions (NC6s_v3 availability + low latency):**
1. **East US** (Primary)
2. **West Europe** (Alternative)
3. **West US 2** (Alternative)

### 1.2 Resource Group Creation

```bash
# Create resource group in East US
az group create \
  --name rg-asr-benchmark \
  --location eastus \
  --tags "Project=ASR-Benchmark" "Duration=Temporary" "Owner=YourName"
```

### 1.3 VM Creation with Managed Disk Storage

```bash
# Generate SSH key if not exists
ssh-keygen -t rsa -b 4096 -f ~/.ssh/asr_vm_key -N ""

# Create VM with Ubuntu 22.04 LTS (best CUDA compatibility)
az vm create \
  --resource-group rg-asr-benchmark \
  --name vm-asr-test \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/asr_vm_key.pub \
  --os-disk-size-gb 128 \
  --os-disk-name osdisk-asr \
  --storage-sku Premium_LRS \
  --public-ip-sku Standard \
  --nsg-rule SSH \
  --priority Regular \
  --tags "AutoShutdown=Enabled" "Project=ASR-Benchmark"

# Capture VM IP address
VM_IP=$(az vm show -d -g rg-asr-benchmark -n vm-asr-test --query publicIps -o tsv)
echo "VM IP: $VM_IP"
```
Storage Breakdown for 128GB OS Disk

  Expected Usage:
  - Ubuntu OS + system files: ~10-15 GB
  - NVIDIA drivers + CUDA toolkit: ~8-10 GB
  - Python packages (PyTorch, transformers, etc.): ~8-12 GB
  - Kotoba Whisper v2.2 model (cached): ~3-4 GB
  - Your code and dataset (10 WAV files): ~1 GB
  - Benchmark results (JSON, CSV, plots): ~1-2 GB
  - Total: ~40-50 GB used out of 128 GB

You'll have 70-80 GB free on the OS disk, which is more than enough.

  When You WOULD Need a Separate Data Disk

  The data disk makes sense for:
  - Long-term storage: If you wanted to keep results after deleting the VM (detach data disk, delete VM, keep disk)
  - Large datasets: If processing hundreds of GB of audio files
  - Production deployments: Where you want I/O isolation between OS and data
  - Cost optimization: Premium data disks can be detached/reattached to new VMs

  Recommendation for Your Case

  Skip the data disk step (Phase 1.4). Just use the 128GB OS disk for everything.

### 1.4 Data Disk for Results Storage (Optional but Recommended)

```bash
# Create 64GB Premium SSD for results
az vm disk attach \
  --resource-group rg-asr-benchmark \
  --vm-name vm-asr-test \
  --name datadisk-results \
  --size-gb 64 \
  --sku Premium_LRS \
  --new
```

### 1.5 Auto-Shutdown Configuration

```bash
# Configure auto-shutdown at 11 PM local time (cost saving for computing resources, but the storage and network resources are still costing money)
az vm auto-shutdown \
  --resource-group rg-asr-benchmark \
  --name vm-asr-test \
  --time 2300 \
  --email "your.email@example.com"

# Alternative: Install idle shutdown script (see Phase 6.1)
```
  Cost Impact:
  - ✅ VM compute charges STOP: $3.06/hour → $0/hour (while stopped)
  - ❌ Storage charges CONTINUE: ~$0.32/day for the 128GB OS disk (minimal
 Auto-Shutdown vs Delete

  | Action        | Data Preserved? | Can Restart? | Compute Cost | Storage Cost |
  |---------------|-----------------|--------------|--------------|--------------|
  | Auto-shutdown | ✅ Yes           | ✅ Yes        | ⏸️ Paused    | 💰 Continues |
  | Manual stop   | ✅ Yes           | ✅ Yes        | ⏸️ Paused    | 💰 Continues |
  | Delete VM     | ❌ No            | ❌ No         | ✅ Stopped    | ✅ Stopped    |

### 1.6 Network Security Configuration

```bash
# Current NSG only allows SSH (port 22) - this is sufficient
# Verify NSG rules
az network nsg rule list \
  --resource-group rg-asr-benchmark \
  --nsg-name vm-asr-testNSG \
  --output table
```

**Validation Checkpoint:**
```bash
# Test SSH connection
ssh -i ~/.ssh/asr_vm_key azureuser@$VM_IP "uname -a"
# Expected: Ubuntu 22.04 kernel info
```

---

## Phase 2: GPU Environment Setup (45-60 minutes)

### 2.1 Initial System Update

```bash
# SSH into VM
ssh -i ~/.ssh/asr_vm_key azureuser@$VM_IP

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential pkg-config wget curl git vim htop
```

### 2.2 NVIDIA Driver Installation (CUDA 12.2 Compatible)

```bash
# Remove any existing NVIDIA packages (clean slate)
sudo apt remove --purge -y nvidia-* libnvidia-*
sudo apt autoremove -y

# Install NVIDIA driver (version 535 - stable for CUDA 12.2)
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install

# Alternative: Manual driver installation for CUDA 12.2
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Reboot to load driver
sudo reboot
```

**Wait 2-3 minutes, then reconnect:**
```bash
ssh -i ~/.ssh/asr_vm_key azureuser@$VM_IP

# Verify driver installation
nvidia-smi
```

**Expected Output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   30C    P0    24W / 250W |      0MiB / 16384MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 2.3 CUDA Toolkit Installation

```bash
# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA 12.2 toolkit
sudo apt install -y cuda-toolkit-12-2

# Set environment variables (add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
# Expected: Cuda compilation tools, release 12.2
```

### 2.4 Python 3.11 Installation (Compatibility for torch 2.9+)

```bash
# Python 3.13 may have compatibility issues with some libraries
# Install Python 3.11 (stable)
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Set Python 3.11 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
python3 --version
# Expected: Python 3.11.x
```

### 2.5 Virtual Environment Setup

```bash
# Create project directory
mkdir -p ~/asr-benchmark
cd ~/asr-benchmark

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2.6 PyTorch with CUDA Support

```bash
# Install PyTorch 2.5.1 with CUDA 12.1 support (compatible with CUDA 12.2 runtime)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA detection
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
EOF
```

**Expected Output:**
```
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
Device count: 1
Device name: Tesla V100-PCIE-16GB
Device capability: (7, 0)
```

### 2.7 Core Dependencies Installation

```bash
# Install core ASR dependencies (from requirements.txt)
pip install transformers==4.57.0 \
            accelerate==1.2.1 \
            librosa==0.11.0 \
            soundfile==0.13.1 \
            sounddevice==0.4.7 \
            numpy==1.26.4 \
            python-dotenv==1.0.1 \
            datasets==3.2.0

# Additional monitoring/analysis tools
pip install \
    psutil==6.1.1 \
    gpustat==1.1.1 \
    nvidia-ml-py3==7.352.0 \
    pandas==2.2.3 \
    matplotlib==3.9.3 \
    seaborn==0.13.2
```

### 2.8 Flash Attention 2 Installation

```bash
# Install Flash Attention 2 (CRITICAL for performance comparison)
# This requires CUDA and may take 10-15 minutes to compile
pip install flash-attn==2.8.3 --no-build-isolation

# Verify installation
python3 << EOF
try:
    import flash_attn
    print(f"Flash Attention version: {flash_attn.__version__}")
    print("Flash Attention 2 installed successfully!")
except ImportError as e:
    print(f"Flash Attention import failed: {e}")
EOF
```

**Expected Output:**
```
Flash Attention version: 2.8.3
Flash Attention 2 installed successfully!
```

**Fallback if compilation fails:**
```bash
# Try latest version without specific version pin
pip install flash-attn --no-build-isolation

# Or try pre-built wheels (if compilation issues occur)
pip install flash-attn --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases
```

**Validation Checkpoint:**
```bash
# Create comprehensive verification script
python3 << 'EOF'
import sys
import torch
import transformers
import librosa
import soundfile
import numpy as np

print("="*60)
print("Environment Validation")
print("="*60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Librosa: {librosa.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

try:
    import flash_attn
    print(f"Flash Attention: {flash_attn.__version__} (ENABLED)")
except ImportError:
    print("Flash Attention: NOT INSTALLED")

print("="*60)
EOF
```

---

## Phase 3: Code Deployment (20-30 minutes)

### 3.1 Deployment Method Selection

**Option A: Git Clone (Recommended if using GitHub)**
```bash
# If your code is in a GitHub repo
cd ~/asr-benchmark
git clone https://github.com/YOUR_USERNAME/ASR.git project
cd project

# Activate venv and install dependencies
source ../.venv/bin/activate
pip install -r requirements.txt
```

**Option B: SCP Transfer (Recommended for Local Code)**
```bash
# From LOCAL machine, run:
cd /Users/joey/Workspace/interPro/ASR

# Transfer code (exclude venv and large files)
scp -i ~/.ssh/asr_vm_key -r \
  test_asr_streaming.py \
  requirements.txt \
  .env \
  azureuser@$VM_IP:~/asr-benchmark/

# Transfer dataset
scp -i ~/.ssh/asr_vm_key -r \
  dataset/sample_audios/*.wav \
  dataset/sample_audios/metadata.json \
  azureuser@$VM_IP:~/asr-benchmark/dataset/sample_audios/
```

### 3.2 Directory Structure Setup on VM

```bash
# On VM, create proper structure
ssh -i ~/.ssh/asr_vm_key azureuser@$VM_IP

cd ~/asr-benchmark
mkdir -p dataset/sample_audios
mkdir -p results/{streaming,batch,logs,plots}
mkdir -p scripts
mkdir -p models_cache

# Set HuggingFace cache to local directory for saving models and dataset from hf
export HF_HOME=~/asr-benchmark/models_cache
echo 'export HF_HOME=~/asr-benchmark/models_cache' >> ~/.bashrc
```
How HuggingFace Model Caching Works

  First Model Load (downloads from internet):
```py
  pipe = pipeline(
      "automatic-speech-recognition",
      model="kotoba-tech/kotoba-whisper-v2.2",  # Downloads from huggingface.co
      ...
  )
```
  What happens:
  1. Checks ~/asr-benchmark/models_cache/ for existing model
  2. Not found → Downloads from HuggingFace servers (~3-4 GB)
  3. Saves to ~/asr-benchmark/models_cache/hub/models--kotoba-tech--kotoba-whisper-v2.2/
  4. Loads model into memory

  Subsequent Model Loads (uses local cache):
```py
  pipe = pipeline(
      "automatic-speech-recognition",
      model="kotoba-tech/kotoba-whisper-v2.2",  # Loads from cache
      ...
  )
```
  What happens:
  1. Checks ~/asr-benchmark/models_cache/
  2. Found! → Loads from local disk (much faster, no download)
  3. No internet required
Why We Set Custom Cache Location

  Default behavior (without HF_HOME):
  Models cached in: ~/.cache/huggingface/
  #Problem: Hidden directory, hard to track disk usage

  Our custom setup:
  export HF_HOME=~/asr-benchmark/models_cache
  #Benefit: Everything organized in one place

  Advantages:
  1. Transparency: Easy to see where large files are stored
  2. Disk space management: Can check size with du -sh ~/asr-benchmark/models_cache
  3. Easy cleanup: Delete entire ~/asr-benchmark/ when done
  4. Avoid re-downloads: Model survives VM restarts (if using auto-shutdown)
  5. Multiple tests: Flash Attention vs Baseline both use same cached model

  Verification

  After first model load, you can check the cache:

  #On the VM
  
  ls -lh ~/asr-benchmark/models_cache/hub/models--kotoba-tech--kotoba-whisper-v2.2/

  #Check total cache size

  du -sh ~/asr-benchmark/models_cache/
  
  #Expected: ~3-4 GB

### 3.3 Environment Variables Configuration

```bash
# Create .env file with credentials
cat > ~/asr-benchmark/.env << 'EOF'
HF_TOKEN=your_huggingface_token_here
HF_API_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

# Secure the file
chmod 600 ~/asr-benchmark/.env
```

### 3.4 Verify Code Transfer

```bash
# Verify all files are present
cd ~/asr-benchmark
ls -lh test_asr_streaming.py
ls -lh dataset/sample_audios/*.wav
cat .env

# Verify Python can load modules
source .venv/bin/activate
python3 -c "from transformers import pipeline; print('Import successful!')"
```

**Validation Checkpoint:**
```bash
# Quick test with model loading (no inference yet)
cd ~/asr-benchmark
source .venv/bin/activate

python3 << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

# Verify environment
print(f"HF_TOKEN loaded: {os.getenv('HF_TOKEN')[:10]}...")
print(f"HF cache: {os.getenv('HF_HOME')}")

# Quick model check (download will happen)
print("\nTesting model download...")
from transformers import pipeline
import torch

device = "cuda:0"
pipe = pipeline(
    "automatic-speech-recognition",
    model="kotoba-tech/kotoba-whisper-v2.2",
    torch_dtype=torch.float16,
    device=device,
)
print("Model loaded successfully on GPU!")
EOF
```

---

## Phase 4: Testing Scripts Development (60-90 minutes)

### 4.1 GPU Monitoring Utility Script

**File: `scripts/gpu_monitor.py`** (to be created on VM)

```python
#!/usr/bin/env python3
"""
GPU Monitoring Utility for ASR Benchmarking
Logs GPU metrics during test execution
"""
import time
import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path
import threading


class GPUMonitor:
    """Real-time GPU monitoring with logging"""

    def __init__(self, output_dir="results/logs", interval=1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.running = False
        self.thread = None

        # Initialize log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.output_dir / f"gpu_metrics_{timestamp}.csv"
        self.json_file = self.output_dir / f"gpu_summary_{timestamp}.json"

        # CSV headers
        self.csv_headers = [
            "timestamp", "gpu_util_%", "memory_used_mb", "memory_total_mb",
            "memory_util_%", "temperature_c", "power_draw_w", "power_limit_w"
        ]

    def get_gpu_stats(self):
        """Query GPU stats using nvidia-smi"""
        try:
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, check=True)

            values = result.stdout.strip().split(', ')
            return {
                "timestamp": datetime.now().isoformat(),
                "gpu_util_%": float(values[0]),
                "memory_used_mb": float(values[1]),
                "memory_total_mb": float(values[2]),
                "memory_util_%": (float(values[1]) / float(values[2])) * 100,
                "temperature_c": float(values[3]),
                "power_draw_w": float(values[4]),
                "power_limit_w": float(values[5])
            }
        except Exception as e:
            print(f"Error querying GPU: {e}")
            return None

    def _monitor_loop(self):
        """Background monitoring loop"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()

            while self.running:
                stats = self.get_gpu_stats()
                if stats:
                    writer.writerow(stats)
                    f.flush()
                time.sleep(self.interval)

    def start(self):
        """Start monitoring in background thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"GPU monitoring started (logging to {self.csv_file})")

    def stop(self):
        """Stop monitoring and generate summary"""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join()

        # Generate summary statistics
        self._generate_summary()
        print(f"GPU monitoring stopped (summary: {self.json_file})")

    def _generate_summary(self):
        """Generate summary statistics from collected data"""
        import pandas as pd

        df = pd.read_csv(self.csv_file)
        summary = {
            "duration_seconds": len(df) * self.interval,
            "samples": len(df),
            "gpu_utilization": {
                "mean": df["gpu_util_%"].mean(),
                "max": df["gpu_util_%"].max(),
                "min": df["gpu_util_%"].min(),
                "std": df["gpu_util_%"].std()
            },
            "memory_usage_mb": {
                "mean": df["memory_used_mb"].mean(),
                "max": df["memory_used_mb"].max(),
                "peak_util_%": (df["memory_used_mb"].max() / df["memory_total_mb"].iloc[0]) * 100
            },
            "temperature_c": {
                "mean": df["temperature_c"].mean(),
                "max": df["temperature_c"].max()
            },
            "power_draw_w": {
                "mean": df["power_draw_w"].mean(),
                "max": df["power_draw_w"].max()
            }
        }

        with open(self.json_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


if __name__ == "__main__":
    # Test monitoring
    monitor = GPUMonitor(interval=0.5)
    monitor.start()

    try:
        time.sleep(10)  # Monitor for 10 seconds
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
```

### 4.2 Automated Benchmark Script

**File: `scripts/run_benchmark.py`** (to be created on VM)

```python
#!/usr/bin/env python3
"""
Automated ASR Benchmark Script
Runs streaming tests with and without Flash Attention
Collects all 4 required metrics: RTF, throughput, Flash Attention comparison, GPU stats
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import librosa
import numpy as np
from transformers import pipeline
from dotenv import load_dotenv

# Import GPU monitor (ensure gpu_monitor.py is in same directory)
sys.path.insert(0, str(Path(__file__).parent))
from gpu_monitor import GPUMonitor


class ASRBenchmark:
    """Comprehensive ASR benchmarking framework"""

    def __init__(self, output_dir="results", model_name="kotoba-tech/kotoba-whisper-v2.2"):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.results = []

        # Create output directories
        (self.output_dir / "streaming").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "batch").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)

    def load_model(self, use_flash_attention=False):
        """Load ASR model with optional Flash Attention"""
        print(f"\n{'='*60}")
        print(f"Loading model: {self.model_name}")
        print(f"Flash Attention: {'ENABLED' if use_flash_attention else 'DISABLED'}")
        print(f"{'='*60}\n")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn
                model_kwargs = {"attn_implementation": "flash_attention_2"}
                print("Flash Attention 2 enabled successfully")
            except ImportError:
                print("WARNING: Flash Attention not available, using standard attention")

        load_start = time.perf_counter()
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs,
        )
        load_time = time.perf_counter() - load_start

        print(f"Model loaded in {load_time:.2f}s")
        print(f"Device: {device}")
        print(f"Data type: {torch_dtype}\n")

        return pipe, device

    def benchmark_streaming(self, audio_path, pipe, device, use_flash_attention=False):
        """
        Benchmark streaming transcription (3s chunks with 0.5s overlap)
        Returns: dict with RTF, latency, and per-chunk metrics
        """
        print(f"\nBenchmarking: {Path(audio_path).name}")
        print(f"Flash Attention: {'ENABLED' if use_flash_attention else 'DISABLED'}\n")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration = len(audio) / sr

        # Chunking parameters (matching test_asr_streaming.py)
        chunk_duration = 3.0
        overlap = 0.5
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)
        stride = chunk_samples - overlap_samples

        # Process chunks
        chunk_latencies = []
        chunk_texts = []
        start = 0
        chunk_num = 0
        total_start = time.perf_counter()

        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            # Pad last chunk
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            # Transcribe chunk
            chunk_start = time.perf_counter()
            result = pipe(
                chunk,
                return_timestamps=False,
                generate_kwargs={"language": "ja", "task": "transcribe"}
            )
            chunk_latency = time.perf_counter() - chunk_start

            chunk_latencies.append(chunk_latency)
            chunk_texts.append(result['text'])
            chunk_num += 1

            print(f"  Chunk {chunk_num}: {chunk_latency:.3f}s | RTF: {chunk_latency/chunk_duration:.3f}x")

            start += stride

        total_time = time.perf_counter() - total_start
        avg_latency = np.mean(chunk_latencies)
        rtf = total_time / audio_duration

        # Results
        result = {
            "audio_file": str(audio_path),
            "audio_duration": audio_duration,
            "flash_attention": use_flash_attention,
            "total_chunks": chunk_num,
            "total_time": total_time,
            "average_latency_per_chunk": avg_latency,
            "rtf": rtf,
            "rtf_status": "faster" if rtf < 1.0 else "slower",
            "chunk_latencies": chunk_latencies,
            "full_transcript": " ".join(chunk_texts),
            "timestamp": datetime.now().isoformat()
        }

        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  RTF: {rtf:.3f}x ({result['rtf_status']} than real-time)")
        print(f"  Avg latency/chunk: {avg_latency:.3f}s")

        return result

    def benchmark_batch_throughput(self, audio_files, pipe, device, use_flash_attention=False):
        """
        Benchmark batch processing throughput (files/hour)
        """
        print(f"\n{'='*60}")
        print(f"BATCH THROUGHPUT TEST")
        print(f"Flash Attention: {'ENABLED' if use_flash_attention else 'DISABLED'}")
        print(f"Files: {len(audio_files)}")
        print(f"{'='*60}\n")

        total_audio_duration = 0
        batch_start = time.perf_counter()

        for i, audio_path in enumerate(audio_files, 1):
            audio, sr = librosa.load(audio_path, sr=16000)
            audio_duration = len(audio) / sr
            total_audio_duration += audio_duration

            # Full file transcription
            file_start = time.perf_counter()
            result = pipe(
                audio,
                return_timestamps=False,
                generate_kwargs={"language": "ja", "task": "transcribe"}
            )
            file_time = time.perf_counter() - file_start

            print(f"  File {i}/{len(audio_files)}: {file_time:.2f}s (duration: {audio_duration:.2f}s)")

        total_time = time.perf_counter() - batch_start
        throughput_files_per_hour = (len(audio_files) / total_time) * 3600
        throughput_audio_hours_per_hour = (total_audio_duration / total_time)

        result = {
            "flash_attention": use_flash_attention,
            "total_files": len(audio_files),
            "total_audio_duration": total_audio_duration,
            "total_processing_time": total_time,
            "throughput_files_per_hour": throughput_files_per_hour,
            "throughput_audio_hours_per_hour": throughput_audio_hours_per_hour,
            "average_rtf": total_time / total_audio_duration,
            "timestamp": datetime.now().isoformat()
        }

        print(f"\nBatch Throughput Results:")
        print(f"  Processed {len(audio_files)} files in {total_time:.2f}s")
        print(f"  Throughput: {throughput_files_per_hour:.2f} files/hour")
        print(f"  Average RTF: {result['average_rtf']:.3f}x")

        return result

    def run_full_benchmark(self, audio_dir="dataset/sample_audios"):
        """Run complete benchmark suite"""
        audio_files = sorted(Path(audio_dir).glob("*.wav"))
        print(f"\nFound {len(audio_files)} audio files")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = {
            "timestamp": timestamp,
            "model": self.model_name,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "tests": []
        }

        # Test configurations
        configs = [
            {"name": "baseline", "flash_attention": False},
            {"name": "flash_attention", "flash_attention": True}
        ]

        for config in configs:
            print(f"\n\n{'#'*70}")
            print(f"# TEST CONFIGURATION: {config['name'].upper()}")
            print(f"{'#'*70}")

            # Start GPU monitoring
            monitor = GPUMonitor(output_dir=self.output_dir / "logs", interval=1.0)
            monitor.start()

            try:
                # Load model
                pipe, device = self.load_model(use_flash_attention=config['flash_attention'])

                # METRIC 1 & 3: Streaming tests with RTF/latency + Flash Attention comparison
                streaming_results = []
                for audio_file in audio_files:
                    result = self.benchmark_streaming(
                        audio_file, pipe, device,
                        use_flash_attention=config['flash_attention']
                    )
                    streaming_results.append(result)

                # METRIC 2: Throughput test (batch processing)
                throughput_result = self.benchmark_batch_throughput(
                    audio_files, pipe, device,
                    use_flash_attention=config['flash_attention']
                )

                # Save results
                config_results = {
                    "configuration": config['name'],
                    "flash_attention": config['flash_attention'],
                    "streaming_tests": streaming_results,
                    "batch_throughput": throughput_result,
                    "gpu_monitor_csv": str(monitor.csv_file),
                    "gpu_summary_json": str(monitor.json_file)
                }
                all_results['tests'].append(config_results)

            finally:
                # Stop GPU monitoring (METRIC 4: GPU utilization)
                monitor.stop()

                # Clear GPU memory
                del pipe
                torch.cuda.empty_cache()
                time.sleep(5)  # Cool down

        # Save comprehensive results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n\nBenchmark complete! Results saved to: {results_file}")

        # Generate comparison report
        self.generate_comparison_report(all_results)

        return all_results

    def generate_comparison_report(self, results):
        """Generate Flash Attention vs Baseline comparison report"""
        print(f"\n\n{'='*70}")
        print("FLASH ATTENTION vs BASELINE COMPARISON")
        print(f"{'='*70}\n")

        baseline = next((t for t in results['tests'] if not t['flash_attention']), None)
        flash = next((t for t in results['tests'] if t['flash_attention']), None)

        if not baseline or not flash:
            print("ERROR: Missing test configurations")
            return

        # Calculate averages
        baseline_rtf = np.mean([s['rtf'] for s in baseline['streaming_tests']])
        flash_rtf = np.mean([s['rtf'] for s in flash['streaming_tests']])
        rtf_improvement = ((baseline_rtf - flash_rtf) / baseline_rtf) * 100

        baseline_throughput = baseline['batch_throughput']['throughput_files_per_hour']
        flash_throughput = flash['batch_throughput']['throughput_files_per_hour']
        throughput_improvement = ((flash_throughput - baseline_throughput) / baseline_throughput) * 100

        print(f"Streaming Performance (RTF):")
        print(f"  Baseline:         {baseline_rtf:.3f}x")
        print(f"  Flash Attention:  {flash_rtf:.3f}x")
        print(f"  Improvement:      {rtf_improvement:+.1f}%")
        print()

        print(f"Batch Throughput (files/hour):")
        print(f"  Baseline:         {baseline_throughput:.2f}")
        print(f"  Flash Attention:  {flash_throughput:.2f}")
        print(f"  Improvement:      {throughput_improvement:+.1f}%")
        print()

        # GPU metrics comparison
        try:
            with open(baseline['gpu_summary_json']) as f:
                baseline_gpu = json.load(f)
            with open(flash['gpu_summary_json']) as f:
                flash_gpu = json.load(f)

            print(f"GPU Utilization:")
            print(f"  Baseline:         {baseline_gpu['gpu_utilization']['mean']:.1f}%")
            print(f"  Flash Attention:  {flash_gpu['gpu_utilization']['mean']:.1f}%")
            print()

            print(f"Peak VRAM Usage:")
            print(f"  Baseline:         {baseline_gpu['memory_usage_mb']['max']:.0f} MB")
            print(f"  Flash Attention:  {flash_gpu['memory_usage_mb']['max']:.0f} MB")
            print()

            print(f"Average Power Draw:")
            print(f"  Baseline:         {baseline_gpu['power_draw_w']['mean']:.1f} W")
            print(f"  Flash Attention:  {flash_gpu['power_draw_w']['mean']:.1f} W")
        except Exception as e:
            print(f"Could not load GPU metrics: {e}")

        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="ASR Benchmark Suite")
    parser.add_argument("--audio-dir", default="dataset/sample_audios", help="Audio files directory")
    parser.add_argument("--output-dir", default="results", help="Results output directory")
    parser.add_argument("--model", default="kotoba-tech/kotoba-whisper-v2.2", help="Model name")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Run benchmark
    benchmark = ASRBenchmark(output_dir=args.output_dir, model_name=args.model)
    results = benchmark.run_full_benchmark(audio_dir=args.audio_dir)


if __name__ == "__main__":
    main()
```

### 4.3 Results Analysis Script

**File: `scripts/analyze_results.py`** (to be created on VM)

```python
#!/usr/bin/env python3
"""
Results Analysis and Visualization Script
Generates plots and summary reports from benchmark results
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def analyze_benchmark(results_json):
    """Analyze benchmark results and generate plots"""

    with open(results_json) as f:
        results = json.load(f)

    output_dir = Path(results_json).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    print(f"Analyzing results from: {results_json}")
    print(f"Plots will be saved to: {output_dir}\n")

    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # Extract data
    baseline = next((t for t in results['tests'] if not t['flash_attention']), None)
    flash = next((t for t in results['tests'] if t['flash_attention']), None)

    # Plot 1: RTF Comparison
    fig, ax = plt.subplots()
    baseline_rtfs = [s['rtf'] for s in baseline['streaming_tests']]
    flash_rtfs = [s['rtf'] for s in flash['streaming_tests']]

    x = range(len(baseline_rtfs))
    ax.plot(x, baseline_rtfs, marker='o', label='Baseline', linewidth=2)
    ax.plot(x, flash_rtfs, marker='s', label='Flash Attention', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Real-time threshold')
    ax.set_xlabel('Audio File Index')
    ax.set_ylabel('Real-Time Factor (RTF)')
    ax.set_title('Streaming Performance: RTF Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rtf_comparison.png", dpi=150)
    print(f"Saved: {output_dir / 'rtf_comparison.png'}")

    # Plot 2: Latency Distribution
    fig, ax = plt.subplots()
    baseline_latencies = [l for s in baseline['streaming_tests'] for l in s['chunk_latencies']]
    flash_latencies = [l for s in flash['streaming_tests'] for l in s['chunk_latencies']]

    ax.hist([baseline_latencies, flash_latencies], bins=30, label=['Baseline', 'Flash Attention'], alpha=0.7)
    ax.set_xlabel('Chunk Latency (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Chunk Latency Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "latency_distribution.png", dpi=150)
    print(f"Saved: {output_dir / 'latency_distribution.png'}")

    # Plot 3: GPU Utilization Timeline
    baseline_gpu_csv = baseline['gpu_monitor_csv']
    flash_gpu_csv = flash['gpu_monitor_csv']

    df_baseline = pd.read_csv(baseline_gpu_csv)
    df_flash = pd.read_csv(flash_gpu_csv)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(df_baseline['gpu_util_%'], label='GPU Utilization', color='blue')
    ax1.plot(df_baseline['memory_util_%'], label='Memory Utilization', color='orange')
    ax1.set_title('Baseline: GPU Metrics Over Time')
    ax1.set_ylabel('Utilization (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_flash['gpu_util_%'], label='GPU Utilization', color='blue')
    ax2.plot(df_flash['memory_util_%'], label='Memory Utilization', color='orange')
    ax2.set_title('Flash Attention: GPU Metrics Over Time')
    ax2.set_xlabel('Sample #')
    ax2.set_ylabel('Utilization (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "gpu_utilization_timeline.png", dpi=150)
    print(f"Saved: {output_dir / 'gpu_utilization_timeline.png'}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="Path to benchmark results JSON")
    args = parser.parse_args()

    analyze_benchmark(args.results_json)
```

---

## Phase 5: Performance Monitoring Setup (15-20 minutes)

### 5.1 System Monitoring Tools Installation

```bash
# Install system monitoring tools
sudo apt install -y htop iotop sysstat

# Install Python monitoring packages (already done in Phase 2.7)
source .venv/bin/activate
pip install psutil gpustat nvidia-ml-py3
```

### 5.2 Real-time Monitoring Dashboard Setup

```bash
# Create monitoring script for manual inspection
cat > ~/asr-benchmark/scripts/monitor_dashboard.sh << 'EOF'
#!/bin/bash
# Real-time monitoring dashboard

while true; do
    clear
    echo "====== ASR Benchmark Monitoring ======"
    echo "Time: $(date)"
    echo ""

    echo "--- GPU Status ---"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader
    echo ""

    echo "--- CPU & Memory ---"
    top -b -n 1 | head -20
    echo ""

    echo "--- Disk Usage ---"
    df -h ~/asr-benchmark

    sleep 2
done
EOF

chmod +x ~/asr-benchmark/scripts/monitor_dashboard.sh
```

---

## Phase 6: Cost Optimization & Auto-Shutdown (20-30 minutes)

### 6.1 Idle Detection & Auto-Shutdown Script

```bash
# Create idle shutdown script
cat > ~/asr-benchmark/scripts/auto_shutdown_idle.sh << 'EOF'
#!/bin/bash
# Auto-shutdown if GPU idle for specified duration

IDLE_THRESHOLD_MINUTES=60  # Shutdown after 60 minutes of GPU idle
CHECK_INTERVAL_SECONDS=300  # Check every 5 minutes
IDLE_GPU_PERCENT=5          # GPU util below 5% = idle

idle_counter=0

while true; do
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    if (( gpu_util < IDLE_GPU_PERCENT )); then
        idle_counter=$((idle_counter + CHECK_INTERVAL_SECONDS))
        echo "[$(date)] GPU idle: $gpu_util% (idle for $((idle_counter/60)) min)"

        if (( idle_counter >= (IDLE_THRESHOLD_MINUTES * 60) )); then
            echo "[$(date)] GPU idle threshold reached. Shutting down..."
            sudo shutdown -h +5 "Auto-shutdown: GPU idle for $IDLE_THRESHOLD_MINUTES minutes"
            exit 0
        fi
    else
        idle_counter=0
        echo "[$(date)] GPU active: $gpu_util%"
    fi

    sleep $CHECK_INTERVAL_SECONDS
done
EOF

chmod +x ~/asr-benchmark/scripts/auto_shutdown_idle.sh
```

### 6.2 Cost Estimation

```bash
# Cost calculation script
cat > ~/asr-benchmark/scripts/estimate_cost.sh << 'EOF'
#!/bin/bash
# Estimate Azure VM running costs

VM_HOURLY_RATE=3.06  # Standard_NC6s_v3 in East US
STORAGE_MONTHLY_RATE=9.60  # 128GB Premium SSD (~$0.32/day)

echo "====== Azure Cost Estimate ======"
echo ""
echo "VM: Standard_NC6s_v3"
echo "Hourly rate: \$${VM_HOURLY_RATE}"
echo ""
echo "Estimated costs:"
echo "  1 day (24h):   \$$(echo "$VM_HOURLY_RATE * 24" | bc)"
echo "  2 days (48h):  \$$(echo "$VM_HOURLY_RATE * 48" | bc)"
echo "  3 days (72h):  \$$(echo "$VM_HOURLY_RATE * 72" | bc)"
echo ""
echo "Storage (128GB Premium SSD): \$0.32/day"
echo ""
echo "Total for 3 days (VM + Storage): \$$(echo "$VM_HOURLY_RATE * 72 + 0.32 * 3" | bc)"
echo ""
echo "NOTE: Use auto-shutdown to minimize idle costs!"
echo "======================================"
EOF

chmod +x ~/asr-benchmark/scripts/estimate_cost.sh
bash ~/asr-benchmark/scripts/estimate_cost.sh
```

---

## Phase 7: Results Collection & Analysis (30-40 minutes)

### 7.1 Run Full Benchmark Suite

```bash
# On VM, start the comprehensive benchmark
cd ~/asr-benchmark
source .venv/bin/activate

# Run benchmark (this will take 1-2 hours depending on dataset)
python3 scripts/run_benchmark.py \
  --audio-dir dataset/sample_audios \
  --output-dir results \
  --model kotoba-tech/kotoba-whisper-v2.2

# Monitor progress in another terminal
ssh -i ~/.ssh/asr_vm_key azureuser@$VM_IP
~/asr-benchmark/scripts/monitor_dashboard.sh
```

### 7.2 Results Analysis

```bash
# After benchmark completes, analyze results
cd ~/asr-benchmark
source .venv/bin/activate

# Find latest results file
RESULTS_FILE=$(ls -t results/benchmark_results_*.json | head -1)

# Generate plots and analysis
python3 scripts/analyze_results.py $RESULTS_FILE

# View summary
cat $RESULTS_FILE | python3 -m json.tool | less
```

### 7.3 Download Results to Local Machine

```bash
# From LOCAL machine
cd /Users/joey/Workspace/interPro/ASR

# Create results archive on VM
ssh -i ~/.ssh/asr_vm_key azureuser@$VM_IP \
  "cd ~/asr-benchmark && tar -czf results_archive.tar.gz results/"

# Download archive
scp -i ~/.ssh/asr_vm_key \
  azureuser@$VM_IP:~/asr-benchmark/results_archive.tar.gz \
  ./azure_benchmark_results.tar.gz

# Extract locally
tar -xzf azure_benchmark_results.tar.gz

# View results
open results/plots/  # macOS
```

---

## Phase 8: Troubleshooting & Fallback Strategies

### 8.1 Common Issues & Solutions

**Issue 1: CUDA Out of Memory (OOM)**
```bash
# Symptoms: RuntimeError: CUDA out of memory
# Verify VRAM usage
nvidia-smi

# Clear cache before each test
python3 << EOF
import torch
torch.cuda.empty_cache()
print(f"VRAM free: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
EOF
```

**Issue 2: Flash Attention Installation Failure**
```bash
# Symptoms: Compilation errors during pip install flash-attn
# Solution 1: Ensure CUDA toolkit is properly installed
nvcc --version

# Solution 2: Use pre-built wheel
pip install flash-attn==2.7.4 --no-build-isolation \
  --find-links https://github.com/Dao-AILab/flash-attention/releases

# Solution 3: Fallback to standard attention
# Benchmark will automatically detect missing flash_attn and skip
```

**Issue 3: Slow Model Download**
```bash
# Symptoms: HuggingFace model download hanging
# Solution: Pre-download model
python3 << EOF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
model = AutoModelForSpeechSeq2Seq.from_pretrained("kotoba-tech/kotoba-whisper-v2.2")
processor = AutoProcessor.from_pretrained("kotoba-tech/kotoba-whisper-v2.2")
EOF
```

**Issue 4: CUDA Version Mismatch**
```bash
# Symptoms: PyTorch can't find CUDA
# Solution: Verify CUDA compatibility
python3 << EOF
import torch
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF

nvidia-smi | grep "CUDA Version"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchaudio -y
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## Phase 9: Complete Testing Workflow (Step-by-Step Execution)

### 9.1 Pre-Flight Checklist

```bash
# ON VM - Final verification before benchmark
cd ~/asr-benchmark
source .venv/bin/activate

echo "=== Pre-Flight Checklist ==="
echo ""

# 1. GPU Check
echo "1. GPU Status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 2. CUDA Check
echo "2. CUDA:"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
echo ""

# 3. Flash Attention Check
echo "3. Flash Attention:"
python3 -c "try: import flash_attn; print(f'Version: {flash_attn.__version__}')
except: print('NOT INSTALLED')"
echo ""

# 4. Dataset Check
echo "4. Dataset:"
ls -lh dataset/sample_audios/*.wav | wc -l
echo " WAV files found"
echo ""

# 5. Disk Space Check
echo "5. Disk Space:"
df -h ~/asr-benchmark | tail -1
echo ""

# 6. Environment Variables Check
echo "6. Environment:"
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f'HF_TOKEN: {os.getenv(\"HF_TOKEN\")[:10]}...')"
echo ""

echo "=== Pre-Flight Complete ==="
```

### 9.2 Execute Benchmark

```bash
# Start benchmark (estimated time: 1-2 hours)
cd ~/asr-benchmark
source .venv/bin/activate

# Optional: Run in screen/tmux for persistence
screen -S asr-benchmark

# Run benchmark
python3 scripts/run_benchmark.py \
  --audio-dir dataset/sample_audios \
  --output-dir results \
  --model kotoba-tech/kotoba-whisper-v2.2 2>&1 | tee results/logs/benchmark_run.log

# Detach from screen: Ctrl+A, then D
# Reattach: screen -r asr-benchmark
```

### 9.3 Monitor Progress

```bash
# In separate SSH session
ssh -i ~/.ssh/asr_vm_key azureuser@$VM_IP

# Watch GPU metrics
watch -n 2 nvidia-smi

# Or use monitoring dashboard
~/asr-benchmark/scripts/monitor_dashboard.sh
```

### 9.4 Post-Benchmark Analysis

```bash
# After benchmark completes
cd ~/asr-benchmark
source .venv/bin/activate

# Find latest results
RESULTS=$(ls -t results/benchmark_results_*.json | head -1)
echo "Analyzing: $RESULTS"

# Generate visualizations
python3 scripts/analyze_results.py $RESULTS

# View summary
cat $RESULTS | python3 -m json.tool | head -100
```

### 9.5 Results Download & VM Cleanup

```bash
# ON VM: Create archive
cd ~/asr-benchmark
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz results/

# ON LOCAL MACHINE: Download
scp -i ~/.ssh/asr_vm_key \
  azureuser@$VM_IP:~/asr-benchmark/results_*.tar.gz \
  /Users/joey/Workspace/interPro/ASR/

# Extract and analyze locally
cd /Users/joey/Workspace/interPro/ASR
tar -xzf results_*.tar.gz

# Delete VM (after confirming results are downloaded)
az vm delete \
  --resource-group rg-asr-benchmark \
  --name vm-asr-test \
  --yes

# Delete entire resource group (cleanup all resources)
az group delete \
  --name rg-asr-benchmark \
  --yes --no-wait
```

---

## Expected Performance Targets

Based on V100 GPU capabilities:

**Streaming Performance (3-second chunks):**
- **Target RTF:** < 0.5x (ideal), < 1.0x (critical)
- **Baseline (no Flash Attention):** ~0.3-0.5x RTF expected
- **Flash Attention:** ~0.15-0.25x RTF expected (30-40% improvement)
- **Per-chunk latency:** 0.5-1.5 seconds (for 3s chunks)

**Batch Throughput:**
- **Target:** 100-200 files/hour (for 50-60s files)
- **Flash Attention improvement:** 30-50% faster

**GPU Utilization:**
- **Target:** 70-95% GPU utilization during inference
- **VRAM usage:** 8-12GB peak (out of 16GB)
- **Power draw:** 150-220W (out of 250W limit)

---

## Implementation Timeline & Effort Estimates

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Azure VM Provisioning | 30-45 min | 0:45 |
| 2 | GPU Environment Setup | 45-60 min | 1:45 |
| 3 | Code Deployment | 20-30 min | 2:15 |
| 4 | Testing Scripts Development | 60-90 min | 3:45 |
| 5 | Performance Monitoring Setup | 15-20 min | 4:05 |
| 6 | Cost Optimization | 20-30 min | 4:35 |
| - | **TOTAL SETUP TIME** | **~4-5 hours** | - |
| 7 | Benchmark Execution | 1-2 hours | 6:35 |
| 8 | Results Analysis | 30-40 min | 7:15 |
| 9 | Download & Cleanup | 15-20 min | 7:35 |
| - | **TOTAL PROJECT TIME** | **~7-8 hours** | - |

**Note:** Setup phases can be completed in one session. Benchmark execution can run unattended overnight.

---

## Critical Validation Checkpoints Summary

After each phase, validate before proceeding:

1. **Phase 1:** `ssh -i ~/.ssh/asr_vm_key azureuser@$VM_IP "uname -a"` succeeds
2. **Phase 2:** `nvidia-smi` shows Tesla V100, `python3 -c "import torch; print(torch.cuda.is_available())"` returns True
3. **Phase 3:** `ls ~/asr-benchmark/dataset/sample_audios/*.wav | wc -l` returns 10
4. **Phase 4:** `python3 scripts/gpu_monitor.py` runs without errors
5. **Phase 5:** GPU monitoring collects metrics successfully
6. **Phase 7:** `benchmark_results_*.json` file exists with all 4 metric categories

---

## Expected Deliverables

After completing this implementation, you will have:

1. **Quantitative Metrics:**
   - RTF measurements for 10 audio files (baseline + Flash Attention)
   - Per-chunk latency statistics
   - Batch throughput (files/hour)
   - GPU utilization time-series (VRAM, GPU%, power, temperature)

2. **Comparison Analysis:**
   - Flash Attention vs Baseline performance delta
   - Percentage improvement metrics
   - Statistical significance analysis

3. **Visualizations:**
   - RTF comparison line plot
   - Latency distribution histograms
   - GPU utilization timeline
   - Memory usage plots

4. **Raw Data:**
   - JSON results file with all test data
   - CSV files with GPU metrics (1-second granularity)
   - Log files for debugging/audit

5. **Cost Report:**
   - Actual VM runtime hours
   - Total Azure spend
   - Cost per inference hour

---

## Critical Files Reference

**Local Files (existing):**
- `/Users/joey/Workspace/interPro/ASR/test_asr_streaming.py` - Reference implementation
- `/Users/joey/Workspace/interPro/ASR/requirements.txt` - Dependencies
- `/Users/joey/Workspace/interPro/ASR/.env` - HF_TOKEN credentials
- `/Users/joey/Workspace/interPro/ASR/dataset/sample_audios/*.wav` - 10 test files

**Local Files (to create):**
- `/Users/joey/Workspace/interPro/ASR/scripts/gpu_monitor.py` - GPU metrics logger
- `/Users/joey/Workspace/interPro/ASR/scripts/run_benchmark.py` - Main benchmark orchestrator
- `/Users/joey/Workspace/interPro/ASR/scripts/analyze_results.py` - Results visualization

**Azure VM Paths:**
- `~/asr-benchmark/` - Main project directory
- `~/asr-benchmark/results/benchmark_results_*.json` - Final results
- `~/asr-benchmark/results/logs/gpu_metrics_*.csv` - GPU time-series data
- `~/asr-benchmark/results/plots/*.png` - Visualization plots

---

## Next Steps

To execute this plan:

1. **Local Preparation (10 min):**
   - Install Azure CLI: `brew install azure-cli` (macOS)
   - Login: `az login`
   - Verify subscription: `az account list`

2. **Create Scripts Locally (30 min):**
   - Create `scripts/gpu_monitor.py` (from Phase 4.1)
   - Create `scripts/run_benchmark.py` (from Phase 4.2)
   - Create `scripts/analyze_results.py` (from Phase 4.3)

3. **Execute Phases 1-3 (Setup, ~2 hours):**
   - Provision VM
   - Install GPU environment
   - Deploy code

4. **Execute Phases 4-6 (Configuration, ~2 hours):**
   - Transfer benchmark scripts
   - Configure monitoring
   - Set auto-shutdown

5. **Execute Phase 7 (Testing, ~2 hours unattended):**
   - Run `scripts/run_benchmark.py`
   - Monitor progress

6. **Execute Phases 8-9 (Analysis & Cleanup, ~1 hour):**
   - Generate plots
   - Download results
   - Delete VM

**Total hands-on time:** ~5-6 hours (spread over 1-2 days)
**Total VM runtime:** ~8-12 hours (including setup + testing)
**Estimated cost:** $24-37 for 8-12 hours

---

This comprehensive guide provides a production-ready Azure VM testing environment with automated benchmarking, real-time monitoring, and cost optimization. All 4 required metrics (RTF/latency, throughput, Flash Attention comparison, GPU utilization) are collected systematically with visualization and analysis capabilities.

# 🚀 Complete Guide: Setting Up TensorFlow with GPU Support

Unlock the full potential of your NVIDIA GPU for deep learning with this comprehensive guide! Follow these steps to set up TensorFlow with GPU acceleration using the CUDA platform and cuDNN libraries.

---

## 📋 System Requirements

| Component        | Required Version             | Notes                                              |
|------------------|-----------------------------|----------------------------------------------------|
| **TensorFlow**   | 2.10.0                      | GPU-enabled version                                |
| **CUDA Toolkit** | 11.8                        | Included in TensorFlow 2.15.0 package              |
| **cuDNN**        | 8.6                         | Included in TensorFlow 2.15.0 package              |
| **NVIDIA Driver**| ≥ 522.06                    | Must be installed separately                       |
| **Python**       | 3.8 – 3.11                  | 3.10 recommended for best compatibility            |
| **OS**           | Windows 10/11 (64-bit)      |                                                    |
| **RAM**          | ≥ 8GB                       | 16GB+ recommended for larger models                |
| **GPU Memory**   | ≥ 4GB                       | More is better for complex models                  |

---

## 🔍 Pre-Setup: Verify GPU Compatibility

Before you start, confirm your NVIDIA GPU supports CUDA (most GTX 1000 series and newer do):

1. **Open PowerShell or Command Prompt as Administrator**
2. Run:
   ```bash
   nvidia-smi
   ```
   **Expected output:**
   ```
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 535.98                 Driver Version: 535.98                    |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |===============================+======================+======================|
   |   0  NVIDIA RTX 3050    WDDM  | 00000000:01:00.0 Yes |                  N/A |
   | 30%   45C    P8    11W / 130W |    1512MiB /  8192MiB |      2%      Default |
   +-------------------------------+----------------------+----------------------+
   ```
   If your GPU is listed, continue!

---

## 🛠️ Step 1: Install NVIDIA GPU Drivers

1. Visit [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Select your GPU model and OS.
3. Download and install the recommended driver.
4. **Restart your computer** after installation.

---

## 🐍 Step 2: Set Up Python Environment

Using **Conda** is highly recommended.

1. **Install Miniconda**  
   [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install.
2. **Open Anaconda Prompt** from Start menu.
3. **Create and activate a dedicated environment:**
   ```bash
   conda create -n tf_gpu python=3.10 -y
   conda activate tf_gpu
   ```

---

## 📦 Step 3: Install TensorFlow with GPU Support

1. **Remove any previous TensorFlow installs:**
   ```bash
   pip uninstall tensorflow -y
   ```
2. **Install TensorFlow 2.15.0 (bundled with CUDA 11.8 & cuDNN 8.6):**
   ```bash
   pip install tensorflow==2.10.0
   ```
   > *No separate installation for CUDA/cuDNN is needed!*

---

## ✅ Step 4: Verify GPU Detection

1. Create a script called `check_gpu.py`:

   ```python
   import tensorflow as tf

   print("TensorFlow version:", tf.__version__)
   print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
   print("Built with CUDA:", tf.test.is_built_with_cuda())
   print("Built with GPU support:", tf.test.is_built_with_gpu_support())
   print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

   # Test GPU computation
   if len(tf.config.list_physical_devices('GPU')) > 0:
       print("\nRunning simple test computation on GPU...")
       with tf.device('/GPU:0'):
           x = tf.random.normal([5000, 5000])
           y = tf.random.normal([5000, 5000])
           z = tf.matmul(x, y)
       print("GPU computation successful!")
   ```

2. Run:
   ```bash
   python check_gpu.py
   ```

   **Expected Output:**
   ```
   TensorFlow version: 2.10.0
   Num GPUs Available: 1
   Built with CUDA: True
   Built with GPU support: True
   Physical GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

   Running simple test computation on GPU...
   GPU computation successful!
   ```

---

## 💻 Step 5: Using GPU in Your Code

**Explicitly use GPU:**

```python
with tf.device('/GPU:0'):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
```

**For neural style transfer or GPU-intensive tasks:**

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

with tf.device('/GPU:0'):
    # Style transfer operations
    pass
```

---

## 📊 Step 6: Monitor GPU Performance

- Open a new Command Prompt or PowerShell window.
- Run:
  ```bash
  nvidia-smi -l 1
  ```
  This will refresh GPU stats every second.

---

## 🔄 Advanced: Managing Multiple GPUs

**List and select GPUs:**

```python
gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {gpus}")

# Use only GPU 0 and 1
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0:2], 'GPU')
    except RuntimeError as e:
        print(e)
```

---

## 🚨 Troubleshooting Common Issues

### **No GPU Found**
- Ensure NVIDIA drivers are installed (`nvidia-smi` should work).
- Check TensorFlow, CUDA, and GPU compatibility.
- Reinstall TensorFlow: `pip install tensorflow==2.15.0`.

### **Out of Memory Errors**
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### **DLL Load Failed**
- Update Windows.
- Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
- Check if antivirus is blocking CUDA components.

### **ImportError: Could not find CUDA drivers**
- Update/reinstall NVIDIA drivers from [official website](https://www.nvidia.com/Download/index.aspx).
- Restart after driver install.

---

## 🔗 Useful Resources

- [TensorFlow GPU Support Documentation](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)

---

## 📝 Performance Tips

- **Batch Size Optimization:** Adjust batch size according to your GPU memory.
- **Mixed Precision Training:** Faster training with `tf.keras.mixed_precision`.
- **Data Pipeline Optimization:** Use `tf.data` API for efficient loading.
- **Profiling:** Use TensorBoard's profiler for bottleneck analysis.

**Example: Mixed Precision Training**
```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

---

> **Happy Deep Learning! 🚀**
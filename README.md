# data_storage_sys_final_project
## Environment Setup (Python 3.10)
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## Usage
### 1. Pipelayer "Simulation"
```bash
python3.10 main.py
```

You can modify estimation parameters in `pipelayer.py`
Key parameters:
- `PER_SPIKE_BIT`
- `time_get_numel(x)`

### 2. MNIST GPU Baseline
```bash
python3.10 mnist_baseline_train.py
python3.10 mnist_baseline_test.py
```
type `A`, `B`, `C` to choice model

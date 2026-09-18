# Windows execution notes

## Reference environment

- Windows 10
- Python 3.10
- PyTorch 2.6.0+cu126
- torchvision 0.21.0
- CUDA Runtime 12.6
- NVIDIA RTX 3080 10GB

Install the CUDA 12.6 PyTorch wheels from the official PyTorch index:

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-windows.txt
```

Run scripts from the repository root. Training scripts use the first available CUDA device through
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`. DataLoader workers default to `0`
for Windows stability and can be changed with `--num-workers`.

## Execution order

1. Prepare/download STL10.
2. Run `googlenet_train.py` to create `google_net.pkl`.
3. Construct the required codebook:
   - `CB_construction.py` creates the task-unaware codebook.
   - `SC_construction.py` creates the semantic-aware codebook and requires `google_net.pkl`.
   - `LSC_construction.py` creates the label-based codebook and requires `google_net.pkl`.
4. Train the semantic communication model matching the selected codebook:
   - `trainCBFading.py`
   - `trainSCFading.py`
   - `trainLSC_LossFading.py`
   - `trainSC_LossFading.py`
5. Evaluate the selected trained checkpoint.

Dependency summary:

```text
STL10
  -> googlenet_train.py
  -> google_net.pkl
  -> CB / SC / LSC codebook
  -> semantic communication model
  -> evaluation
```

The public repository does not include `google_net.pkl`, generated codebook `.npy` files, or trained
`.model` checkpoints. Full training is intentionally not part of the compatibility validation.

## Evaluation correctness fix

The original `googlenet_train.py` did not specify the STL10 split for either loader. Because
`torchvision.datasets.STL10` defaults to `split="train"`, both training and evaluation used the
5,000-image training split. The loaders now explicitly use `split="train"` for training and
`split="test"` for validation/test, which uses the official 8,000-image STL10 test split.

This is an evaluation correctness fix, not a Windows compatibility fix. Dataset preprocessing is
unchanged: resize to 96 x 96, convert to float32 and scale to [0, 1], then transpose HWC to CHW,
with no augmentation or normalization.

## Phase 2 scope

This compatibility pass changes device selection, Windows-safe paths, DataLoader worker setup,
PyTorch 2.6 loading behavior, and output-directory creation only. It does not change the paper's
model dimensions, codebook assignment, proposed loss, optimizer schedule, SNR policy, or channel
equations.

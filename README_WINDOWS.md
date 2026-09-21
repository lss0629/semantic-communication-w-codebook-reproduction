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

The semantic communication loaders in `data/datasets.py` also explicitly use the official STL10
splits: `split="train"` for the 5,000-image training set and `split="test"` for the 8,000-image
test set. Their existing preprocessing is unchanged: resize to 256 x 256, convert to float32 and
scale to [0, 1], then transpose HWC to CHW, with no augmentation or normalization. This is an
evaluation correctness fix and does not change the training algorithm.

## Phase 2 scope

This compatibility pass changes device selection, Windows-safe paths, DataLoader worker setup,
PyTorch 2.6 loading behavior, and output-directory creation only. It does not change the paper's
model dimensions, codebook assignment, proposed loss, optimizer schedule, SNR policy, or channel
equations.

## Semantic-aware codebook final output

`SC_construction.py` accepts `--SCsize {10,16,32,64}` and preserves the original default value of
`10`. The periodic checkpoint behavior is unchanged: a codebook is saved every 10 loop epochs.
After all construction epochs complete, the script saves the final in-memory codebook state once
more to the same `results_data/SC_size{SCsize}.npy` path. This final-output correctness fix prevents
the last nine update epochs from being absent from the output file; it does not change codeword
distance, assignment, update, preprocessing, or iteration-count logic.

## Pilot resume correctness note

The Phase 7C pilot resume path loads only the codec `model_state_dict` with `strict=True`. It does
not restore `optimizer_state_dict`, because the original training loop intentionally creates a new
Adam optimizer at the beginning of every epoch. RNG and DataLoader state are also not restored.

The Epoch 10 pilot checkpoint predates classifier checkpointing and therefore has no
`classifier_state_dict`. On an Epoch 10 resume, the classifier is initialized from
`google_net.pkl`; any BatchNorm running-statistic changes made during Epochs 1-10 cannot be
recovered. This limitation is reported in the resume log. Starting with the Epoch 15 checkpoint,
pilot checkpoints include `classifier_state_dict`, and the pilot resume path restores it with
`strict=True` when present. Adding and restoring this saved state does not change classifier or
codec training behavior outside the requested resume continuity.

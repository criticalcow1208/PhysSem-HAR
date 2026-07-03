# PhysSem-HAR

PhysSem-HAR is a physical-semantic human activity recognition model for
channel-state information (CSI). It combines CSI patch representations, a
differentiable physical signal descriptor, language-model embeddings of
class-independent physical attributes, and a training-split-calibrated
class-attribute prior.

## Repository layout

```text
PhysSem-HAR/
├── main.py                 # Public entry point
├── model.py                # PhysSemHAR model
├── layers.py               # Model building blocks
├── physical_attributes.py  # Attribute definitions and weak targets
├── attribute_estimator.py  # Training-split Q calibration
├── prior.py                # Q serialization and validation
├── config.py               # Configuration dataclass
├── config.example.json
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

The model expects a local Hugging Face-compatible GPT-2 checkpoint. CSI tensors
must use shape `[batch, time, subcarrier]`.

## 1. Calibrate the attribute prior

Build Q using the training split only:

```python
from main import pretrain_attribute_estimator_and_build_prior

prior, estimator, metadata = pretrain_attribute_estimator_and_build_prior(
    train_loader=train_loader,
    class_names=["walk", "run", "sit"],
    device="cuda",
    save_path="attribute_prior.json",
    estimator_ckpt_path="attribute_estimator.pt",
)
```

Do not use validation or test samples in this step. Q is fixed during final
model training and inference.

## 2. Initialize PhysSem-HAR

```python
from main import PhysSemHAR, PhysSemHARConfig

config = PhysSemHARConfig(
    num_classes=3,
    patch_len=16,
    stride=8,
    enc_in=30,
    llm_path="/path/to/gpt2",
    label2id_path="label2id.json",
    attr_prior_path="attribute_prior.json",
    d_model=768,
    dropout=0.1,
    n_heads=8,
)
model = PhysSemHAR(config)
```

Alternatively, edit `config.example.json` and run:

```bash
python main.py --config config.example.json
```

## Forward outputs

- `model(x)`: class logits with shape `[batch, num_classes]`.
- `model(x, labels)`: logits, patch projection, attribute projection,
  attribute logits, and attribute targets.
- In `signal_only` ablation mode, the model always returns class logits.

Supported ablation modes are `full`, `no_prompt`, `rand_prompt`,
`no_descriptor`, `no_ppa`, `no_llm`, and `signal_only`.

## Reproducibility notes

- The GPT-2 parameters are frozen.
- The class IDs in `label2id.json` must be contiguous and start at zero.
- `d_model` must equal the hidden size of the selected GPT-2 checkpoint.
- Dataset preparation, loss composition, optimization, and evaluation belong
  to the surrounding experiment pipeline and are intentionally not assumed by
  this model-only repository.

Before publishing, add the license, citation, trained-weight policy, dataset
instructions, and experimental settings appropriate to your release.

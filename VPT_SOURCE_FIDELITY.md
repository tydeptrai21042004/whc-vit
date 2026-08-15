# VPT source-fidelity record

Source supplied by the user: `vpt-main.zip`.

The VPT implementation used by this repository was compared byte-for-byte with that source. The following SHA-256 hashes are enforced by `run_fair_vit_comparison.py` and `verify_fair_protocol.py`:

| File | SHA-256 |
|---|---|
| `src/models/vit_prompt/vit.py` | `06613eabb95b3b488c46957334bf4cf0d6fbe875afff9785eb1787d09581af1a` |
| `src/models/vit_prompt/vit_ablations.py` | `a49b2a6d49ad48e390d540eb1af7ad1dd7877fd165bd7d4a93a561b5be133291` |
| `src/models/build_vit_backbone.py` | `9d8d33c89852013854c46fd950724929cebee7e9c51a367827f856d61d8eb576` |
| `src/models/build_model.py` | `4712b2a1b10e13c1fea079cc5d19cbcd93f761971f429f7005e0a5c1486cf1cc` |
| `configs/base-prompt.yaml` | `5f2a75839468b730f89fa0774051a52673197f09bc1de1bcf9b3f0d9ed1a3f37` |
| `configs/prompt/flowers.yaml` | `b654b0f65255607718478c1a5b366c210d6b0fdfe2d610613cdc63da1b27baf5` |
| `configs/base-linear.yaml` | `5f437c22d78c59ea90e2967f724053f1825a972b1fa91f6abd1d671bb4c6daba` |
| `configs/base-finetune.yaml` | `cb774eb2c3f80e1396a1581db51719196a74e48bbd9b6f50887efcfb19c15adf` |

## Original VPT behavior reproduced

From the supplied source:

- shallow VPT uses `TRANSFER_TYPE: prompt`;
- prompt location is `prepend`;
- initialization is `random`;
- prompt projection defaults to disabled (`PROJECT=-1`);
- `DEEP=False` for shallow VPT;
- `VIT_POOL_TYPE=original`;
- prompt dropout is `0.0`;
- base prompt optimizer is SGD with momentum `0.9`;
- base weight decay is `1e-4`;
- the tuning scripts scale nominal learning rate by `batch_size/256`;
- Flowers102 prompt config does not override prompt length, so the supplied source default is 5 tokens.
- The supplied ZIP does not contain the separate dataset-specific VTAB prompt-length hyperparameter file. The fair VTAB-Caltech101 runner preserves the manuscript's 10-token VPT setting and labels it as manuscript-defined.

The VPT model/build implementation itself is not rewritten for DT1D.

## Intentional differences for the reviewer comparison

These are protocol changes, not changes to the VPT architecture:

1. The manuscript comparison uses the same 10-epoch budget for all methods rather than VPT's original 100-epoch FGVC base config.
2. Warmup is 1 epoch for the common 10-epoch paper budget rather than the original 10-epoch warmup inside a 100-epoch VPT schedule.
3. Each method receives 10 LR tuning candidates so the hyperparameter-search budget is equal.
4. Weight decay is fixed at the common/source-default value `1e-4`; it is not separately tuned for one method and not another.
5. VTAB keeps `train800`, `val200`, and test separated throughout. The original VPT final stage retrains on `train800+val200`, but the paper protocol does not because test accuracy must correspond to validation-selected checkpoints under the same rule for every method.
6. Final reported results use seeds `0,1,2` as requested for the manuscript rather than VPT's original five final seeds.

These differences should be stated if the results are described as a controlled paper comparison. They should not be described as an exact replication of every training detail in the VPT paper.

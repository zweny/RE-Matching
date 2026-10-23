# REMatching

This repository contains the offical code implementation of the ACL2023 paper "RE-Matching: A Fine-Grained Semantic Matching Method for Zero-Shot Relation Extraction"

## Environment and dependencies

The original experiments used Python 3.8.3 and Torch 1.7.0. These are
historical settings, not a verified environment for the updated dependencies.

### Package combination verified in issue #3

[Issue #3](https://github.com/zweny/RE-Matching/issues/3) reported that the
following package combination resolved the Hugging Face model-download URL
error. The issue was closed as completed on August 20, 2025.

| Package | Reported working version |
| --- | --- |
| `huggingface_hub` | `0.23.2` |
| `sentence-transformers` | `3.0.1` |
| `transformers` | `4.41.2` |
| `requests` | `>=2.31` |

These versions are now reflected in `requirements.txt`. The issue did not
report Python, Torch, NumPy, or scikit-learn versions, nor establish full
training or evaluation reproducibility with this package combination.
The existing NumPy and scikit-learn pins are retained.

Use a clean Python 3.8 environment for the retained dependencies. Install a
PyTorch build compatible with the updated sentence-transformers package and
your CPU/CUDA platform; do not force the historical Torch 1.7.0 pin when
installing the updated stack.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Use `python -m pip` so installation and execution use the same interpreter.
For an existing environment, upgrade to the requirements together:

```bash
python -m pip install --upgrade -r requirements.txt
python -m pip check
```

### Troubleshooting

- **`MissingSchema: Invalid URL '/api/resolve-cache/...': No scheme supplied`**
  when running `bash run_fewrel.sh`: this error means a relative URL reached
  Requests without an `http://` or `https://` scheme during model download.
  The reporter resolved it with the four-package combination above; the
  maintainer suggested a Requests version issue, but the exact root cause
  was not established. Reinstall the requirements together and retry the
  command from `model/`. No application-code URL patch was needed for the
  reported resolution.
- **The same error persists:** check the active interpreter and package
  versions with `python -m pip show huggingface_hub sentence-transformers transformers requests`,
  then run `python -m pip check`. Reproduce in a clean environment if an
  older installation is still being used.
- **Installation or dependency conflicts:** do not combine Torch 1.7.0 or
  the previous Transformers/sentence-transformers pins with the updated
  stack. The retained NumPy 1.19.2 pin also limits newer Python environments;
  use Python 3.8 rather than assuming support for the latest Python release.
- **Still unable to download models:** check access to Hugging Face and any
  proxy or custom endpoint configuration. If unresolved, open a new issue
  with the full traceback, Python/Torch versions, the four package versions,
  and the command used.

## File Structure:

The file structure of this repository is shown below.

```
REMatching
├── readme.md
├── ckpt
├── data
│   ├── fewrel
│   │   ├── fewrel_dataset.json
│   │   └── relation_description
│   ├── rel2id
│   │   ├── fewrel_rel2id
│   │   └── wikizsl_rel2id
│   └── wikizsl
│       ├── wikizsl_dataset.json
│       └── relation_description
├── model
│   ├── data_process.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── train.py
│   ├── model.py
│   ├── run_fewrel.sh
│   ├── run_inference.sh
│   ├── run_wikizsl.sh
│   └── log
└── requirements.txt
```

## Datasets：

You can download the dataset used in this work via the following google drive link,and then store them in the *data/fewrel* and *data/wikizsl* paths respectively.

[FewRel (Xu et al., 2018)](https://drive.google.com/file/d/1PgSTaEEUxsE-9lhQan3Yj91pzLhxv7cT/view?usp=sharing)

[WikiZSL (Daniil Sorokin and Iryna Gurevych, 2017)](https://drive.google.com/file/d/1kGmhlpTTq8UmIUPZ2CSIruWWsi_l_ERH/view?usp=share_link)

## Train&Inference :

You can easily run training as well as inference with the following scripts

```
# training stage
# The following two script files differ only in datasets
cd model
bash run_fewrel.sh
bash run_wikizsl.sh

# testing stage
bash run_inference.sh
```

## Acknowledgement

Our implementation and experiments are based on the codes from [ZS-BERT](https://github.com/dinobby/ZS-BERT), we appreciate their open-resourced codes.

## Citation:

If the code is useful for your research project, we appreciate if you cite the following:

```
@inproceedings{zhao2023rematching,
      title={RE-Matching: A Fine-Grained Semantic Matching Method for Zero-Shot Relation Extraction}, 
      author={Jun Zhao and Wenyu Zhan and Xin Zhao and Qi Zhang and Tao Gui and Zhongyu Wei and Junzhe Wang and Minlong Peng and Mingming Sun},
      year={2023},
      booktitle = {Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      address = {Toronto, Canada},
      publisher = {Association for Computational Linguistics},
}
```

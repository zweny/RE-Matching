# REMatching

This repository contains the offical code implementation of the ACL2023 paper "RE-Matching: A Fine-Grained Semantic Matching Method for Zero-Shot Relation Extraction"

## Env:

> python: 3.8.3
> 
> Torch: 1.7.0

## Dependencies:

Run the following script to install dependencies.

```
pip3 install -r requirements.txt
```

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

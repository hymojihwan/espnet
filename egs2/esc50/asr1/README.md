# ESC-50 Audio Classification Recipe

This recipe implements the audio classification task with a BEATs encoder and linear layer decoder model on the ESC-50 dataset, very close to what is described in [this paper](https://arxiv.org/abs/2212.09058).
More specifically, we provide the fine-tuning config and results for second last row in Table 1 (BEATS-iter3) from the paper.
We reuse part of the code from the [BEATs repository](https://github.com/microsoft/unilm/tree/master/beats) for this implementation.

# Training Details and Requirements
We perform 5-fold cross validation on ESC-50 dataset.
This dataset has 2k samples with 400 samples in each fold.
Please note that the hyper-parameters might be different from those in appendix A.1 of the BEATs paper, but the ones used here gave us best results.
They were tuned on fold 5 and then re-used for other folds.
Fine-tuning for one run needs 1 GPU with 33 GB memory and runs for ~4.5 hours on L40S.

### Steps to run

1. Download ESC-50 dataset from [this repo](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download) and set the path to its root directory in db.sh.
2. Download the BEATs checkpoint: [BEATs_iter3](https://github.com/microsoft/unilm/tree/master/beats) and change the `beats_ckpt_path` path in `conf/beats_classification.yaml`
3. Launch with `run.sh`


## Trained checkpoints
All trained checkpoints are available at:
* Fold-1: https://huggingface.co/shikhar7ssu/BEATs-ESC-FinetunedFold1 94.3
* Fold-2: https://huggingface.co/shikhar7ssu/BEATs-ESC-FinetunedFold2 97.0
* Fold-3: https://huggingface.co/shikhar7ssu/BEATs-ESC-FinetunedFold3 94.8
* Fold-4: https://huggingface.co/shikhar7ssu/BEATs-ESC-FinetunedFold4 96.3
* Fold-5: https://huggingface.co/shikhar7ssu/BEATs-ESC-FinetunedFold5 91.8

Average acc: 94.8

# Error Analysis
We also observe that top confusion in fold-5 are from the class `helicopter`, which is mainly confused with `washing machine`,  and `airplane`.

<!-- Generated by scripts/utils/show_asr_result.sh -->
# RESULTS
## Environments
- date: `Sat Dec 14 19:04:56 EST 2024`
- python version: `3.9.20 (main, Oct  3 2024, 07:27:41)  [GCC 11.2.0]`
- espnet version: `espnet 202412`
- pytorch version: `pytorch 2.4.0`
- Git hash: `cb80e61a15d6a13dc342ae5a413d2b870dd869c6`
  - Commit date: `Fri Dec 13 11:57:16 2024 -0500`

## /compute/babel-13-33/sbharad2/expdir/asr_fast.fold[i]/inference_ctc_weight0.0_maxlenratio-1_asr_model_valid.acc.best
### Accuracy

|dataset|Snt|Wrd|Acc|
|---|---|---|---|
|org/val1|400|400|94.3|
|org/val2|400|400|97.0|
|org/val3|400|400|94.8|
|org/val4|400|400|96.3|
|org/val5|400|400|91.8|
|Average|||94.8|
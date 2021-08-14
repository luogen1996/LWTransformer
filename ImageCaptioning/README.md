#  LightWeight Transformer For Image Captioning
This repository contains the  code of official Transformer and lightweight Transformer.

We have re-implemented the Transformer and optimized the codes so that its performance is much higher than all of existing IC repositorys.

This  codes are based on  repository of meshed memory transformer.

To perpare datas, please refer to [here](https://github.com/aimagelab/meshed-memory-transformer) 

## Evaluation
Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |

#### Expected output
Under `output_logs/`, you may also find the expected output of the  code.

## Training procedure
Run `python train.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--FFN` | Activate Lightweight FFN |
| `--head` | Number of heads (default: 8) |
| `--warmup` | Warmup value for learning rate scheduling (default: 10000) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |
| `--logs_folder` | Path folder for tensorboard logs (default: "tensorboard_logs")|

For example, to train our model with the parameters used in our experiments, use
```
python train.py --exp_name m2_transformer --batch_size 64  --LFFN  --head 8 --warmup 10000 --features_path /path/to/features --annotation_folder /path/to/annotations
```

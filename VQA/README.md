# LightWeight Transformer For Visual Question Answering
This repository contains the  code of official Transformer and lightweight Transformer for VQA.

The code is based on [OpenVQA](https://openvqa.readthedocs.io/en/latest/), please refer to the settings in OpenVQA for perapring datas.


## Evaluation
Run `python run.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--RUN` | Batch size ('train', 'val' and 'test') |
| `--MODEL` | The model name |
| `--DATASET` | The dataset ('vqa','gqa' and 'clevr') |
| `--CKPT_V` | Id number of your checkpoint file|
| `--CKPT_E` | The number of epoches you want to evaluate |

for example, to evaluate your model, you can use:
```
python3 run.py --RUN='val' --MODEL='lwtransformer' --DATASET='vqa' --CKPT_V=$ID --CKPT_E=13
```

#### Expected output
Under `results/`, you may also find the expected output of the  code.

## Training procedure
Run `python run.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--RUN` | Batch size ('train', 'val' and 'test') |
| `--MODEL` | The model name |
| `--DATASET` | The dataset ('vqa','gqa' and 'clevr') |
| `--SPLIT` | The dataset split you use for training (defalut: train)|

For example, to train our model with the parameters used in our experiments, use
```
python3 run.py --RUN='train' --MODEL='lwtransformer' --DATASET='vqa'
```



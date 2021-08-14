# LightWeight Transformer For Referring Expression Comprehension
This repository contains the  code of official Transformer and lightweight Transformer for REC.

This code is implemented by us, a part of codes are referred to [DDPN(caffe)](https://github.com/XiangChenchao/DDPN). 

We also have implemented DDPN as the baseline model, which slightly outperforms the official DDPN.

For data preparation, you can refer to the settings of  [DDPN(caffe)](https://github.com/XiangChenchao/DDPN)  to use it.

## Evaluation

| Argument | Possible values |
|------|------|
| `--version` | Id number of your checkpoint files (see ./modes/) |

Please check the config file in `config/base_config.py`, then  run `python test.py --version $version` .

#### Expected output
Under `models/`, you may  find the expected output.

## Training procedure
Please check the config file in `config/base_config.py` is set correctly, then you can just run `python train.py` 



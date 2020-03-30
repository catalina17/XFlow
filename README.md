# XFlow

[**XFlow: Cross-modal Deep Neural Networks for Audiovisual Classification**](https://arxiv.org/abs/1709.00572)  
*[IEEE Transactions on Neural Networks and Learning Systems](https://ieeexplore.ieee.org/document/8894404), IEEE ICDL-EPIROB Workshop on Computational Models for Crossmodal Learning (CMCML) 2017, [ARM Research Summit 2017](https://developer.arm.com/research/summit/previous-summits/2017/speakers)*  
[Cătălina Cangea](www.cl.cam.ac.uk/~ccc53/), [Petar Veličković](www.cl.cam.ac.uk/~pv273/), [Pietro Liò](www.cl.cam.ac.uk/~pl219/)  

We propose **XFlow, cross-modal deep learning architectures** that allow for dataflow between several feature extractors. Our models derive more interpretable features and achieve better performances than models which do not exchange representations. They represent a novel method for performing cross-modality **before** features are learned from individual modalities, usefully exploiting correlations between audio and visual data, which have a different dimensionality and are nontrivially exchangeable. We also provide the research community with **Digits**, a new dataset consisting of three data types extracted from videos of people saying the digits 0-9. Results show that both cross-modal architectures outperform their baselines (by up to 11.5%) when evaluated on the AVletters, CUAVE and Digits datasets, achieving state-of-the-art results.

<img src="https://github.com/catalina17/XFlow/blob/master/images/high_level.png" height=250>
<img src="https://github.com/catalina17/XFlow/blob/master/images/xconn.png" height="350">

## Getting started

```
$ git clone https://github.com/catalina17/XFlow
$ virtualenv -p python3 xflow
$ source xflow/bin/activate
$ pip install tensorflow-gpu==1.8.0
$ pip install keras==2.1.4
```

## Dataset

The **Digits** benchmark data can be found [here](https://www.cl.cam.ac.uk/~ccc53/files/digits.tar.gz). After expanding the archive in a specific directory, please update `BASE_DIR` (declared in `Datasets/data_config.py`) with that directory.

<img src="https://github.com/catalina17/XFlow/blob/master/images/frames.png" height="150">

## Running the models

The script `eval.py` contains command-line arguments for models and datasets. For example, you can run the {CNN x MLP}--LSTM baseline on Digits as follows:
```
CUDA_VISIBLE_DEVICES=0 python eval.py --model=cnn_mlp_lstm_baseline --dataset=digits --batch_size=64
```

## Citation
Please cite us if you get inspired by or use XFlow and/or the Digits dataset:
```
@ARTICLE{8894404,
  author={C. {Cangea} and P. {Veličković} and P. {Liò}},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={XFlow: Cross-Modal Deep Neural Networks for Audiovisual Classification},
  year={2019},
  volume={}, number={}, pages={1-10},
}
```

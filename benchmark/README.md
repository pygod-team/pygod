# PyGOD Benchmark

Official implementation of paper [Benchmarking Node Outlier Detection on Graphs](https://arxiv.org/abs/2206.10071). **Please star, watch, and fork us for the active updates!**

## Usage

**Please update to the latest PyGOD version before the experiments.**

To obtain the main result of each model on each dataset, run:

```shell
python main.py [-h] [--model MODEL] [--gpu GPU] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      supported model: [lof, if, mlpae, scan, radar, anomalous,
                     gcnae, dominant, done, adone, anomalydae, gaan, guide,
                     conad]. Default: dominant
  --gpu GPU          GPU Index. Default: -1, using CPU.
  --dataset DATASET  supported dataset: [inj_cora, inj_amazon, inj_flickr,
                     weibo, reddit]. Default: inj_cora
```

To obtain the result of different types of outliers, run:

```shell
python type.py [-h] [--model MODEL] [--gpu GPU] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      supported model: [lof, if, mlpae, scan, radar, anomalous,
                     gcnae, dominant, done, adone, anomalydae, gaan, guide,
                     conad]. Default: dominant
  --gpu GPU          GPU Index. Default: -1, using CPU.
  --dataset DATASET  supported dataset: [inj_cora, inj_amazon, inj_flickr].
                     Default: inj_cora.

```

To obtain the runtime, run:

```shell
python time.py [-h] [--model MODEL] [--gpu GPU]

optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  supported model: [lof, if, mlpae, scan, radar, anomalous,
                 gcnae, dominant, done, adone, anomalydae, gaan, guide,
                 conad]. Default: dominant
  --gpu GPU      GPU Index. Default: -1, using CPU.
```

For the GPU memory consumption experiments, we use pytorch_memlab to measure the peak of the active bytes. See [pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab) for more details.

## Citing us

Our [paper](https://arxiv.org/abs/2206.10071) is available on arxiv. If you use PyGOD in a scientific publication, we would appreciate citations to the following paper:

```
@article{liu2022benchmarking,
  author  = {Liu, Kay and Dou, Yingtong and Zhao, Yue and Ding, Xueying and Hu, Xiyang and Zhang, Ruitong and Ding, Kaize and Chen, Canyu and Peng, Hao and Shu, Kai and Sun, Lichao and Li, Jundong and Chen, George H. and Jia, Zhihao and Yu, Philip S.},
  title   = {Benchmarking Node Outlier Detection on Graphs},
  journal = {arXiv preprint arXiv:2206.10071},
  year    = {2022},
}
```

or:

```
Liu, K., Dou, Y., Zhao, Y., Ding, X., Hu, X., Zhang, R., Ding, K., Chen, C., Peng, H., Shu, K., Sun, L., Li, J., Chen, G.H., Jia, Z., and Yu, P.S. 2022. Benchmarking Node Outlier Detection on Graphs. arXiv preprint arXiv:2206.10071.
```

# PyGOD Benchmark

Official implementation of paper [BOND: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs](https://proceedings.neurips.cc/paper_files/paper/2022/hash/acc1ec4a9c780006c9aafd595104816b-Abstract-Datasets_and_Benchmarks.html). Our datasets are publicly available in the [data repository](https://github.com/pygod-team/data). **Please star, watch, and fork us for the active updates!**

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
                     weibo, reddit, disney, books, enron]. Default: inj_cora
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

For DGraph, we are not able to load the dataset automatically, because of the authors' restrictions. To reproduce the results, the dataset is publicly available [here](https://dgraph.xinye.com/dataset), and we detect the outliers on the whole graph and evaluate only on the test set. As for the GPU memory consumption experiments, we use pytorch_memlab to measure the peak of the active bytes. See [pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab) for more details.

## Cite us

Our [benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/acc1ec4a9c780006c9aafd595104816b-Abstract-Datasets_and_Benchmarks.html) is publicly available. If you use BOND in a scientific publication, we would appreciate citations to the following paper:

```
    @article{liu2022bond,
      title={Bond: Benchmarking unsupervised outlier node detection on static attributed graphs},
      author={Liu, Kay and Dou, Yingtong and Zhao, Yue and Ding, Xueying and Hu, Xiyang and Zhang, Ruitong and Ding, Kaize and Chen, Canyu and Peng, Hao and Shu, Kai and Sun, Lichao and Li, Jundong and Chen, George H. and Jia, Zhihao and Yu, Philip S.},
      journal={Advances in Neural Information Processing Systems},
      volume={35},
      pages={27021--27035},
      year={2022}
    }
```

or:

```
Liu, K., Dou, Y., Zhao, Y., Ding, X., Hu, X., Zhang, R., Ding, K., Chen, C., Peng, H., Shu, K. and Sun, L., Li, J., Chen, G.H., Jia, Z., and Yu, P.S. 2022. Bond: Benchmarking unsupervised outlier node detection on static attributed graphs. Advances in Neural Information Processing Systems, 35, pp.27021-27035.
```

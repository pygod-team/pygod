# Benchmarking Node Outlier Detection on Graphs

To obtain the main result of each model on each dataset, run:
```shell
python main.py --model dominant --dataset inj_cora --gpu 0
```

To obtain the result of different types of outliers, run:
```shell
python type.py --model dominant --dataset inj_cora --gpu 0
```

To obtain the runtime, run:
```shell
python time.py --model dominant --dataset inj_cora --gpu 0
```

For the GPU memory consumption, we use pytorch_memlab to measure the peak of the active bytes. See [pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab) for more details.

# HAI
Heterogeneous Graph Learning for Explainable Recommendation over Academic Networks
[HAI](https://arxiv.org/abs/2202.07832)

# Reference
```
@inproceedings{chen2021heterogeneous,
  title={Heterogeneous Graph Learning for Explainable Recommendation over Academic Networks},
  author={Chen, Xiangtai and Tang, Tao and Ren, Jing and Lee, Ivan and Chen, Honglong and Xia, Feng},
  booktitle={IEEE/WIC/ACM International Conference on Web Intelligence},
  pages={29--36},
  year={2021}
}
```

# Requirements
This model is inspired by [HAN](https://github.com/dmlc/dgl/tree/master/examples/pytorch/han)
```
conda create -n hai --clone pytorch
conda activate hai
pip install dgl nni scikit-learn
```

# Preprocess
The processed dataset is stored in folder Dataprocessing.

# Run
``` python main.py ```

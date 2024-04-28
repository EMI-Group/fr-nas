
## FR-NAS: Forward-and-Reverse Graph Predictor for Efficient Neural Architecture Search
  <a href="https://arxiv.org/abs/2404.15622">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="FR-NAS Paper on arXiv">
  </a>

Neural Architecture Search (NAS) has emerged as a key tool in identifying optimal configurations of deep neural networks tailored to specific tasks.
However, training and assessing numerous architectures introduces considerable computational overhead. 
We propose a GIN predictor and new training procedure for performance evaluation in NAS. 


## Train and Evaluation
run `experiments/predictor/predictor_compare.py` for prediction compare. 

The results are stored in `experiments/predictor/result)`.

Process the result using code in `experiments/predictor/result_parser.py`.

## Source File Structure
- `experiments`: Experiments conducted for the paper.
- `nas_lib/model_predictor`:
  - `agent`: Neural predictors.
  - `search_space`: Search spaces, including database interactions and data preprocessing.
  - `trainer`: Training procedures for models.


## Environment Setup

 - python=3.8
 - scipy=1.4.1
 - torch=1.12.1+cu116
 - torch-geometric 


### Installing EvoXBench

To utilize the [EvoXBench](https://github.com/EMI-Group/evoxbench) database, configure it after installation:

```python
from evoxbench.database.init import config

config("Path to database", "Path to data")
# For instance:
# With this structure:
# /home/Downloads/
# └─ database/
# |  |  __init__.py
# |  |  db.sqlite3
# |  |  ...
# |
# └─ data/
#    └─ darts/
#    └─ mnv3/
#    └─ ...
# Then, execute:
# config("/home/Downloads/database", "/home/Downloads/data")
```

### Installation Commands

#### For Windows (CPU version)
```bash
# pytorch 1.12.1
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
# torch-geometric
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_cluster-1.6.0-cp38-cp38-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_scatter-2.0.9-cp38-cp38-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_sparse-0.6.14-cp38-cp38-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_spline_conv-1.2.1-cp38-cp38-win_amd64.whl
pip install torch-geometric
```

#### For Windows (CUDA 11.3 version)
```bash
# pytorch 1.12.1
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# torch-geometric
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0-cp38-cp38-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.14-cp38-cp38-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1-cp38-cp38-win_amd64.whl
pip install torch-geometric
```

## Citing FR-NAS

If you use FR-NAS in your research and want to cite it in your work, please use:

```
@misc{zhang2024frnas,
      title={FR-NAS: Forward-and-Reverse Graph Predictor for Efficient Neural Architecture Search}, 
      author={Haoming Zhang and Ran Cheng},
      booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
      year={2024}
}
```


## Acknowledge
This code is implemented based on the framework provided by [NPENASv1](https://github.com/auroua/NPENASv1?tab=readme-ov-file#acknowledge)

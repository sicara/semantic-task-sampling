# Semantic Task Sampling

Code for **Few-Shot Image Classification Benchmarks are Too Far From Reality: Build Back Better with Semantic Task Sampling**,
published in CVPR 2022 Workshop on Vision Datasets Understanding.

Implementations of few-shot datasets, loading tools and methods are mostly derived from the
open-source library for Few-Shot Learning research [EasyFSL](https://github.com/sicara/easy-few-shot-learning).

If you're having any issue while retrieving the data or checkpoints, or running the experiments, 
please raise an issue.

Check out the project's dashboard [here](https://share.streamlit.io/sicara/semantic-task-sampling).

## Installation

### Requires
- Python 3.8
- PyTorch 1.7
- To install PyGraphViz (for graph visualization):
  
    `sudo apt-get install graphviz graphviz-dev libpython3.8-dev`

### Do

1. Create a virtualenv with Python 3.8
2. `pip install -r dev_requirements.txt`

### Data
We expect images for tieredImageNet to be stored like this:
```
data
|
|----tiered_imagenet
|    |
|    |----images
|    |    |
|    |    |----n04542943
|    |    |----n04554684
|    |    |----...
```
If you can't host the data there for any reason, you can create a symlink:
```bash
ln -s path/to/where/your/data/really/is data/tiered_imagenet/images
```

For Fungi: we expect all images to be directly in `data/fungi/images`, with no further file structure.

To retrieve the weights for ResNet12 trained on _tiered_ImageNet's base set:

```bash
dvc pull data/tiered_imagenet/models/resnet12_tiered_imagenet_classic.tar
```

## Run

We run our pipelines with DVC:

```
dvc repro pipelines/fungi/dvc.yaml
```

(and same with tieredImageNet)

You can update the parameters of the experiment in the `params.yaml` file next to your `dvc.yaml` pipeline.
It's there that you can select your method (_eg_ `PT_MAP`) and the shape of the testbed.

For tieredImageNet you need the pretrained weights (added in supplementary materials). 
For Fungi we use the pretrained weights on ImageNet that are directly downloadable with PyTorch.

## Visualize testbeds

Most visualization posted in the papers can be recreated from our Streamlit scripts
`st_explore_tiered_imagenet.py` and `st_explore_fungi.py`, runnable like this:

```
 PYTHONPATH=. streamlit run st_scripts/st_explore_tiered_imagenet.py 
```

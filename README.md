# Semantic Task Sampling

Research code for experiments on semantic task sampling.

## Installation

### Requires
- Python 3.8
- PyTorch 1.7
- To install PyGraphViz (for graph visualization):
  
    `sudo apt-get install graphviz graphviz-dev libpython3.8-dev`

### Do

1. Create a virtualenv with Python 3.8
2. `pip install -r dev_requirements.txt`

### Paths to datasets
Paths to images are defined in specification files such as [this one](data/tiered_imagenet/specs/train.json).
All images are expected to be found in `data/{dataset_name}/images`. For instance,
for tieredImageNet we expect a structure like this one:
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

# Prerequisites

- Linux or macOS (Windows is not currently officially supported)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)


The required versions of MMCV and MMDetection for different versions of MMDetection3D are as below. Please install the correct version of MMCV and MMDetection to avoid installation issues.

| MMDetection3D version | MMDetection version |    MMCV version     |
|:-------------------:|:-------------------:|:-------------------:|
| master              | mmdet>=2.5.0        | mmcv-full>=1.2.4, <=1.4|
| 0.11.0              | mmdet>=2.5.0        | mmcv-full>=1.2.4, <=1.4|
| 0.10.0              | mmdet>=2.5.0        | mmcv-full>=1.2.4, <=1.4|
| 0.9.0               | mmdet>=2.5.0        | mmcv-full>=1.2.4, <=1.4|
| 0.8.0               | mmdet>=2.5.0        | mmcv-full>=1.1.5, <=1.4|
| 0.7.0               | mmdet>=2.5.0        | mmcv-full>=1.1.5, <=1.4|
| 0.6.0               | mmdet>=2.4.0        | mmcv-full>=1.1.3, <=1.2|
| 0.5.0               | 2.3.0               | mmcv-full==1.0.5|

# Installation

## Install MMDetection3D

**a. Create a conda virtual environment and activate it.**

```shell
conda create -n bevfusion python=3.8 -y
conda activate bevfusion
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
conda install pytorch torchvision -c pytorch
or
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

 ```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install [MMDetection](https://github.com/open-mmlab/mmdetection).**

```shell
pip install git+https://github.com/open-mmlab/mmdetection.git
```

Optionally, you could also build MMDetection from source in case you want to modify the code:

```shell
pip install mmsegmentation==0.14.1
cd mmdetection-2.11.0
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

**d-1. Install requirements.**
```shell
pip install -r requirements/runtime.txt
```

**e. Clone the MMDetection3D repository.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**f. Install timm.**
```shell
pip install timm
```

Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C++/CUDA codes are modified, then this step is compulsory.

    > Important: Be sure to remove the `./build` folder if you reinstall mmdet with a different CUDA/PyTorch version.

    ```shell
    pip uninstall mmdet3d
    rm -rf ./build
    find . -name "*.so" | xargs rm
    ```

1. Following the above instructions, mmdetection is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

2. If you would like to use `opencv-python-headless` instead of `opencv-python`,
you can install it before installing MMCV.

1. Some dependencies are optional. Simply running `pip install -v -e .` will only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.

2. The code can not be built for CPU only environment (where CUDA isn't available) for now.

## Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmdetection3d/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmdetection3d docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
```

## A from-scratch setup script

Here is a full script for setting up mmdetection with conda.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install mmcv
pip install mmcv-full

# install mmdetection
pip install git+https://github.com/open-mmlab/mmdetection.git

# install mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v -e .
```

## Using multiple MMDetection3D versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMDetection3D in the current directory.

To use the default MMDetection3D installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

# Verification

## Demo

### Point cloud demo

We provide a demo script to test a single sample. Pre-trained models can be downloaded from [model zoo](model_zoo.md)

```shell
python demo/pcd_demo.py ${PCD_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--score-thr ${SCORE_THR}] [--out-dir ${OUT_DIR}]
```

Examples:

```shell
python demo/pcd_demo.py demo/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
```
If you want to input a `ply` file, you can use the following function and convert it to `bin` format. Then you can use the converted `bin` file to generate demo.
Note that you need to install pandas and plyfile before using this script. This function can also be used for data preprocessing for training ```ply data```.
```python
import numpy as np
import pandas as pd
from plyfile import PlyData

def conver_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)
```
Examples:

```python
convert_ply('./test.ply', './test.bin')
```

## High-level APIs for testing point clouds

### Synchronous interface
Here is an example of building the model and test given point clouds.

```python
from mmdet3d.apis import init_detector, inference_detector

config_file = 'configs/votenet/votenet_8x8_scannet-3d-18class.py'
checkpoint_file = 'checkpoints/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
point_cloud = 'test.bin'
result, data = inference_detector(model, point_cloud)
# visualize the results and save the results in 'results' folder
model.show_results(data, result, out_dir='results')
```

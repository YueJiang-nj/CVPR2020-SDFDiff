# SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization

**IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020 (Oral)**

Authors: **Yue Jiang, Dantong Ji, Zhizhong Han, Matthias Zwicker**

**Paper:** http://www.cs.umd.edu/~yuejiang/papers/SDFDiff.pdf

**Video:** https://www.youtube.com/watch?v=l3h9JZHAOqI&t=13s
>

## Prerequisite Installation

    1. Python3 
    2. CUDA10
    3. Pytorch


## To Get Started: 

SDFDiff has been implemented and tested on Ubuntu 18.04 with python >= 3.7.

Clone the repo:
``` bash
git clone https://github.com/YueJiang-nj/CVPR2020-SDFDiff.git
```

Install the requirements using `virtualenv` or `conda`:
``` bash
# pip
source virtual_env/install_pip.sh

# conda
source virtual_env/install_conda.sh
```

## Introduction

The project has the following file layout:

    README.md
    multi_view_code/
        bunny.sdf
        dragon.sdf
        code/
            main.py
            renderer.cpp
            renderer_kernel.cu
            setup.py
     single_view_code/
        differentiable_rendering.py
        main.py
        models.py
        renderer.cpp
        renderer_kernel.cu
        setup.py


**multi_view_code** contains the source code for multi-view 3D reconstruction using our SDFDiff.

**single_view_code** contains the source code for single-view 3D reconstruction using our SDFDiff and deep learning models.

## Running the Demo

We have prepared a demo to run SDFDiff on a bunny object. 

To run the multi-view 3D reconstruction on bunny, you can follow the following steps in the folder multi_view_code/code:

``` bash
1. You need to run “python setup.py install” to compile our SDF differentiable renderer.

2. Once built, you can execute the bunny reconstruction example via “python main.py”
```

## Parameter Tuning

There are two kinds of parameters you can modify to get better results:

```
1. Weighted Loss
In the line: loss = image_loss[cam] + sdf_loss[cam] + Lp_loss
You can make it weighted. loss = a * image_loss[cam] + b * sdf_loss[cam] + c * Lp_loss and try different a, b, c. For example, the surface would be smoother if you increase c.

2. Intermediate Resolutions
In the line: voxel_res_list = [8,16,24,32,40,48,56,64]
You can add more intermediate resolutions in the list. It can also produce better results when we have more intermediate resolutions.
```

## Generating SDF from Mesh

If you have a mesh file xxx.obj, you need to generate SDF from the mesh file to run our SDFDiff code.

First, you need to git clone the following tools.

``` bash
# a tool to generate watertight meshes from arbitrary meshes
git clone https://github.com/hjwdzh/Manifold.git

# A tool to generate SDF from watertight meshes
git clone https://github.com/christopherbatty/SDFGen.git
```

Then you can run the following to get SDF from your mesh file xxx.obj.

``` bash
# Generate watertight meshes from arbitrary meshes
./Manifold/build/manifold ./obj_files/xxx.obj ./watertight_meshes_and_sdfs/xxx.obj

# Generate SDF from watertight meshes
./SDFGen/build/bin/SDFGen ./watertight_meshes_and_sdfs/xxx.obj 0.002 0 
```

## Citation
```bibtex
@InProceedings{jiang2020sdfdiff,
    author = {Jiang, Yue and Ji, Dantong and Han, Zhizhong and Zwicker, Matthias},
    title = {SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020} 
}
```

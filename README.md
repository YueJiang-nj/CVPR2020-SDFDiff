# SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization

**IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020 (Oral)**

Authors: **Yue Jiang, Dantong Ji, Zhizhong Han, Matthias Zwicker**

**Paper:** http://www.cs.umd.edu/~yuejiang/papers/SDFDiff.pdf

**Video:** https://www.youtube.com/watch?v=l3h9JZHAOqI&t=13s
>

## Prerequisite installation

    1. Python3 
    2. CUDA10
    3. Pytorch


> **To get started:** 

SDFDiff has been implemented and tested on Ubuntu 18.04 with python >= 3.7.

Clone the repo:
> 
>     git clone https://github.com/YueJiang-nj/CVPR2020-SDFDiff.git


Install the requirements using `virtualenv` or `conda`:
```
# pip
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```


## Layout

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


**multi_view_code** contains the source code for multi-view 3D reconstruction using our SDFDiff;

**single_view_code** contains the source code for single-view 3D reconstruction using our SDFDiff and deep learning models;

To run multi-view 3D reconstruction example, you can follow the following steps in the folder multi_view_code/code:

1. You need to run “python setup.py install” to compile our SDF differentiable renderer.

2. Once built, you can execute the bunny reconstruction example via “python main.py”
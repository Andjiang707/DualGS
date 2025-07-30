# [SIGGRAPH Asia 2024] Robust Dual Gaussian Splatting for Immersive Human-centric Volumetric Videos
[Yuheng Jiang](https://nowheretrix.github.io/), [Zhehao Shen](https://moqiyinlun.github.io/), [Yu Hong](https://github.com/xyi1023), [Chengcheng Guo](https://github.com/gcccccccccccc12345), [Yize Wu](https://github.com/wuyize25), [Yingliang Zhang](https://scholar.google.com/citations?user=SJJwxTQAAAAJ&hl=en), [Jingyi Yu](https://faculty.sist.shanghaitech.edu.cn/yujingyi/), [Lan Xu](http://xu-lan.com/)<br>
| [Webpage](https://nowheretrix.github.io/DualGS/) | [Full Paper](https://arxiv.org/abs/2409.08353) |
[Video](https://youtu.be/vwDE8xr78Bg) |
<br>
![Teaser image](assets/teaser.png)

## Overview
Official implementation of [DualGS](https://arxiv.org/abs/2409.08353) (Robust Dual Gaussian Splatting for Immersive Human-centric Volumetric Videos)

We present a novel Gaussian-Splatting-based approach, dubbed DualGS, for real-time and high-fidelity playback of complex human performance with excellent compression ratios. Our key idea in DualGS is to separately represent motion and appearance using the corresponding skin and joint Gaussians. Such an explicit disentanglement can significantly reduce motion redundancy and enhance temporal coherence. We begin by initializing the DualGS and anchoring skin Gaussians to joint Gaussians at the first frame. Subsequently, we employ a coarse-to-fine training strategy for frame-by-frame human performance modeling. It includes a coarse alignment phase for overall motion prediction as well as a fine-grained optimization for robust tracking and high-fidelity rendering.
<details open>
<summary style="cursor: pointer; font-weight: bold; color: #0366d6;">Show/Hide</summary>
<div style="display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-top: 10px;">
  <img src="assets/piano.webp" alt="Piano" style="width: 32.5%; height: auto; object-fit: contain;">
  <img src="assets/flute.webp" alt="Flute" style="width: 32.5%; height: auto; object-fit: contain;">
  <img src="assets/guitar.webp" alt="Guitar" style="width: 32.5%; height: auto; object-fit: contain;">
</div>
</details>

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# HTTPS
git clone https://github.com/HiFi-Human/DualGS.git --recursive
```
or
```shell
# SSH
git clone git@github.com:HiFi-Human/DualGS.git --recursive
```

## Setup

Our provided install method is based on Conda package and environment management:

Create a new environment
```shell
conda create -n dualgs python=3.10
conda activate dualgs
```
First install CUDA and PyTorch, our code is evaluated on CUDA 11.8 and PyTorch 2.1.2+cu118. Then install the following dependencies:
```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
pip install -r requirements.txt
```

## Dataset
Our code mainly evaluated on multi-view human centric datasets including [HiFi4G](https://github.com/moqiyinlun/HiFi4G_Dataset) and [DualGS](https://github.com/xyi1023/DualGS_Dataset) datasets. Please download the data you needed.

The overall file structure is as follows:
```shell
<location>
├── image_white
│    ├── %d                - The frame number, starts from 0.
│    │   └──%d.png         - Masked RGB images for each view. View number starts from 0.
│    └── transforms.json   - Camera extrinsics and intrinsics in instant-NGP format.
│
├── image_white_undistortion
│    ├── %d                - The frame number, starts from 0.
│    │   └──%d.png         - Undistorted maksed RGB images for each view. View number starts from 0.
│    └── colmap/sparse/0   - Camera extrinsics and intrinsics in Gaussian Splatting format.
```

## Training

```shell
python train.py \
  -s <path to HiFi4G or DualGS dataset> -m <output path> \
  --frame_st 0 --frame_ed 1000 \
  --iterations 30000 --subseq_iters 15000 \
  --training_mode 0 \
  --parallel_load \
  -r 2
```
</details>
Note that to achieve better training results, it is recommended to use undistorted images(image_white_undistortion) for training. 

### Core Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| <code style="white-space: nowrap;">--iterations</code> | int | Total training iterations for the **first frame** covering both stages:<br>- Stage 1: JointGS training.<br>- Stage 2: SkinGS training. |
| <code style="white-space: nowrap;">--subseq_iters</code> | int | Training iterations per frame for **subsequent frames** after the first frame. |
| <code style="white-space: nowrap;">--frame_st</code> | int | Start frame number. |
| <code style="white-space: nowrap;">--frame_ed</code> | int | End frame number. |
| <code style="white-space: nowrap;">--training_mode</code> | {0,1,2} | Training pipeline selection:<br>- <code>0</code>: Both stage training (JointGS + SkinGS).<br>- <code>1</code>: JointGS only.<br>- <code>2</code>: SkinGS only. |
| <code style="white-space: nowrap;">--ply_path</code> | str | The path to the point cloud used for initialization (defaults to points3d.ply in the dataset). |
| <code style="white-space: nowrap;">--motion_folder</code> | str | If you already have a trained JointGS and want to train only SkinGS, you can use this parameter to manually specify the path to JointGS. |
| <code style="white-space: nowrap;">--parallel_load</code> | flag | Enables multi-threaded image loading during dataset loading. |
| <code style="white-space: nowrap;">--seq</code> | flag | By default, training warps each frame from the first frame to the n-th frame. Enabling this parameter will instead warp from the (n-1)-th frame to the n-th frame. |

The results are as follows:
```shell
<location>
├── track
│    └── ckt                   - The results of JointGS.
│    │   ├── point_cloud_0.ply            
│    │   ...
│    │   └── point_cloud_n.ply  
│    ├── cameras.json    
│    └── cfg_args
├── ckt                        - The results of SkinGS.
│    ├── point_cloud_0.ply            
│    ...
│    └── point_cloud_n.ply   
├── joint_opt                  - The RT matrix of JointGS after stage2 optimization.
│    ├── joint_RT_0.npz    
│    ...
│    └── joint_RT_n.npz        
├── cameras.json    
└── cfg_args
```


## Evaluation
### Render
```shell
python render.py -m <path to trained model> -st <start frame number> -e <end frame number> --parallel_load # Generate renderings
```
You are able to select the desired views by using the --views parameter.

### Evaluate
```shell
python scripts/evaluation.py -g <path to gt> -r <path to renderings> # Compute error metrics on renderings
```
## Viewer

### Installation

Our modified Viewer is located in the `DynamicGaussianViewer/` directory.  
The build process is identical to that of the official Gaussian Splatting repository.

To compile the viewer, please follow the official instructions:  
👉 [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

```bash
cd DynamicGaussianViewer/
# Follow the same steps as in the official repo to build the viewer
```
### Usage
```
./install/bin/SIBR_gaussianViewer_app_rwdi.exe -m <path to the folder where cfg_args and cameras.json exist> -d <path to point clouds folder> -start <start frame> -end <end frame> 
# optional: --step 1 --rendering-size 1920 1080 
```

## Acknowledgments
We would like to thank the authors of [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs) for their excellent implementation, which was used in our project to replace the original 3DGS for acceleration.

## License

This project contains code from multiple sources with distinct licensing terms:

### 1. Original Gaussian Splatting Code
The portions of code derived from the original [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) implementation are licensed under the **Gaussian Splatting Research License**.  
📄 See: [LICENSE.original](LICENSE.original.md)

### 2. Our Modifications and Additions
All code modifications, extensions, and new components developed by our team are licensed under MIT License.  
📄 See: [LICENSE](LICENSE.md)

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{jiang2024robust,
  title={Robust dual gaussian splatting for immersive human-centric volumetric videos},
  author={Jiang, Yuheng and Shen, Zhehao and Hong, Yu and Guo, Chengcheng and Wu, Yize and Zhang, Yingliang and Yu, Jingyi and Xu, Lan},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--15},
  year={2024},
  publisher={ACM New York, NY, USA}
}</code></pre>
  </div>
</section>


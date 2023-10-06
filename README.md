# 3D Human Mesh Recovery with Sequentially Global Rotation Estimation

[[`Paper`](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_3D_Human_Mesh_Recovery_with_Sequentially_Global_Rotation_Estimation_ICCV_2023_paper.pdf)]

> [3D Human Mesh Recovery with Sequentially Global Rotation Estimation](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_3D_Human_Mesh_Recovery_with_Sequentially_Global_Rotation_Estimation_ICCV_2023_paper.html)  
> Dongkai Wang, Shiliang Zhang  
> ICCV 2023  


## Installation

### 1. Clone code
```shell
    git clone https://github.com/kennethwdk/SGRE
    cd ./SGRE
```
### 2. Create a conda environment for this repo
```shell
    conda create -n SGRE python=3.9
    conda activate SGRE
```
### 3. Install PyTorch >= 1.6.0 following official instruction, *e.g.*,
```shell
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```
We conduct experiments based on PyTorch 1.11.0 and cudatoolkit 11.3.1. You can follow this setting to reproduce our results.
### 4. Install other dependency python packages
```shell
    pip install -r requirements.txt
```
You should slightly change torchgeometry kernel code following [here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527).
### 5. Prepare dataset
Please follow [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE/blob/main/assets/directory.md) to preparee dataset, and they should look like the following structure.

```  
${ROOT}  
|-- data 
|   |-- J_regressor_extra.npy 
|   |-- Human36M  
|   |   |-- images  
|   |   |-- annotations   
|   |   |-- J_regressor_h36m_correct.npy
|   |-- MuCo  
|   |   |-- data  
|   |   |   |-- augmented_set  
|   |   |   |-- unaugmented_set  
|   |   |   |-- MuCo-3DHP.json
|   |   |   |-- smpl_param.json
|   |-- MSCOCO  
|   |   |-- images  
|   |   |   |-- train2017  
|   |   |   |-- val2017  
|   |   |-- annotations  
|   |   |-- J_regressor_coco_hip_smpl.npy
|   |-- MPII
|   |   |-- data  
|   |   |   |-- annotations
|   |   |   |-- images
|   |-- PW3D
|   |   |-- data
|   |   |   |-- 3DPW_latest_train.json
|   |   |   |-- 3DPW_latest_validation.json
|   |   |   |-- 3DPW_latest_test.json
|   |   |   |-- 3DPW_validation_crowd_hhrnet_result.json
|   |   |   |-- imageFiles
```  
## Usage

### 1. Download trained model
* [SGRE Model](https://1drv.ms/f/s!AhpKYLhXKpH7g8t-xRI2d3_U6rPWeg?e=KNF5QQ)
* [Pre-trained Model](https://1drv.ms/f/s!AhpKYLhXKpH7g8t8Midw6Mh_0fu-eQ?e=YACLOC)
* [SMPL Model](https://smpl.is.tue.mpg.de/), you should download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to ${ROOT}/common/utils/smplpytorch/smplpytorch/native/models.
### 2. Evaluate Model
In `main` folder run
```python
# evaluate on 3DPW test set with 1 gpus
python test.py --gpu 0-0 --exp_dir ../output/exp_09-01_09_32 --test_epoch 10
```

### 3. Train Model
We use the pre-trained ResNet-50 weights on COCO. Download the file of weights from above pre-trained model link and place it under ${ROOT}/tool/.

In `main` folder run

```python
python train.py --amp --continue --gpu 0-0
```

The experimental results are obtained by training on one NVIDIA RTX 3090.

## Acknowledgement
The code is developed upon [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE). Many thanks to their contributions.
# NeuFace: A Large-Scale 3D Face Mesh Video Dataset via Neural Re-parameterized Optimization (ICLR 2025 / TMLR 2024)
### [Project Page](https://kim-youwang.github.io/neuface) | [Paper](https://openreview.net/forum?id=zVDMh6JvWc)
This repository contains the official implementation of the ICLR 2025 / TMLR 2024 paper, 
"NeuFace: A Large-Scale 3D Face Mesh Video Dataset via Neural Re-parameterized Optimization."

![Dataset Sample](assets/dataset_sample.gif)

NeuFace is a novel 3D face tracking pipeline that leverages deep network prior via neural 3DMM re-parameterization. Please refer to our [paper](https://openreview.net/forum?id=zVDMh6JvWc) for the details.

In this codebase, we provide FLAME annotation (tracking) scripts for MEAD and CelebV-HQ datasets.

## Environment setup
This code was developed on Ubuntu 18.04 with Python 3.10, CUDA 11.3 and PyTorch 1.12.0, using NVIDIA RTX A6000 (48GB) GPU. Later versions should work, but have not been tested.

1. Set up the virtual environment
```bash
pip install -r requirements.txt
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1120/download.html
# re-install numpy to specific version
pip install numpy==1.23.1 
```

2. Prepare assets 
- Prepare FLAME-related data by running this script:
```bash
sh fetch_data.sh
```
- Follow the [instructions](https://github.com/TimoBolkart/BFM_to_FLAME?tab=readme-ov-file#create-texture-model) to get `FLAME_albedo_from_BFM.npz`, put it into `./data`.

## Download Raw Videos
We do not provide the raw videos or images of the original datasets (MEAD, CelebV-HQ). Instead, we provide codes and scripts to annotate FLAME parameters for the videos. Thus, you first need to download your preferred video datasets.
1. **Set Up Directories**

Set your preferred directory as `$DATA_DIR`. Since we save tracked FLAME parameters and, optionally, rendered mesh images, you may need to set up a directory on a disk with sufficient storage.
```bash
mkdir ./data && export DATA_DIR=./data
```

2. **Download Datasets** 
- MEAD dataset: [Link](https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j)
- CelebV-HQ dataset: [Link](https://github.com/CelebV-HQ/CelebV-HQ/issues/8#issue-1336655726) 

After downloading the datasets, please make sure you have a `$DATA_DIR` structure as below: 
```bash
$DATA_DIR
├── MEAD
│   ├── W024
│   │   ├── videos
│   │   │   ├── down
│   │   │   ├── front
│   │   │   └── ...
│   ...
├── CelebV_HQ
│   ├── videos
│   │   ├── LEq4-b61Hoo_3.mp4
│   │   ├── A5R-wZbzLEA_3.mp4    
│   │   └── ...    
...
``` 


## Run NeuFace Optimization
1. **Extract Frames from Videos**
- For MEAD dataset, run this script to extract frames. Currently, the script is for the MEAD ID W024. To process other IDs, please modify the [INPUT_DIR](https://github.com/kaist-ami/NeuFace/blob/9d35981421fdc554309126aa36d2a584738500c0/extract_mead_frames.sh#L5) and [OUTPUT_DIR](https://github.com/kaist-ami/NeuFace/blob/9d35981421fdc554309126aa36d2a584738500c0/extract_mead_frames.sh#L6) accordingly. 
```bash
# This shell script processes W024 ID in MEAD dataset. 
# Please change the INPUT_DIR and OUTPUT_DIR in the script, if you process different IDs. 
chmod +x ./extract_mead_frames.sh
# ./extract_mead_frames.sh                    # process all emotion, level and videos for ID W024
# ./extract_mead_frames.sh angry              # process all 'angry' videos for ID W024
# ./extract_mead_frames.sh angry level_1      # process all 'angry' 'level_1' videos for ID W024
./extract_mead_frames.sh angry level_1 010    # process all 'angry' 'level_1' '010.mp4' for ID W024
```
- For CelebV-HQ dataset, run this script to extract frames.
```bash
chmod +x ./extract_celebv_frames.sh
./extract_celebv_frames.sh
```

- After extracting all the frames, the `$DATA_DIR` should look like this:
```bash
$DATA_DIR
├── MEAD
│   ├── W024
│   │   ├── videos
│   │   ├── images
│   │   │   ├── angry
│   │   │   │   ├── level_1
│   │   │   │   │   ├── 010
│   │   │   │   │   │   ├── img_0000_down.jpg
│   │   │   │   │   │   ├── img_0000_front.jpg
│   │   │   │   │   │   └── ...
│   └── ...
├── CelebV_HQ
│   ├── videos
│   ├── images
│   │   ├── LEq4-b61Hoo_3
│   │   │   ├── LEq4-b61Hoo_3_frame0000.jpg
│   │   │   ├── LEq4-b61Hoo_3_frame0001.jpg
│   │   │   └── ...
│   │   ├── A5R-wZbzLEA_3   
│   │   │   ├── A5R-wZbzLEA_3_frame0000.jpg
│   │   │   ├── A5R-wZbzLEA_3_frame0001.jpg
│   │   │   └── ...
│   │   └── ...
``` 


2. **Run Optimization for a Single Video**
After downloading and extracting video frames, run this command to run NeuFace optimization for a single video:
```bash
# for MEAD sequence
python ./neuface_optim.py --cfg configs/neuface_mead.yml --test_seq_path $DATA_DIR/MEAD/W024/images/angry/level_1/010
# for CelebV-HQ sequence
python ./neuface_optim.py --cfg configs/neuface_celebv.yml --test_seq_path $DATA_DIR/celebv/
```

3. **Run Optimization for all the Videos**
- To process all the videos in MEAD dataset, make sure you downloaded all the videos. Then, run this script
```bash
chmod +x ./mead_construct_neuface_dataset.sh
./mead_construct_neuface_dataset.sh
```
You'll need to add your preferred IDs [in the script]().
- To process all the videos in CelebV-HQ dataset, make sure you downloaded all the videos. Then, run this script:
```bash
chmod +x ./celebv_construct_neuface_dataset.sh
./celebv_construct_neuface_dataset.sh
```
Note that these scripts will take long time to finish, as they run optimizations for all the videos in the datasets. For other datasets, you can write similar scripts to extract frames and run the optimization.


## Contact
Kim Youwang (youwang.kim@postech.ac.kr)

## Acknowledgement
The implementation of NeuFace is largely inspired by the seminal projects.
We would like to express our sincere gratitude to the authors for making their code public.
- DECA (https://github.com/yfeng95/DECA)
- Exemplar Fine-Tuning (https://github.com/facebookresearch/eft)

## Citation
If you find our code or paper helps, please consider citing:
````BibTeX
@article{youwang2024neuface,
  author    = {Kim Youwang and Lee Hyun and Kim Sung-Bin and Suekyeong Nam and Janghoon Ju and Tae-Hyun Oh},
  title     = {A Large-Scale 3D Face Mesh Video Dataset via Neural Re-parameterized Optimization},
  journal   = {Transactions on Machine Learning Research},
  year      = {2024},
}
```
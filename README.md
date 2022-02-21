# Inpainting Forensics

An official implementation code for paper "IID-Net: Image Inpainting Detection via Neural Architecture Search and Attention"

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Usage](#usage)
- [Citation](#citation)


## Background
In this work, we tackle the challenge of providing a forensic solution that can generalize well to accurately detect various unseen inpainting manipulations. More specifically, we propose a novel end-to-end Image Inpainting Detection Network (IID-Net), to detect the inpainted regions at pixel accuracy. The schematic diagram of the IID-Net is shown in the following figure.

<p align='center'>  
  <img src='https://github.com/HighwayWu/InpaintingForensics/blob/master/imgs/framework.jpg' width='870'/>
</p>
<p align='center'>  
  <em>Framework of IID-Net. The upper part shows how to generate images for training, while for inference, by directly using an image as input, the detection result can be obtained.</em>
</p>

The proposed IID-Net consists of three sub-blocks: the enhancement block, the extraction block and the decision block. Specifically, the enhancement block aims to enhance the inpainting traces by using hierarchically combined input layers. The extraction block, automatically designed by Neural Architecture Search (NAS) algorithm, is targeted to extract features for the actual inpainting detection tasks. In order to further optimize the extracted latent features, we integrate global and local attention modules in the decision block, where the global attention reduces the intra-class differences by measuring the similarity of global features, while the local attention strengthens the consistency of local features.

An example of the detection result of IID-Net is shown in the below figure. Here we would like to emphasize that none of the original images in the demo or the corresponding
inpainting methods were involved during the training of IID-Net.

<p align='center'>
  <img src='https://github.com/HighwayWu/InpaintingForensics/blob/master/imgs/demo.png' width='870'/>
</p>
<p align='center'>  
  <em>Demo figures. In each pair, the left is the original image; the middle is the forged image where the key objects are removed by some inpainting methods; the right is the output of IID-Net by using forged image as input.</em>
</p>

Furthermore, we thoroughly study the generalizability of our IID-Net, and find that different training data could result in vastly different generalization capability. By carefully examining 10 popular inpainting methods, we identify that the IID-Net trained on only one specific deep inpainting method exhibits desirable generalizability; namely, the obtained IID-Net can accurately detect and localize inpainting manipulations for various unseen inpainting methods as well. Our results would suggest that common artifacts are shared across diverse image inpainting methods. Finally, we build a diverse inpainting dataset of 10K image pairs for the future research in this area.

## Dependency
- torch 1.6.0
- tensorflow 1.8.0

## Usage

To train or test the IID-Net:
```bash
python main.py {train,test}
```

For example to test the IID-Net:
```bash
python main.py test
```
Then the IID-Net will detect the images in the `./demo_input/` and save the results in the `./demo_output/` directory.

**Note 1: The training dataset (Dresden) is released on:
[Google Drive](https://drive.google.com/file/d/1crJnKMvjF3P6rqNFZks4PuQAz83nE_g-/view?usp=sharing) or 
[Baidu Yun (Code: rpag)](https://pan.baidu.com/s/1GGUqMOS-VSBd0ybm9leOPg);
The training dataset (Places) is released on:
[Google Drive](https://drive.google.com/file/d/1iGxScWk_O745ojUMD-jdJXelqZiWhhPu/view?usp=sharing) or 
[Baidu Yun (Code: qcbt)](https://pan.baidu.com/s/1qmD0NUZjEh1651rkZs9O1w);**

**Note 2: The diverse inpainting dataset (testing dataset) dataset can be downloaded from:
[Google Drive](https://drive.google.com/file/d/1prC20Ux7pKwWYw8EfLQV2bXE6dk4xLJ8/view?usp=sharing) or 
[Baidu Yun (Code: biva)](https://pan.baidu.com/s/162pm40PEN-8kzbybLf--7A)**

**Note 3: The test environment requires at least 2 GPUs, as I found that using only 1 GPU would lead to totally different results (may be caused by DataParallel).**

## Citation
If you use this code for your research, please cite our paper
```
@ARTICLE{9410590, 
author={Wu, Haiwei and Zhou, Jiantao},
journal={IEEE Transactions on Circuits and Systems for Video Technology},
title={IID-Net: Image Inpainting Detection Network via Neural Architecture Search and Attention},
year={2021},
volume={},
number={},
pages={1-1},
doi={10.1109/TCSVT.2021.3075039}}

@INPROCEEDINGS{9506778,
author={Wu, Haiwei and Zhou, Jiantao},
booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
title={GIID-NET: Generalizable Image Inpainting Detection Network}, 
year={2021},
volume={},
number={},
pages={3867-3871},
doi={10.1109/ICIP42928.2021.9506778}}
```

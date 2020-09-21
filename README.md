# Inpainting Forensics

(updating)

An official implementation code for paper "GIID-Net: Generalizable Image Inpainting Detection via Neural Architecture Search and Attention"

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Demo](#demo)


## Background
In this work, we tackle the challenge of providing a forensic solution that can generalize well to accurately detect various unseen inpainting manipulations. More specifically, we
propose a novel end-to-end Generalizable Image Inpainting Detection Network (GIID-Net), to detect the inpainted regions at pixel accuracy. The schematic diagram of the GIID-Net is shown in the following figure.

<p align='center'>  
  <img src='https://github.com/HighwayWu/InpaintingForensics/blob/master/imgs/framework.jpg' width='870'/>
  Framework of GIID-Net. The upper part shows how to generate images for training, while for inference, by directly using an image as input, the detection result can be obtained.
</p>

The proposed GIID-Net consists of three sub-blocks: the enhancement block, the extraction block and the decision block. Specifically, the enhancement block aims to enhance the inpainting traces by using hierarchically combined input layers. The extraction block, automatically designed by Neural Architecture Search (NAS) algorithm, is targeted to extract features for the actual inpainting detection tasks. In order to further optimize the extracted latent features, we integrate global and local attention modules in the decision block, where the global attention reduces the intra-class differences by measuring the similarity of global features, while the local attention strengthens the consistency of local features.

An example of the detection result of GIID-Net is shown in the below figure. Here we would like to emphasize that none of the original images in the demo or the corresponding
inpainting methods were involved during the training of GIID-Net.

<p align='center'>
  <img src='https://github.com/HighwayWu/InpaintingForensics/blob/master/imgs/demo.png' width='870'/>
  Framework of SLI model.
</p>

## Dependency
- torch 1.6.0
- tensorflow 1.8.0

## Demo

To train or test the GIID-Net:
```bash
python main.py {train,test}
```

For example to test the GIID-Net:
```bash
python main.py --model test
```
Then the GIID-Net will detect the images in the `./demo_input/` and save the results in the `./demo_output/` directory.
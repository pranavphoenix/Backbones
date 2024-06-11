# Clash of Backbones
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/which-backbone-to-use-a-resource-efficient/breast-cancer-histology-image-classification)](https://paperswithcode.com/sota/breast-cancer-histology-image-classification?p=which-backbone-to-use-a-resource-efficient) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/which-backbone-to-use-a-resource-efficient/image-classification-on-plantvillage)](https://paperswithcode.com/sota/image-classification-on-plantvillage?p=which-backbone-to-use-a-resource-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/which-backbone-to-use-a-resource-efficient/image-classification-on-eurosat)](https://paperswithcode.com/sota/image-classification-on-eurosat?p=which-backbone-to-use-a-resource-efficient)

## Which Pytorch Backbone to Use for Low Data Fine-tuning? 
### Resource-efficient Image Classification

#### Criteria for choosing backbones for our experiments?
- ImageNet-1k pre-trained weights available in torchvision or github (WaveMix)
- Number of parameters less than 30 M
- Only model with highest ImageNet performance from one architecture family

### Backbones Compared

| Architecture          | # Params (M) | ImageNet-1k Top-1 Accuracy (%) |
|-----------------------|--------------|---------------------------------|
| ResNet-50             | 25.6         | 76.13                           |
| WaveMix               | 27.9         | 75.32                           |
| ConvNeXt-Tiny         | 28.6         | 82.52                           |
| Swin-Tiny             | 28.3         | 81.47                           |
| SwinV2-Tiny           | 28.4         | 82.07                           |
| EfficientNetV2-S      | 21.5         | 84.23                           |
| DenseNet-161          | 28.7         | 77.14                           |
| MobileNetV3-Large     | 5.5          | 75.27                           |
| RegNetY-3.2GF         | 19.4         | 81.98                           |
| ResNeXt-50 32×4d      | 25.0         | 81.20                           |
| ShuffleNetV2 2.0×     | 7.4          | 76.23                           |


### Datasets used for benchmarking

| Dataset              | Domain                                   | # Training Images | # Testing Images | # Classes |
|----------------------|------------------------------------------|-------------------|------------------|-----------|
| CIFAR-10             | 🖼️ Natural Images                       | 50,000            | 10,000           | 10        |
| CIFAR-100            | 🖼️ Natural Images                       | 50,000            | 10,000           | 100       |
| TinyImageNet         | 🖼️ Natural Images (ImageNet subset)     | 100,000           | 10,000           | 200       |
| Stanford Dogs        | 🖼️ Natural Images (Dog breeds)          | 12,000            | 8,580            | 120       |
| Flowers-102          | 🖼️ Natural Images (Flower species)      | 2,040             | 6,149            | 102       |
| CUB-200-2011         | 🖼️ Natural Images (Bird species)        | 5,994             | 5,794            | 200       |
| Stanford Cars        | 🖼️ Natural Images (Car models)          | 8,144             | 8,041            | 196       |
| Food-101             | 🖼️ Natural Images (Food categories)     | 75,750            | 25,250           | 101       |
| DTD                  | 🎨 Texture Images                       | 1,880             | 1,880            | 47        |
| UCMerced Land Use    | 🛰️ Remote Sensing Images                | 1,680             | 420              | 21        |
| EuroSAT              | 🛰️ Remote Sensing Images                | 18,900            | 8,100            | 10        |
| PlantVillage         | 🌿 Plant Images                         | 44,343            | 11,105           | 39        |
| PlantCLEF            | 🌿 Plant Images                         | 10,455            | 1,135            | 20        |
| Galaxy10 DECals      | 🌌 Astronomy Images (Galaxy Morphology) | 15,962            | 1,774            | 10        |
| BreakHis 40×         | 🏥 Medical Images (Histopathology)      | 1,398             | 606              | 2         |
| BreakHis 100×        | 🏥 Medical Images (Histopathology)      | 1,458             | 632              | 2         |
| BreakHis 200×        | 🏥 Medical Images (Histopathology)      | 1,411             | 611              | 2         |
| BreakHis 400×        | 🏥 Medical Images (Histopathology)      | 1,276             | 553              | 2         |


### Code to run benchmarking for each dataset 

```bash
python <dataset.py> -model <backbone> -bs <batch-size> 
```
If you want to create a train test split
```bash
python split.py --input_dir <input folder path> --output_dir <output folder path> --test_size <fraction to be split>  
```


#### Cite this paper
```
@misc{jeevan2024backbone,
      title={Which Backbone to Use: A Resource-efficient Domain Specific Comparison for Computer Vision}, 
      author={Pranav Jeevan and Amit Sethi},
      year={2024},
      eprint={2406.05612},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

# 4SwithODandDE

> Self-Supervised Semantic Segmentation with Object Detection and Depth Estimation  
> Aro Yun
---

## Installation

### Using mamba (fastest):

```bash
# Install micromamba
curl micro.mamba.pm/install.sh | bash

# Activate environment
micromamba env create --file environment.yaml
micromamba activate 4S
```

### Using conda:

```bash
conda env create --file environment.yaml
conda activate 4S
```

---

## Usage

### arguments

+ --source: Input source
+ --od: Object detection model (default : yolov8x)
+ --de: Depth estimation model (default : ZoeD_NK)
+ --save: Save results as images.

### Example

```bash
python main.py --source assets/Puppies.jpg --save
```

---

## Acknowledgement

Ultralytics [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)    
ZoeDepth [https://github.com/isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth)    

# Adaptive query selection for RT-DETRv2
## Quick start

<details >
<summary>Setup</summary>

```shell
pip install -r requirements.txt
```

The following is the corresponding `torch` and `torchvision` versions.
`rtdetr` | `torch` | `torchvision`
|---|---|---|
| `-` | `2.4` | `0.19` |
| `-` | `2.2` | `0.17` |
| `-` | `2.1` | `0.16` |
| `-` | `2.0` | `0.15` |

</details>


## Download pretrained weights

| Model | Dataset | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS | config| checkpoint | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
**RT-DETRv2-S** | COCO | 640 | **48.1** <font color=green>(+1.6)</font> | **65.1** | 20 | 217 | [config](./configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth) |
**RT-DETRv2-M**<sup>*<sup> | COCO | 640 | **49.9** <font color=green>(+1.0)</font> | **67.5** | 31 | 161 | [config](./configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth)
**RT-DETRv2-M** | COCO | 640 | **51.9** <font color=green>(+0.6)</font> | **69.9** | 36 | 145 | [config](./configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth)
**RT-DETRv2-L** | COCO | 640 | **53.4** <font color=green>(+0.3)</font> | **71.6** | 42 | 108 | [config](./configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth)
**RT-DETRv2-X** | COCO | 640 | 54.3 | **72.8** <font color=green>(+0.1)</font> | 76 | 74 | [config](./configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth)
<!-- rtdetrv2_hgnetv2_l | COCO | 640 | 52.9 | 71.5 | 32 | 114 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_l_6x_coco_from_paddle.pth) 
rtdetrv2_hgnetv2_x | COCO | 640 | 54.7 | 72.9 | 67 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_x_6x_coco_from_paddle.pth) 
rtdetrv2_hgnetv2_h | COCO | 640 | 56.3 | 74.8 | 123 | 40 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_h_6x_coco_from_paddle.pth) 
rtdetrv2_18vd | COCO+Objects365 | 640 | 49.0 | 66.5 | 20 | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_5x_coco_objects365_from_paddle.pth)
rtdetrv2_r50vd | COCO+Objects365 | 640 | 55.2 | 73.4 | 42 | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_2x_coco_objects365_from_paddle.pth)
rtdetrv2_r101vd | COCO+Objects365 | 640 | 56.2 | 74.5 | 76 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_2x_coco_objects365_from_paddle.pth)
 -->

**Notes:**
- `AP` is evaluated on *MSCOCO val2017* dataset.
- `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT>=8.5.1$.
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.



## Usage
<details>

1. Data

- Download and extract COCO 2017 train and val images.
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
- Modify config [`img_folder`, `ann_file`](configs/dataset/coco_detection.yml)

2. Adaptive hyperparameters

- Modify adaptive hyperparameters in [rtdetrv2_r50vd.yml](configs/rtdetrv2/include/rtdetrv2_r50vd.yml)


3. Evaluating
```shell
python tools/train.py -c path/to/config -r path/to/checkpoint --test-only
```

<!-- <summary>4. Export onnx </summary> -->
4. Export onnx
```shell
python tools/export_onnx.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml -r scheckpoint/rtdetrv2_r50vd_6x_coco_ema.pth --check
```

<!-- <summary>5. Export tensorrt </summary> -->
5. Export tensorrt
```shell
python tools/export_trt.py -i path/to/onnxfile
```


</details>


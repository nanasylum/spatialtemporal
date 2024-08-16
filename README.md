# Spatial-temporal Consistency Constraint for Depth and Ego-motion Estimation of Laparoscopic Images

<!-- ![Image](https://github.com/BeileiCui/EndoDAC/blob/main/assets/main.jpg) -->
![Image](https://github.com/nanasylum/spatialtemporal/blob/main/img/our_work.png)\
![Image](./img/our_work.png)\
our network architecture is:\
![Image](https://github.com/nanasylum/spatialtemporal/blob/main/img/arch.jpg)


<!-- ### [__[arxiv]__](http://arxiv.org/abs/2405.08672) -->

<!-- * 2024-05-14 Our paper has been early accepted (top 11%) by MICCAI 2024!
* 2024-05-15 arxiv version is online. -->

## Abstract
Estimating depth and ego-motion are crucial tasks for laparoscopic navigation and robotic-assisted surgery. Most current self-supervised methods involve warping one frame onto an adjacent frame using the estimated depth and camera pose. The photometric loss between the estimated and original frames then serves as the training signal. However, these methods encounter major challenges due to non-Lambertian reflection regions and the textureless surfaces of organs, leading to significant performance degradation and scale ambiguity in monocular depth estimation. In this paper, we introduce a network that predicts depth and ego-motion using spatial-temporal consistency constraints. Spatial consistency is derived from the left and right views of the stereo laparoscopic image pairs, while temporal consistency comes from consecutive frames. To enhance the understanding of semantic information in surgical scenes, we employ the Swin Transformer as the encoder and decoder for depth estimation, due to its superior semantic segmentation capabilities. To address issues of illumination variance and scale ambiguity, we incorporate a SIFT loss term to eliminate oversaturated regions in laparoscopic images. Our method is evaluated on the SCARED dataset and shows remarkable results. 

## Results

| Method | Year | Abs Rel | Sq Rel | RMSE | RMSE log | &delta; |
|  :----:  | :----:  | :----:   |  :----:  | :----:  | :----:  | :----:  | 
| Fang et al. | 2020 | 0.078 |	0.794 |	6.794 |	0.109 |	0.946 |
| Endo-SfM | 2021 | 0.062 |	0.606 |	5.726 |	0.093 |	0.957 |
| AF-SfMLeaner | 2022 | 0.059 |	0.435 |	4.925 |	0.082 |	0.974 |
| Yang et al. | 2024 | 0.062 |	0.558 |	5.585 |	0.090 |	0.962 |
|__Ours__ | | __0.057__ |	__0.436__ |	__4.972__ |	__0.081__ |	__0.972__ | 

## Initialization


Install required dependencies with pip:
```
pip install -r requirements.txt
```

Download pretrained model from: [depth_anything_vitb14](https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll/view?usp=sharing). Create a folder named ```pretrained_model``` in this repo and place the downloaded model in it.

## Dataset
### SCARED
Please follow [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) to prepare the SCARED dataset.

## Utilization

### Training
```
CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py --data_path <your_data_path> --log_dir './logs'
```

### Evaluation

Export ground truth depth and pose before evaluation:
```
python export_gt_depth.py --data_path PATH_TO_YOUR_DATA --split endovis
```
```
python export_gt_pose.py --data_path PATH_TO_YOUR_DATA --split endovis --sequence YOUR_SEQUENCE
```

If you want to evaluate your model:
```
python evaluate_depth.py --data_path PATH_TO_YOUR_DATA --load_weights_folder PATH_TO_YOUR_MODEL --eval_mono
```

If you want to evaluate your model:
```
python evaluate_pose.py --data_path PATH_TO_YOUR_DATA --load_weights_folder PATH_TO_YOUR_MODEL --eval_mono
```
If you want to generate your depthmap:
```
python generate_pred.py
```
If you want to generate point cloud map:
```
python generate_pred_nocolor.py
```
```
cd depth2pointcloud
```
```
python generate_depthmap.py
```
```
python generate_pc_rgb.py
```

## Acknowledgment
Our code is based on the implementation of [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything). We thank their excellent works.



## Rotation Adaptive Visual Object Tracking with Motion Consistency
- - - -
WACV published Paper : <https://www.computer.org/csdl/proceedings/wacv/2018/4886/00/488601b047.pdf>

The code in this repository is based on the work originally done by Luca Bertinetto et al. in "Fully-Convolutional Siamese Networks for Object Tracking". We thank Luca Bertinetto and his co-authors for making their code publicly available to all researchers, which helped us a lot in contributing to visual object tracking community. The code in this repository enables you to reproduce the experimental results of our paper. In contrast to their approach, the proposed contributions have been clearly highlighted in our [WACV paper](https://www.computer.org/csdl/proceedings/wacv/2018/4886/00/488601b047.pdf). For successful execution of our tracker `SiameseFC-DSR`, please follow the instructions given below.
- - - -
SiameseFC-DSR
- - - -

![image1](modSiam1.png "SiameseFC-DSR")
- - - -
CFNet-DSR
- - - -
![image2](https://gitlab.com/LituRout/SiameseFC-DSR-master/blob/master/modCF1.png "CFNet-DSR")
- - - -
If you find our work useful, please cite:
```
@INPROCEEDINGS{rout2018rotation,
	title={Rotation Adaptive Visual Object Tracking with Motion Consistency}, 
	author={Rout, Litu and Sidhartha and Manyam, Gorthi RKSS and Mishra, Deepak}, 
	booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
	year={2018},
	pages={1047-1055},
	month={March}}
```
- - - -

[ **Tracking only** ] If you don't care much about training, simply plug one of our pretrained networks to our basic tracker and see it in action.
  1. Prerequisites: GPU, CUDA drivers, [cuDNN](https://developer.nvidia.com/cudnn), Matlab (we used 2016b), [MatConvNet](http://www.vlfeat.org/matconvnet/install/) (we used `v1.0-beta24`).
  2. Clone the repository.
  3. Download one of the pretrained networks from <http://www.robots.ox.ac.uk/~luca/siamese-fc.html>
  4. Go to `SiameseFC-DSR/tracking/` and edit `env_paths_tracking.m`, `startup.m` appropriately.
  5. Be sure to have at least one video sequence in the appropriate format. You can find an example here in the repository (`SiameseFC-DSR/demo-sequences/vot15_bag`).
  6. `SiameseFC-DSR/tracking/tracker.m` is the entry point to execute the tracker, have fun! Alternatively, one can edit `run_tracker.m` to call the `tracker.m` function and execute the tracker.

 [ **Training and tracking** ] We have not trained our model from scratch, therefore we request the reader to follow the instructions of original SiameseFC [training and tracking] (https://github.com/bertinetto/siamese-fc) in order to train the model.

Note: The hyper parameters in the provided code may not be exactly what we have used for our evaluation. So we request the reader to go through our paper to use the exact hyper parameter settings. However, the provided code is suitable for understanding the key contributions of the paper.
`This work can only be used for research purposes. For commercial use of this work, please contact the authors.`

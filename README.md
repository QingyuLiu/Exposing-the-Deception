# Exposing the Deception: Uncovering More Forgery Clues for Deepfake Detection
![](https://github.com/QingyuLiu/Exposing-the-Deception/blob/main/framework.png)
This repo is the official implementation of “Exposing the Deception: Uncovering More Forgery Clues for Deepfake Detection”. Accepted by AAAI-2024.

## Installation
Our code is implemented and evaluated on pytorch. The following packages are used by our code.
- `torch==2.0.1`
- `albumentations==1.3.1`
- `opencv-python==4.8.1.78`
- `scipy==1.10.1`
- `tensorboard==2.14.0`
- `numpy==1.24.3`
- `tqdm==4.66.1`

Our code is evaluated on `Python 3.8.11` and `CUDA 11.7`.


## Training
### Prepare Datasets
- Prepare face forgery datasets: [FaceForensics++](https://github.com/ondyari/FaceForensics), [Celeb-DF-V1](https://github.com/yuezunli/celeb-deepfakeforensics), [Celeb-DF-V2](https://github.com/yuezunli/celeb-deepfakeforensics), [DFDC-Preview](https://ai.meta.com/datasets/dfdc/), [DFDC](https://www.kaggle.com/c/deepfake-detection-challenge/data)
- Preprocess the video: extract frames from videos, and then extract facial images using [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface). 
To train or test the model, you should provide a dataset path and label txt, which need to have the following folder structure. 
```Shell
dataset
|-- FF++ < dataset name >
    |-- train_fake.txt < data line: label,path,the number of frames\n >
    |-- train_real.txt
    |-- val_fake.txt
    |-- val_real.txt
    |-- test_fake.txt
    |-- test_real.txt
|-- Celeb-DF-V1
    |-- ...
|-- ...
```

### Train Models
After preprocessing datasets, you can detect anomalies with various settings using the following command:

```
python training.py  --name                                      \

                    (arguments for training)
                    --gpu_num 0,1                                                   \
                    --model resnet,efficientnet,mobilenet                           \
                    --epoch 20                                                      \
                    --weight_decay 1e-6                                             \ 
                    --lr 1e-3                                                       \
                    --bs 256                                                        \
                    --test_bs 1000                                                  \
                    --num_workers 12                                                \
                    --size 224                                                      \
                    --dataset FF++_c23,Celeb-DF-v2,Celeb-DF-v1,DFDC-Preview,DFDC    \
                    --mixup True                                                    \
                    --alpha 0.5                                                     \

                    (arguments for loss)
                    --lil_loss True                             \
                    --gil_loss True                             \
                    --temperature 1.5                           \
                    --mi_calculator kl                          \
                    --balance_loss_method auto,hyper            \
                    --scales [1,2,10]                           \

                    (model parameters)
                    --num_LIBs 4                                \
                    --resume_model output/{name}/....           \

                    (checkpoint)
                    --test False                                \
                    --save_model True                           \
                    --save_path output                          \
```
The following is a description of some parameters in the configuration file:
- `model`: the backbone type of LIB, which supports resnet, efficientnet, mobilenet.
- `dataset`: is the dataset name, which corresponds to `dataset/{dataset_name}` path.
- `lil_loss`, `gil_loss`: `True` is to use Local Information Loss or Gocal Information Loss proposed by our work.
- `mi_calculator`: is the algorithm to calculate the mutual information, which supports KL divergence and Wasserstein distance.
- `balance_loss_method`: is the method for determining $\alpha$ and $\beta$ in equation (13) of the paper, and supports auto and hyper. If it is hyper, "scales" is the setting of weights, which respectively represent the weights of classification loss, local information loss and Gocal information loss.
- `num_LIBs`: is the number of Local Information Block.


## Citation
Coming soon...

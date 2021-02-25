# Robotic_Image_Segmentation_Architecture
人が正解を与えることなく物体の位置とその輪郭を検出し，区別することを行うRobotic Image Segmentation Architectureのリポジトリ．
本手法は，カメラとそれを駆動させるアクチュエータ(パン・チルト軸)で構成されるロボットを用いて，Image Segmentationを行う．
ロボットは深層強化学習によって最適化された行動をし，その画像差分から物体の位置と輪郭を検出する．
以下に，各パッケージの説明とその使い方を示す．

## robotic_img_seg_archit
Robotic_Image_Segmentation_Architectureのメインとなるパッケージ．
深層強化学習(DQN)のプログラムや画像処理(色やラベル付与)のプログラムが入ってる．

### How to use
```
# azure KinectとDynamixelサーボモータを起動させる．
$ roslaunch robotic_img_seg_archit robotic_img_seg_archit.launch

# 深層強化学習によって，Image Segmentationのための最適な行動を獲得する．
$ rosrun robotic_img_seg_archit create_learning_model.py

# 学習したエージェントを読み込み，使用する．
$ rosrun robotic_img_seg_archit load_learning_model.py
```


## pantilt_control
Robotic_Image_Segmentation_Architectureで用いるロボットを制御するパッケージ．
DynamixelサーボモータのSDKを用いた制御プログラムが入っている．


## calculation_mean_IoU
Image Segmentationの精度を評価するMean IoUを計算するパッケージ．
Mean IoUを計算するためのプログラムが入っている.

### How to use
```
# プログラムのある階層まで移動
$ cd calculation_mean_IoU/src

# Mean IoUを計算する
$ python3 calculation_mean_IoU.py
```

## azure_kinect_ros
azure_kinectをROSで動かすためのドライバとなるパッケージ．


## sobit_common
DynamixelのSDKやROSで用いるライブラリがまとまったパッケージ．
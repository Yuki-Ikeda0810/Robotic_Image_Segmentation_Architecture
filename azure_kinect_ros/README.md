# azure_kinect_ros
azure_kinect_rosのSDKとドライバーのリポジトリです。

## Install & Setup

```
cd ~/catkin_ws/src/
git clone https://gitlab.com/TeamSOBITS/azure_kinect_ros.git
cd azure_kinect_ros/
bash install.sh  # install する際はインタラクティブなやり取りがあるので注意
sudo reboot	#再起動
```
インタラクティブなやり取り
```
# jackedの説明
<ok>
# リアルタイム優先の設定
<yes>
# Do you accept the EULA license terms?
<yes>
```

## How To Use
k4aviewer
ROS無しでAzure Kinectを扱うプログラム
```
cd ~/catkin_ws/src/azure_kinect_ros/Azure-Kinect-Sensor-SDK/build/bin/
./k4aviewer
```

ROS
```bash
roslaunch azure_kinect_ros_driver driver.launch.
```

### Topic

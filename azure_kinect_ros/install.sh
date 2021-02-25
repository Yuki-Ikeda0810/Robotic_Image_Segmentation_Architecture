#!/bin/bash

###Azure-Kinect-Sensor-SDK セットアップ
cd
#プログラムインストール
sudo apt-get update
sudo apt-get install -y  \
    ninja-build \
    doxygen \
    clang \
    gcc-multilib-arm-linux-gnueabihf \
    g++-multilib-arm-linux-gnueabihf
sudo apt-get update
sudo apt-get install -y \
    freeglut3-dev \
    libgl1-mesa-dev \
    mesa-common-dev \
    libsoundio-dev \
    libvulkan-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    uuid-dev \
    libsdl2-dev \
    usbutils \
    libusb-1.0-0-dev \
    openssl \
    libssl-dev \
    wget \
    git \
    expect \
    software-properties-common

sudo apt-get update
sudo apt-get install -y jack-tools
# expect "no"command (修正中)
#expect -c "
#set timeout 10
#spawn sudo apt-get install -y jack-tools
#expect \"Configuring jackd2\"
#send \"ok\n\"
#expect \"リアルタイム実行優先度の設定を有効にしますか?\"
#send \"yes\n\"
#expect \"\\\\$\"
#exit 0
#"

# expect Press [ENTER] key
expect -c "
set timeout 10
spawn sudo add-apt-repository ppa:ubuntu-toolchain-r/test
expect \"Press \[ENTER\] to continue or Ctrl-c to cancel adding it.\"
send \"\n\"
expect \"\\\\$\"
exit 0
"

sudo apt-get update
sudo apt-get -y install gcc-4.8
echo 'deb http://dk.archive.ubuntu.com/ubuntu/ xenial main' | sudo tee /etc/apt/source.list
echo 'deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe' | sudo tee /etc/apt/source.list
sudo apt-get update
#sudo apt-get -y install gcc-4.9
sudo apt-get -y install libstdc++6

# install Azure Kinect SDK following https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
sudo apt update
sudo apt install -y libk4a1.3 libk4a1.3-dev k4a-tools=1.3.0
# expect "yes"command
#expect -c "
#set timeout 10
#spawn sudo apt install -y libk4a1.3 libk4a1.3-dev k4a-tools=1.3.0
#expect \"Do you accept the EULA license terms?\"
#send \"yes\n\"
#expect \"\\\\$\"
#exit 0
#"



#依存関係を設定
cd ~/catkin_ws/src/azure_kinect_ros/Azure-Kinect-Sensor-SDK/scripts/
sudo ./bootstrap-ubuntu.sh
sudo cp 99-k4a.rules /etc/udev/rules.d/
sudo cp libdepthengine.so.2.0 /usr/lib/x86_64-linux-gnu/
sudo cp libdepthengine.so.2.0 /lib/x86_64-linux-gnu/
sudo chmod a+rwx /usr/lib/x86_64-linux-gnu/
sudo chmod a+rwx -R /lib/x86_64-linux-gnu/
sudo chmod a+rwx /etc/udev/rules.d/99-k4a.rules
chmod a+rwx -R ~/catkin_ws/src/azure_kinect_ros/Azure-Kinect-Sensor-SDK/build/bin/

#cmakeをアップグレード
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v3.14/cmake-3.14.5.tar.gz
tar -xzvf cmake-3.14.5.tar.gz
cd cmake-3.14.5/
./bootstrap
make -j8
sudo make install

#cmake
cd ~/catkin_ws/src/azure_kinect_ros/Azure-Kinect-Sensor-SDK/
mkdir -p build
cd build/
cmake .. -GNinja
ninja
cmake .. -GNinja
ninja
sudo ninja install


###Azure_Kinect_ROS_Driver セットアップ
cd
#プログラムインストール
sudo apt-get update
sudo apt-get -y install gcc-7
sudo apt-get -y install g++-7

#バージョン設定
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
sudo update-alternatives --config gcc
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100
sudo update-alternatives --config g++

#catkin_make
cd ~/catkin_ws && catkin_make
catkin_make install
source ./devel/setup.bash

#再起動
#sudo reboot
echo "------再起動してください-------"

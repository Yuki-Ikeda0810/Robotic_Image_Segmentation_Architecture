#!/usr/bin/env python3
# coding: utf-8

# ROS
import rospy

# 画像処理
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# 基本
import math
import random
import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定義した画像処理のクラスや関数を読み込む
import include.image_processing as img


########################################################################
# 初期設定

rospy.init_node("load_learning_model")
rospy.sleep(3)

# 画像を取得するクラスの初期設定
get_image = img.GetImage()
sub = rospy.Subscriber("/rgb/image_raw", Image, get_image.callback)
rospy.sleep(1)

# deviceの設定(GPUを使う場合)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


########################################################################
# Qネットワークの定義

# ここでは、現在の画像と前の画像の差を取り込む畳み込みニューラルネットワークを定義している
# これは、Q(s,left)とQ(s,right)を表す2つの出力を持っている(sはネットワークへの入力)
# このネットワークは、現在の入力が与えられた各アクションを実行した場合の期待リターンを予測する

class DQN(nn.Module):

    # DQNクラスのコンストラクタ
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)          # 2次元の畳み込み
        self.bn1 = nn.BatchNorm2d(16)                                   # 2次元のバッチ正規化
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)         # 2次元の畳み込み
        self.bn2 = nn.BatchNorm2d(32)                                   # 2次元のバッチ正規化
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)         # 2次元の畳み込み
        self.bn3 = nn.BatchNorm2d(32)                                   # 2次元のバッチ正規化
        
        # 入力画像のサイズを計算
        # (線形入力接続数は、conv2dレイヤーの出力に依存するため)
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))    # 入力画像の幅を算出
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))    # 入力画像の高さを算出
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)               # 全結合(線型変換)

    # 順伝播
    # (次のアクションを決定するための1つの要素、)
    # (または最適化中のバッチのいずれかで呼び出される。)
    # (ensor([[left0exp,right0exp]...])を返す。)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))                             # 畳み込み -> バッチ正規化 -> 活性化(ReLU)
        x = F.relu(self.bn2(self.conv2(x)))                             # 畳み込み -> バッチ正規化 -> 活性化(ReLU)
        x = F.relu(self.bn3(self.conv3(x)))                             # 畳み込み -> バッチ正規化 -> 活性化(ReLU)
        return self.head(x.view(x.size(0), -1))                         # 行列の形状を変換 -> 全結合


########################################################################
# 定義

# 画像のサイズを取得
size_image = get_image.get_image()
screen_height, screen_width, _ = size_image.shape

# アクション数を指定
n_actions = 16

#----------------------------------------------------------------------#
# load learning model

print('Load Training Model')
model_path = "/home/sobits/catkin_ws/src/Robotic_Image_Segmentation_Architecture/robotic_img_seg_archit/model/experiment01/"

# 方策(policy)を決定するためのネットワークを作成
policy_net = DQN(screen_height, screen_width, n_actions).to(device)

# 学習したモデルの読み込み
policy_net.load_state_dict(torch.load(model_path + "policy_net_1000.pth"))


########################################################################
# 行動の選択

action_flag = True
#action_flag = False

def select_action(state, policy_net, n_actions, device):
    if action_flag == True :
        with torch.no_grad():
            # t.max(1)は、各行の最大カラムの値を返す
            # (max結果の2番目のカラムは、max要素が見つかった場所のインデックスなので、期待される報酬が大きいアクションを選ぶ。)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # random.randrange(start, stop, step) - start数からstop数をstep数で割った数のうちランダムな値を返す
        # (start, stepを省略した場合は、start=0, step=1)
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


########################################################################
# Robotic Image Segmentation Architecture メイン

if __name__ == "__main__":

    #-------------------------------------------------------------------
    # テスト(テストループ：メイン)

    # ここでは、深層強化学習のテストを行う
    # ①：環境をリセットし、状態配列を初期化
    # ②：あるアクションをサンプリングし、それを実行
    # 　　次の画像と報酬(常に1)を観察し、モデルを1回最適化する
    # ③：エピソードが終了すると、ループを再開する

    print("\n#############################################################")
    print("                      Start test!!")
    print("#############################################################\n")

    #---------------------------------------------------------------
    # パラメータの設定

    num_episodes   =  10   # エピソード数の設定
    num_step       =  4      # ステップ数の設定

    image_path = "/home/sobits/catkin_ws/src/Robotic_Image_Segmentation_Architecture/robotic_img_seg_archit/result_img/"

    for i_episode in range(num_episodes):

        print("=============================================================")
        print("                    Start {0:03d}_episode!!".format(i_episode+1))
        print("=============================================================")

        #---------------------------------------------------------------
        # 初期化

        step = 1             # ステップ数の初期化
        sum_reward_num = 0   # 報酬の総和を初期化

        # カメラパンチルトの初期化
        img.move_pantilt(0)
        rospy.sleep(3)

        # 現在の画像と前の画像の差分を取るための画像を取得(この段階では、同じ画像の差分)
        base_image = get_image.get_image()
        current_image = get_image.get_image()

        # エピソードごとに保存
        cv2.imwrite(image_path + "base_img_{0:03d}.png".format(i_episode + 1), base_image)

        # 画像の差分を取る処理(現在の状態をstateに保存)
        diff_image = cv2.absdiff(base_image, current_image)
        state = img.change_image_type(diff_image, device)
        print(state.size())
        print("-------------------------------------------------------------")

        #---------------------------------------------------------------
        # 試行

        # stepを4回繰り返したら終了
        while not step == num_step + 1:

            print("【step {0}】".format(step))

            # 行動を選択(カメラパンチルトの動作)
            action = select_action(state.to(device), policy_net, n_actions, device)

            # 選択した行動を実行(Service Clientでカメラパンチルトを動かす)
            img.move_pantilt(action.item() + 1)
            rospy.sleep(3)

            # 現在の画像と前の画像の差分を取るための画像を取得(行動による画像の差分)
            current_image = get_image.get_image()

            # 新しい状態を観察
            # 画像の差分を取る処理(現在の状態をstateに保存)
            next_diff_image = cv2.absdiff(base_image, current_image)

            # 画像の差分を足し合わせる
            add_image = cv2.add(diff_image, next_diff_image)
            next_state = img.change_image_type(add_image, device)

            # 価値(報酬)の取得
            reward_num = img.get_value(add_image, action.item() + 1, i_episode + 1, step)

            # 次の状態に移動
            diff_image = add_image
            state = next_state

            # ステップをカウント
            step += 1

    print("Complete")


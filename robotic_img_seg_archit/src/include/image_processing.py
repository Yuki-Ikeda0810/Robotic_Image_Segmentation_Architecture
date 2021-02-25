#!/usr/bin/env python3
# coding: utf-8

# ROS
import rospy

# カメラパンチルト
from pantilt_control.srv import PanTilt

# ランダム
import random

# 配列
import numpy as np

# 画像処理
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Pytorch
import torch


########################################################################
# 画像を取得

# Azure Kinectから画像を取得
class GetImage():
    def __init__(self):
        self.image_path = "/home/sobits/catkin_ws/src/Robotic_Img_Seg_Archit/robotic_img_seg_archit/result_img/"
        self.img_msg = Image()
        self.num = 1

    def callback(self, msg):
        self.img_msg = msg

    # cvBridgeでOpenCV型に変えた画像を返す関数
    def get_image(self):
        cv_img = CvBridge().imgmsg_to_cv2(self.img_msg, "bgr8") # 3ch

        # 画像をダウンサンプリング
        down_image1 = cv2.pyrDown(cv_img)       # 横幅/2、縦幅/2
        down_image2 = cv2.pyrDown(down_image1)  # 横幅/2、縦幅/2

        # 画像をトリミング
        trim_image = down_image2[79-1 : 161-1, 119-1: 201-1] # 82×82ピクセル

        return trim_image

    # cvBridgeでOpenCV型に変えた画像を保存する関数
    def save_image(self):
        save_path = self.image_path + "img_{0:03d}.png".format(self.num)
        print(save_path)
        cv_img = CvBridge().imgmsg_to_cv2(self.img_msg, "bgr8") # 3ch
        cv2.imwrite(save_path, cv_img)
        self.num += 1


########################################################################
# 画像型を変換

# OpenCV型の画像をtorch型の画像に変換
def change_image_type(cv_image, device):

    # numpy(1536, 2048, 3)
    state = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) 

    # numpy(1536, 2048, 3) -> numpy(1, 3, 1536, 2048)
    state_0 = state[:,:,0]
    state_1 = state[:,:,1]
    state_2 = state[:,:,2]
    state = np.stack([state_0, state_1, state_2], 0)   # numpy(3, 1536, 2048)
    state = state[np.newaxis, :, :, :]                 # numpy(1, 3, 1536, 2048)

    # numpy(1, 3, 1536, 2048) -> torch(1, 3, 1536, 2048)
    state = torch.from_numpy(state)

    # dtypeをfloat型に指定
    state = state.to(device=device, dtype=torch.float)

    return state


########################################################################
# カメラパンチルトの動作

# actionの値をモータの駆動角度に変換
def move_pantilt(action):
    # rospy.loginfo('Waiting service')
    rospy.wait_for_service('/pantilt_control')
    service = rospy.ServiceProxy('/pantilt_control', PanTilt)
    print("action :", action)

    # action(1~16)に対するロボットの行動

    ### 左上 ###   ### 右上 ###
    # 01 : 小  #   # 05 : 小  #
    # 02 : 中  #   # 06 : 中  #
    # 03 : 大1 #   # 07 : 大1 #
    # 04 : 大2 #   # 08 : 大2 #
    ############   ############

    ### 左下 ###   ### 右下 ###
    # 09 : 小  #   # 13 : 小  #
    # 10 : 中  #   # 14 : 中  #
    # 11 : 大1 #   # 15 : 大1 #
    # 12 : 大2 #   # 16 : 大2 #
    ############   ############

    # パラメータの設定
    small   = 5
    medium  = 10
    large   = 15
    l_large = 20

    ### 初期ポジション ###
    if action == 0:
        response = service(0, 0)

    ### 左上 ###
    elif action == 1:
        response = service(small, small)
    elif action == 2:
        response = service(medium, medium)
    elif action == 3:
        response = service(large, large)
    elif action == 4:
        response = service(l_large, l_large)

    ### 右上 ###
    elif action == 5:
        response = service(-small, small)
    elif action == 6:
        response = service(-medium, medium)
    elif action == 7:
        response = service(-large, large)
    elif action == 8:
        response = service(-l_large, l_large)

    ### 左下 ###
    elif action == 9:
        response = service(small, -small)
    elif action == 10:
        response = service(medium, -medium)
    elif action == 11:
        response = service(large, -large)
    elif action == 12:
        response = service(l_large, -l_large)

    ### 右下 ###
    elif action == 13:
        response = service(-small, -small)
    elif action == 14:
        response = service(-medium, -medium)
    elif action == 15:
        response = service(-large, -large)
    elif action == 16:
        response = service(-l_large, -l_large)

    # print(response)


########################################################################
# セグメンテーションの価値を設定

def get_value(image, action, episode, step):

    image_path = "/home/sobits/catkin_ws/src/Robotic_Img_Seg_Archit/robotic_img_seg_archit/result_img/"

    #-------------------------------------------------------------------
    # 前処理
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu法で画像を2値化
    thresh, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)

    # 収縮処理
    erosion_image = cv2.erode(binary_image, kernel, iterations = 3)

    # 膨張処理
    dilation_image = cv2.dilate(erosion_image, kernel, iterations = 2)

    """
    # オープニング(白いノイズ除去)
    opening_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # クロージング(黒いノイズ除去)
    closing_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    """

    #-------------------------------------------------------------------
    # 塗りつぶし処理

    # 連結成分のラベリングを行う
    # n_labels  : ラベル数(背景はラベル0に振り分けられるため、実際のオブジェクト数はn_labels - 1)
    # labels    : 画像のラベリング結果を保持している二次元配列(配列の要素は、各ピクセルのラベル番号となっている)
    # stats     : 各ラベルの構造情報を保持(オブジェクトのバウンディングボックス(開始点の x 座標、y 座標、幅、高さ)とオブジェクトのサイズ)
    # centroids : オブジェクトの重心
    n_labels, labels_image, stats, centroids = cv2.connectedComponentsWithStats(dilation_image)

    print("number of labels :", n_labels - 1)

    #-----------------------------------------------------------------------
    # 全てのオブジェクト用の変数

    # セグメント画像の初期設定
    seg_image = np.zeros(image.shape[0:3])

    # 画像の縦幅と横幅
    height, width = image.shape[0:2]

    # ラベルの色を保存する配列
    cols = []

    #-----------------------------------------------------------------------
    # オブジェクトに付与する色を用意

    if(n_labels - 1 <= 2):
        # 正解画像に合わせて決まった色を用意
        # 背景はラベル0に振り分けられるため、実際のオブジェクトのラベルは1から始まる
        cols.append(np.array([255, 255, 255])) # 白(BGR)
        cols.append(np.array([185, 173, 0]))   # 青(BGR)
        cols.append(np.array([0, 234, 247]))   # 黃(BGR)
    elif(n_labels - 1 == 3):
        # 正解画像に合わせて決まった色を用意
        # 背景はラベル0に振り分けられるため、実際のオブジェクトのラベルは1から始まる
        cols.append(np.array([255, 255, 255])) # 白(BGR)
        cols.append(np.array([185, 173, 0]))   # 青(BGR)
        cols.append(np.array([0, 234, 247]))   # 黃(BGR)
        cols.append(np.array([138, 24, 255]))  # 赤(BGR)
        #cols.append(np.array([0, 234, 247]))   # 黃(BGR)

    elif(3 < n_labels - 1):
        # オブジェクトのラベルの数だけ、ランダムな色を用意する
        # 背景はラベル0に振り分けられるため、実際のオブジェクトのラベルは1から始まる
        for i in range(n_labels):
            cols.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    #-----------------------------------------------------------------------
    # 大きいオブジェクト用の変数

    # 画像中の何％を大きいオブジェクトとするかの閾値
    percent_100 = 0.1 # ％
    percent_1 = percent_100 / 100

    # ラベル数を保存する変数
    big_n_labels = 0
    small_n_labels = 0

    # 大きいオブジェクトのラベル保存する配列
    big_labels = []

    # 各ラベルの重心を保存する配列
    big_centroids = []
    small_centroids = []

    # 重心の差を保存する配列
    diff_centroids_k = []
    diff_centroids_j = []

    # 新しいラベルの色を保存する配列
    big_cols = []

    #-----------------------------------------------------------------------
    # セグメンテーション処理
    
    # 大きいオブジェクトのみを保存
    pixels = height * width
    for l in range(1, n_labels):
        if pixels * percent_1 <= stats[l][4]:
            big_n_labels += 1
            big_cols.append(cols[l])
            big_centroids.append(centroids[l])
            big_labels.append(l)
            seg_image[labels_image == l, ] = cols[l]
        else:
            small_n_labels += 1
            small_centroids.append(centroids[l])

    print("new number of labels :", big_n_labels)

    """
    #小さいオブジェクトを大きいオブジェクトと結合
    for j in range(small_n_labels):
        for k in range(big_n_labels):
            diff_centroids_k.append(abs(big_centroids[k][0] - small_centroids[j][0]) + abs(big_centroids[k][1] - small_centroids[j][1]))

        diff_centroids_j.append(diff_centroids_k)
        diff_centroids_k = []

        if (0 < len(diff_centroids_j[j])):
            including_label = diff_centroids_j[j].index(min(diff_centroids_j[j]))
            seg_image[labels_image == big_labels[k], ] = big_cols[including_label]
    """

    #-----------------------------------------------------------------------
    # 出力

    if (episode <= 10 or 990 <= episode):
        cv2.imwrite(image_path + "seg_img_{0:03d}_{1}.png".format(episode, step), seg_image)

    if ((episode % 200) == 0):
        # エピソードごとに保存
        cv2.imwrite(image_path + "seg_img_{0:03d}_{1}.png".format(episode, step), seg_image)

    #-----------------------------------------------------------------------
    # 報酬の設定

    """
    for i, row in enumerate(stats):
        #print(f"label {i}")
        #print(f"* topleft: ({row[cv2.CC_STAT_LEFT]}, {row[cv2.CC_STAT_TOP]})")
        #print(f"* size: ({row[cv2.CC_STAT_WIDTH]}, {row[cv2.CC_STAT_HEIGHT]})")
        #print(f"* area: {row[cv2.CC_STAT_AREA]}")
        if i == 0:
            edge_pixels = row[cv2.CC_STAT_AREA]
    
    # 価値(報酬)の設定
    value = big_n_labels * (1 - float(edge_pixels/pixels/2))
    #print("edge_pixels/pixels :", float(edge_pixels/pixels))
    print("value : {0:03f}".format(value))
    """

    # 価値を求めるパラメータの設定
    small   = 5
    medium  = 10
    large   = 15
    l_large = 20

    # 価値に変換
    if action == 1 or action == 5 or action == 9 or action == 13:
        action = small
    elif action == 2 or action == 6 or action == 10 or action == 14:
        action = medium
    elif action == 3 or action == 7 or action == 11 or action == 15:
        action = large
    elif action == 4 or action == 8 or action == 12 or action == 16:
        action = l_large

    # 価値(報酬)の設定
    value = big_n_labels * (1 + (small/l_large) - (action / l_large))
    print("value : {0:03f}".format(value))

    return value


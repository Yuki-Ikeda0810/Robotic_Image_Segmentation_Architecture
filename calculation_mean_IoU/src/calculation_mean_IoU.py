#!/usr/bin/env python3
# coding: utf-8

# 配列
import numpy as np

# 画像処理
import cv2

########################################################################
# IoUを算出する関数

def calculation_IoU(correct_image, result_image, correct_B, correct_G, correct_R, result_B, result_G, result_R):
    
    #=======================================================================
    # 変数の設定

    # 変数の初期化
    # TP(True Positive)  : 正解の画像と同じ場所を物体であると検出した
    # FP(False Positive) : 正解の画像にない場所を物体であると誤った検出した
    # FN(False Negative) : 正解の画像にある場所を物体であると検出できなかった
    TP, FP, FN = 0, 0, 0
    
    # 画像の縦幅と横幅を取得
    height = result_image.shape[0]
    width = result_image.shape[1]

    #=======================================================================
    # TP, FP, FNの判定

    for i in range(height):
        for j in range(width):

            # 評価対象の画像の画素が指定された物体の色である場合(物体を検出した場所)
            if all(result_image[i][j] == np.array([result_B, result_G, result_R])):

                # 正解の画像の画素が指定された物体の色である場合(物体がある場所)
                if all((correct_image[i][j]) == np.array([correct_B, correct_G, correct_R])):
                    TP = TP + 1
                else:
                    FP = FP + 1
            
            # 評価対象の画像の画素が指定された物体の色でない場合(物体を検出した場所)
            else:

                # 正解の画像の画素が指定された物体の色である場合(物体がある場所)
                if all(correct_image[i][j] == np.array([correct_B, correct_G, correct_R])):
                    FN = FN + 1

    #=======================================================================
    # 結果の出力

    # TP, FP, FNの数
    print("TP:{0}, FP:{1}, FN:{2}".format(TP, FP, FN))

    # IoU
    IoU = TP/(TP + FP + FN)
    print("IoU :", IoU)

    return IoU


########################################################################
# Segmentation評価用プログラムのメイン

if __name__ == '__main__':

    #=======================================================================
    # 変数の設定
    
    # 評価する画像の枚数
    num_img = 10

    # 色のパラメータ設定(正解の画像)
    correct1_B, correct1_G, correct1_R = 185, 173, 0   # 物体1の画素値(背景)
    correct2_B, correct2_G, correct2_R = 0, 234, 247   # 物体2の画素値(机)
    correct3_B, correct3_G, correct3_R = 138, 24, 255  # 物体3の画素値(物体

    # 色のパラメータ設定(評価対象の画像)
    result1_B, result1_G, result1_R = 185, 173, 0      # 物体1の画素値(背景)
    result2_B, result2_G, result2_R = 0, 234, 247      # 物体2の画素値(机)
    result3_B, result3_G, result3_R = 138, 24, 255     # 物体3の画素値(物体)

    """
    # 色のパラメータ設定(評価対象の画像)
    result1_B, result1_G, result1_R = 90, 60, 199      # 物体1の画素値(背景)
    result2_B, result2_G, result2_R = 1, 189, 139      # 物体2の画素値(机)
    result3_B, result3_G, result3_R = 94, 163, 203     # 物体3の画素値(物体)
    """

    # 各画像に対するmean IoUの合計の初期化
    sum_mean_IoU = 0

    #=======================================================================
    # 各画像の各クラスのIoUとmean IoUの計算

    # 正解の画像を読み込む
    image_path = "/home/sobits/catkin_ws/src/Robotic_Image_Segmentation_Architecture/calculation_mean_IoU/img/experiment02/right_position/"
    correct_image = cv2.imread(image_path + "correct_img.png")
    print("correct_image :", correct_image.shape)

    for i_img in range(num_img):

        # 評価対象の画像を読み込む
        result_image = cv2.imread(image_path + "result_img{0}.png".format(i_img + 1))

        # 画像の画素数を表示
        print("\n=============================")
        print("result_image{0} : {1}".format(i_img + 1, result_image.shape))
        print("=============================")

        print("class 01")
        IoU1 = calculation_IoU(correct_image, result_image, correct1_B, correct1_G, correct1_R, result1_B, result1_G, result1_R)

        print("-----------------------------")
        print("class 02")
        IoU2 = calculation_IoU(correct_image, result_image, correct2_B, correct2_G, correct2_R, result2_B, result2_G, result2_R)

        print("-----------------------------")
        print("class 03")
        IoU3 = calculation_IoU(correct_image, result_image, correct3_B, correct3_G, correct3_R, result3_B, result3_G, result3_R)

        # Mean IoU
        print("=============================")
        mean_IoU = (IoU1 + IoU2 + IoU3) / 3
        print("Mean IoU :", mean_IoU)

        sum_mean_IoU = sum_mean_IoU + mean_IoU

    #=======================================================================
    # 各画像のmean IoUから平均を計算

    result_mean_IoU = sum_mean_IoU / num_img
    print("\n\n{0}枚の画像に対する Mean IoU の平均 : {1}".format(num_img, result_mean_IoU))
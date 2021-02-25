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
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

# Pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

# 定義した画像処理のクラスや関数を読み込む
import include.image_processing as img


########################################################################
# 初期設定

rospy.init_node("create_learning_model")
rospy.sleep(3)

# 画像を取得するクラスの初期設定
get_image = img.GetImage()
sub = rospy.Subscriber("/rgb/image_raw", Image, get_image.callback)
rospy.sleep(1)

# deviceの設定(GPUを使う場合)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# matplotlibの初期設定
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# グラフ表示のインタラクティブモードをオンにする
plt.ion()


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

# 方策(policy)を決定するためのネットワークを作成
policy_net = DQN(screen_height, screen_width, n_actions).to(device)

# ターゲットネットワークを作成
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())  # policy_netのパラメータを読み込む
target_net.eval()  # 文字列を式に変換

# モデルの最適化を行うパラメータの更新
optimizer = optim.RMSprop(policy_net.parameters())
# 他のoptimizerを使うのもあり(optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9))


########################################################################
# リプレイメモリ(遷移を保存する)

# ここでは、DQNを訓練するために、エクスペリエンス・リプレイ・メモリを定義している。
# これにより、エージェントが観測した遷移を記憶し、後でこのデータを再利用することができる。
# そして、ここからランダムにサンプリングすることで、バッチを構成する遷移は非相関化される。
# これにより、DQNの学習手順が大幅に安定化され、改善されることが示されている。

# Transitionは、環境内の単一の遷移を表す名前付きタプル
# (基本的には(state, action)のペアを、(next_state, reward)の結果にマッピングする。)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# ReplayMemoryは、最近観測された遷移を保持するサイズに制限のあるサイクリックバッファ
# (学習のために遷移のランダムなバッチを選択する.sample() メソッドも実装されている。)
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # ランダムに複数の要素を選択(重複なし)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


########################################################################
# 行動の選択

# select_action関数は、epsilon greedy法の方策に従ってアクションを選択する
# epsilon greedy法は、確率epsilonで行動選択を行う方法である(最もQ値の高い行動を選択する or ランダムに行動を選択する)
# ランダムな行動を選択する確率は、EPS_STARTから始まり、EPS_ENDに向かって指数関数的に減衰していく
# このとき、EPS_DECAY は減衰の速度を制御している

EPS_START  =  0.9    # 最初のランダムな行動を選択する確率
EPS_END    =  0.05   # 最後のランダムな行動を選択する確率
EPS_DECAY  =  200    # 確率が減衰する速度

steps_done = 0

def select_action(state, policy_net, n_actions, device):
    global steps_done
    sample = random.random() # random.random() - 0〜1のランダムな値を返す
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if eps_threshold < sample:
        with torch.no_grad():
            # t.max(1)は、各行の最大カラムの値を返す
            # (max結果の2番目のカラムは、max要素が見つかった場所のインデックスなので、期待される報酬が大きいアクションを選ぶ。)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # random.randrange(start, stop, step) - start数からstop数をstep数で割った数のうちランダムな値を返す
        # (start, stepを省略した場合は、start=0, step=1)
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


########################################################################
# トレーニング(トレーニングループ：定義)

# ここでは、最適化の1つのステップを実行するoptimize_model関数を定義している。
# 最初にバッチをサンプリングし、すべての配列を1つに連結し、
# Q(st,at)とV(st+1)=maxaQ(st+1,a)を計算、それらを損失に結合する。
# 定義により、sが終端状態である場合、V(s)=0とする。
# また、V(st+1)を計算するためにターゲット・ネットワークを使用して安定性を高めていく。
# ターゲット・ネットワークの重みはほとんどの時間は凍結されているが、
# ポリシー・ネットワークの重みで更新される。
# これは通常は設定されたステップ数ですが、簡単にするためにエピソードを用いる。

BATCH_SIZE = 5
GAMMA = 0.999

# 遷移を記録するためのメモリを作成
memory = ReplayMemory(100)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transitionsのバッチ配列を、バッチ配列のTransitionに変換
    batch = Transition(*zip(*transitions))

    # 非最終状態のマスクを計算し、バッチ要素を連結
    # (最終状態はシミュレーションが終了した後の状態)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a)を計算するには、モデルはQ(s_t)を計算し、次に、取られたアクションの列を選択
    # (これらは、policy_netに従って各バッチ状態に対して取られたであろうアクションである。)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # すべての次の状態についてV(s_{t+1})を計算
    # (non_final_next_statesのアクションの期待値は、"古い" target_netに基づいて計算される(max(1)[0]で最高の報酬を選択する)。)
    # (これはマスクに基づいて合わされ、状態が最終的なものであった場合には、期待される状態値か0のどちらかが得られるようになる。)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.to(device)).max(1)[0].detach()

    # 期待されるQ値を計算
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber損失を計算
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # モデルの最適化
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


########################################################################
# エピソードごとの持続時間をグラフに表示

# plot_valueは、エピソードの持続時間をプロットするヘルパーで、
# 過去100エピソードの平均値(公式評価で使用されます)と一緒にプロットする。
# プロットは、メインのトレーニングループを含むセルの下に表示され、各エピソードの後に更新される。

episode_value = []
episode = []

def plot_value():

    # pltの初期化
    plt.clf
    plt.close()

    # グラフ領域を作成
    plt.figure(figsize=(6, 5))

    # グラフ間のスペースを作成
    plt.subplots_adjust(hspace=0.5)

    episode_x = torch.tensor(episode, dtype=torch.float)
    value_y = torch.tensor(episode_value, dtype=torch.float)

    plt.subplot(2,1,1)
    plt.title("Reward Graph")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(episode_x.numpy(), value_y.numpy(), color="blue")

    # エピソード-平均報酬グラフ(50エピソードの平均)
    if 50 <= len(value_y):
        means = value_y.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.subplot(2,1,2)
        plt.title("Average Reward Graph")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.plot(episode_x.numpy(), means.numpy(), color="orange")

    # 区画が更新されるように一時停止
    plt.pause(0.001) 

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


########################################################################
# Robotic Image Segmentation Architecture メイン

if __name__ == "__main__":

    #-------------------------------------------------------------------
    # トレーニング(トレーニングループ：メイン)

    # ここでは、深層強化学習のメイントレーニングを行う
    # ①：環境をリセットし、状態配列を初期化
    # ②：あるアクションをサンプリングし、それを実行
    # 　　次の画像と報酬(常に1)を観察し、モデルを1回最適化する
    # ③：エピソードが終了すると、ループを再開する

    print("\n#############################################################")
    print("                      Start training!!")
    print("#############################################################\n")

    #---------------------------------------------------------------
    # パラメータの設定

    num_episodes   =  1000   # エピソード数の設定
    num_step       =  4      # ステップ数の設定
    TARGET_UPDATE  =  10     # 更新の頻度

    image_path = "/home/sobits/catkin_ws/src/Robotic_Image_Segmentation_Architecture/robotic_img_seg_archit/result_img/"
    model_path = "/home/sobits/catkin_ws/src/Robotic_Image_Segmentation_Architecture/robotic_img_seg_archit/model/"
    graph_path = "/home/sobits/catkin_ws/src/Robotic_Image_Segmentation_Architecture/robotic_img_seg_archit/result_img/"

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

        if (i_episode + 1 <= 10 or 990 <= i_episode + 1):
            cv2.imwrite(image_path + "base_img_{0:03d}.png".format(i_episode + 1), base_image)

        if (((i_episode + 1) % 200) == 0):
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
            sum_reward_num = sum_reward_num  + reward_num 
            reward_tensor = torch.tensor([sum_reward_num], device=device)

            # 遷移をメモリに保存
            memory.push(state, action, next_state, reward_tensor)

            # 次の状態に移動
            diff_image = add_image
            state = next_state

            # 最適化の1つのステップを実行(ターゲット・ネットワーク上)
            optimize_model()

            # ステップをカウント
            step += 1

        # グラフの描画
        episode_value.append(sum_reward_num)
        episode.append(i_episode+1)
        plot_value()

        # DQN内のすべての重みとバイアスをコピーして、ターゲットネットワークを更新
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (((i_episode + 1) % 200) == 0):
            # 学習したモデルの保存
            print("Save Training Model")
            policy_net_path = model_path + "policy_net_{}.pth".format(i_episode + 1)
            torch.save(policy_net.state_dict(), policy_net_path)

    print("Complete")

    # グラフ表示のインタラクティブモードをオフにする
    plt.ioff()

    # グラフの表示
    plt.savefig(graph_path + "graph.png")
    plt.show()


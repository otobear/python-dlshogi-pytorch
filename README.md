#  python-dlshogi-pytorch

「将棋AIで学ぶディープラーニング」の[ソースコード](https://github.com/TadaoYamaoka/python-dlshogi)を PyTorch で書き換えたものです。

## 環境構築

GPU
```
docker image build -t [USERNAME]/python-dlshogi-pytorch .
docker run -it --rm --gpus all [USERNAME]/python-dlshogi-pytorch /bin/bash
```

CPU
```
docker image build -t [USERNAME]/python-dlshogi-pytorch .
docker run -it --rm [USERNAME]/python-dlshogi-pytorch /bin/bash
```

## 棋譜準備

```
mkdir ../kifu
cd ../kifu
wget --trust-server-names "https://osdn.net/projects/shogi-server/downloads/68500/wdoor2016.7z/"
7z x wdoor2016.7z
cd ../python-dlshogi-pytorch
python utils/filter_csa.py ../kifu/2016
python utils/make_kifu_list.py ../kifu/2016/ kifulist
```

## 学習実行

- 方策ネット(P124)

```
python train_policy.py kifulist_train.txt kifulist_test.txt
```

- 価値ネット(P178)

```
python train_value.py kifulist_train.txt kifulist_test.txt
```

- 転移学習(P188)

```
python transfer_policy_to_value.py model/model_policy model/model_value_transferred
python train_value.py kifulist_train.txt kifulist_test.txt -m model/model_value_transferred
```

- マルチタスク学習(P199)

```
python train_policy_value.py kifulist_train.txt kifulist_test.txt
```

- ResNet(P204)

```
python train_policy_value_resnet.py kifulist_train.txt kifulist_test.txt
```

## USI エンジン

- 方策ネット(P145)

```
python -m pydlshogi.usi.usi_policy_player
```

- 価値ネット(P182)

```
python -m pydlshogi.usi.usi_search1_player
```

- モンテカルロ木探索(P235)

```
python -m pydlshogi.usi.usi_mcts_player
```

- 並列化モンテカルロ木探索(P251)

```
python -m pydlshogi.usi.usi_mcts_player
```

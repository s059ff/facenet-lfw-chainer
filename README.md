# facenet-mnsit-chainer
MNIST データセットを使用した FaceNet による手書き数字画像の類似度推定 及び クラスタリングのテストです。

## 実行環境
* Ubuntu 18.04 64bit LTS
* Python 3.7.2
* Chainer 5.2.0
* cupy 5.2.0

## 実行方法
* モデルのトレーニング

    ```bash
    python train.py
    ```

* クラスタリング結果 可視化 (PCA により 128 次元から 2 次元へ圧縮して表示します)

    ```bash
    python visualize_embeddings.py -s (モデルのトレーニングで作成された embeddings-*.npy のパス)
    ```

* クラスタリング結果 サンプル間類似度(距離) 可視化

    ```bash
    python visualize_distances.py -s (モデルのトレーニングで作成された embeddings-*.npy のパス) -d (距離の定義, l2 もしくは cos のどちらか)
    ```

## 実行結果
* クラスタリング結果  
すべての validation データの FaceNet による 28x28(=768) => 128 次元への特徴抽出 => PCA により 2 次元へ圧縮したもの  

![](https://github.com/s059ff/facenet-mnist-chainer/blob/master/examples/embeddings-100.png)

* サンプル間距離 推定 結果  
validation データからランダムに抽出したデータ, それらのデータ間の距離  

![](https://github.com/s059ff/facenet-mnist-chainer/blob/master/examples/embeddings-100-choices.png)
![](https://github.com/s059ff/facenet-mnist-chainer/blob/master/examples/embeddings-100-distances.png)

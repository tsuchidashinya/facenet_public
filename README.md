# facenet_demo
## ラズパイへのインストール方法
#### venvで仮想環境を作る
```
python3 -m venv env
source `pwd`/env/bin/activate
```
#### pytorchをインストール
pytorchは普通にインストールしてbuildするとめちゃくちゃ重くなってしまう。
そのため、公開されているbuild済みのパッケージをインストールして行う
```
//とりあえず準備
sudo apt update
sudo apt upgrade
sudo apt install libopenblas-dev m4 cmake cpython python3-dev python3-yaml python3-setuptools
//githubからインストール
git clone https://github.com/Kashu7100/pytorch-armv7l.git
cd pytorch-armv7l
pip install torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
pip install torchvision-0.8.0a0+45f960c-cp37-cp37m-linux_armv7l.whl
```
### facenet-pytorchをインストール
```
pip install facenet-pytorch
```
### facenet_demo(このパッケージ)をインストールして実行
```
git clone https://github.com/s-tsuchida-safie/facenet_demo
cd facenet_demo
python face_detect.py
```


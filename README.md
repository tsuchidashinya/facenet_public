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
```bash
#とりあえず準備
sudo apt update
sudo apt upgrade
sudo apt install libopenblas-dev m4 cmake cpython python3-dev python3-yaml python3-setuptools
#githubからインストール
git clone https://github.com/Kashu7100/pytorch-armv7l.git
cd pytorch-armv7l
pip install torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
pip install torchvision-0.8.0a0+45f960c-cp37-cp37m-linux_armv7l.whl
```
### facenet-pytorchをインストール
```
pip install facenet-pytorch
```
## サンプルプログラムの実行方法(ラズパイとdesktop環境共通)
### facenet_demo(このパッケージ)をクローン
```
git clone https://github.com/s-tsuchida-safie/facenet_demo
cd facenet_demo
```
### yamlファイルに実行したい画像リストのパスを記入
yaml/sample_images.yaml
```yaml
- images/same_person_images/tsuchida_normal.jpg
- images/same_person_images/tsuchida_smile.jpg
- images/same_person_images/tsuchida_look_away.jpg
- images/same_person_images/tsuchida_blink.jpg
- images/same_person_images/tsuchida_mask.jpg
- images/same_person_images/angelina_front_1.jpg
- images/same_person_images/angelina_front_2.jpg
- images/same_person_images/angelina_front_3.jpg
- images/same_person_images/angelina_look_away_1.jpg
- images/same_person_images/angelina_look_away_2.jpg
```
### Pythonファイルを実行
```
python face_detect.py -y sample_images.yaml
```



# emotion_recognition
## 概要

このリポジトリでは、音声ファイルに含まれている感情を推論します。

.wavファイルを用いて、その音声が驚きや喜びなど、どの感情が含まれているのかを推測します。

ローカルPCから.wavファイルを与えることで、そのファイルがどのような感情かを推論できます。

*動作環境*：　AWS SageMakerStudio

*制作期間*: 2日

※２０２２年08月　現在、精度に課題があります。

## 処理課程
*音声データの前処理*: メルスペクトログラムへと変換

*機械学習アルゴリズム*: Visual Transformer


### メルスペクトログラムを用いる理由
従来の音声感情認識では、音声からMFCC特徴量を抽出して学習していました。
しかし、以下のDossou　et al. (2021)にあるように、メルスペクトログラムを使用することによって、精度が上がることが知られました。

https://openaccess.thecvf.com/content/ICCV2021W/ABAW/html/Dossou_FSER_Deep_Convolutional_Neural_Networks_for_Speech_Emotion_Recognition_ICCVW_2021_paper.html


### メルスペクトログラム生成のポイント
今回、メルスペクトログラムを生成するにあたり、学習目的のため、定番ライブラリであるlibrosaは用いていません。

`modules.mel_spectrogram.py`にて、実装しています。


### Vision Transformerを用いる理由
上記の論文では、古典的なCNNが用いられていましたが、新しい画像認識モデルを用いることで精度が向上すると考えました。

Visual Transformer(以下、ViT)は、Dosovitskiy et al. (2021)によってできた、

自然言語処理で有名なTransformerを画像認識タスクに用いたモデルです。

画像の各部分(patch)を自然言語処理の「単語」と捉え、

画像を「文章」として認識て処理する方法で、

大多数の画像認識モデルで使われる「畳み込み」を一切使わないことに特徴があります。

画像認識タスクにてSoTAを達成しています。

Dosovitskiy et al. (2021)：
https://openreview.net/forum?id=YicbFdNTTy

### Vision Transformerの特徴
Transformerはattentionを用いたモデルです。

これは、自然言語処理にて、文章の中のどの単語同士が強く関連しているかに着目する仕組みです。

そのためほぼ同様のアルゴリズムであるViTは、画像の離れた部分同士の関係性を考慮できます。

これは、CNNで行われていた畳み込みが、画像の各部分に近しい部分を集中的に考慮する演算であることと対照的です。

いわば、画像の全体像を重視するモデルです。


## 使用データ


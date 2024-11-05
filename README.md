# 有害文書フィルタリング

## 実行環境の準備

以下では、次の環境を想定しております。

```
> python -V
Python 3.10.15

> cat /etc/lsb-release
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04.3 LTS"

> uname -a
Linux template-gpu 5.15.0-87-generic #97-Ubuntu SMP Mon Oct 2 21:09:21 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
```

```
> nvidia-smi
Thu Oct 31 12:14:56 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:03:00.0 Off |                    0 |
| N/A   39C    P0             200W / 400W |  19123MiB / 40960MiB |     72%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM4-40GB          On  | 00000000:05:00.0 Off |                    0 |
| N/A   41C    P0             399W / 400W |  19123MiB / 40960MiB |     15%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
```

```
|   2  NVIDIA A100-SXM4-40GB          On  | 00000000:0D:00.0 Off |                    0 |
| N/A   34C    P0             277W / 400W |  19123MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM4-40GB          On  | 00000000:1E:00.0 Off |                    0 |
| N/A   41C    P0             363W / 400W |  19123MiB / 40960MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     19642      C   python3                                   19110MiB |
|    1   N/A  N/A     18146      C   python3                                   19110MiB |
|    2   N/A  N/A     19198      C   python3                                   19110MiB |
|    3   N/A  N/A     18887      C   python3                                   19110MiB |
+---------------------------------------------------------------------------------------+
```

### CUDA 12.1のインストール

次のようにしてインストールします。

``` sh
> wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
> sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
> wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
> sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
> sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
> sudo apt-get update
> sudo apt-get -y install cuda-toolkit-12-1
```

インストールされているか確認します。

```sh
> nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0```
```

### TensorRT 10.5.0のインストール

まず`nv-tensorrt-local-repo-ubuntu2204-10.5.0-cuda-12.6_1.0-1_amd64.deb`を`https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing`からダウンロードします。

そして以下を実行します。

```sh
> os="ubuntu2204"
> tag="10.5.0-cuda-12.6"
> sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
> sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
> sudo apt-get update
> sudo apt-get install tensorrt
```

インストールされているか確認します。

```sh
> dpkg-query -W tensorrt
tensorrt        10.5.0.18-1+cuda12.6
```

cuda12.6とありますがCUDA 12.1で動きます。

### ONNXのインストール

ONNX経由でTensorRTを使うので、ONNXとONNX Runtimeをインストールします。

```sh
> pip uninstall onnx onnxruntime-gpu
> pip install onnx onnxruntime-gpu
```

念のため、先にアンインストールしてからインストールしました。
インストールされているか確認します。

```sh
> python3
Python 3.10.15 (main, Oct  3 2024, 07:27:34) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import onnxruntime as ort; print("Available providers:", ort.get_available_providers())
Available providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
>>>
```

`TensorrtExecutionProvider`がインストールされているのでOKです。
（もしからしたら`CUDAExecutionProvider`も必須かも... インストールされているのでいいですが。）

### torchとtransformersのインストール

CUDA 12.1用をインストールします。

```sh
> pip3 install torch --index-url https://download.pytorch.org/whl/cu121
> pip3 install transformers
```

インストールされているか確認します。

```sh
> python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
[{'label': 'POSITIVE', 'score': 0.9998704195022583}]
```

### GNU parallelのインストール

並列処理にGNU parallelを使うので、以下のようにインストールします。

```sh
> $ sudo apt-get install parallel
```

インストールされているか確認します。

```sh
> parallel -V
GNU parallel 20210822
Copyright (C) 2007-2021 Ole Tange, http://ole.tange.dk and Free Software
Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
GNU parallel comes with no warranty.

Web site: https://www.gnu.org/software/parallel

When using programs that use GNU Parallel to process data for publication
please cite as described in the manpage.
```


## ファイルの準備

- `/path/to/ja_cc1/`
- `/path/to/ja_cc2/`
- `/path/to/ja_cc3/`
    - フィルタリング対象のファイル`CC-MAIN-2013-2016.jsonl.gz`、`CC-MAIN-2017-04.jsonl.gz`などがあるディレクトリです。

- `final_model/`
    - 有害文書フィルタリング用にfinetuningした、HuggingFace形式のDeBERTaモデルです。モデル自体はONNXのものを使いますが、この中にあるトークナイザーが必要になるのでこの中に含めています。（トークナイザーもHuggingfaceからダウンロードできるのでなくてもいいのですが...）

- `deberta.onnx`
    - 上記モデルを、TensorRTによりNVIDIAハードウェアに最適化されたONNX形式にexportしたものです。これを使って入力された文書に有害スコアを付与します。

- `batch_inference_onnx_logits.py`
    - `deberta.onnx`で有害スコアを付与するためのPythonスクリプトです。

- `batch_inference_onnx_logits_ja_cc1.sh`
- `batch_inference_onnx_logits_ja_cc2.sh`
- `batch_inference_onnx_logits_ja_cc3.sh`
    - `batch_inference_onnx_logits.py`を使って`ja_cc1/`、`ja_cc2/`、`ja_cc3/`にあるファイルの各文書に有害スコアを付与するスクリプトです。ほとんど同じです。

- `classify_jsonl.py`
    - `batch_inference_onnx_logits_ja_cc[123].sh`で付与された有害スコアに基づいて各文書を有害か無害かに分類します。

- `parallel_classify.sh.py`
    - `classify_jsonl.py`を並列に実行するためのスクリプトです。


## フィルタリングの手順

### 処理対象ファイルのリストの作成

まず、`ja_cc1/`、`ja_cc2/`、`ja_cc3/`へのリンクをカレントディレクトリに作っておきます。

```sh
> ln -s /path/to/ja_cc1
> ln -s /path/to/ja_cc2
> ln -s /path/to/ja_cc3
```

次に、GPU4枚を使って並列処理するために、`ja_cc1/`、`ja_cc2/`、`ja_cc3/`にあるファイルのリストを作成します。

```sh
> parallel 'ls ja_cc{}|cut -d\. -f1 > file_list{}' ::: 1 2 3
```

`ja_cc1/`、`ja_cc2/`、`ja_cc3/`のファイルリスト`file_list1`、`file_list2`、`file_list3`ができました。次にそれらを4分割します。それぞれ65ファイル（65行）ある場合、17行ずつ（最後のファイルは14行）に分割します。

```sh
> split -l 17 file_list1 -a 2 --numeric-suffixes=1 file_list1_
> split -l 17 file_list2 -a 2 --numeric-suffixes=1 file_list2_
> split -l 17 file_list3 -a 2 --numeric-suffixes=1 file_list3_
```

```sh
> wc -l file_list[123]_0[1-4]
  17 file_list1_01
  17 file_list1_02
  17 file_list1_03
  14 file_list1_04
  17 file_list2_01
  17 file_list2_02
  17 file_list2_03
  14 file_list2_04
  17 file_list3_01
  17 file_list3_02
  17 file_list3_03
  14 file_list3_04
 195 total
```


### 有害スコアの付与

次に、`file_list[123]_0[1-4]`と同じディレクトリで以下を実行します。

```sh
> bash batch_inference_onnx_logits_ja_cc1.sh
> bash batch_inference_onnx_logits_ja_cc2.sh
> bash batch_inference_onnx_logits_ja_cc3.sh

```

すると、同じディレクトリに`toxic_scores1/`、`toxic_scores2/`、`toxic_scores3/`が作られます。これらのディレクトリには`CC-MAIN-2013-2016.txt`、`CC-MAIN-2017-04.txt`などのファイルがあり、その中に、`ja_cc1/CC-MAIN-2013-2016.jsonl.gz`、`ja_cc1/CC-MAIN-2017-04.jsonl.gz`などのファイルの各`text`に付与された有害スコア（positiveクラス（有害）のlogit）が出力されます。`ja_cc1/`, `ja_cc2/`, `ja_cc3/`の分類が、A100 (40GB) x 4枚で、各々だいたい30時間、12時間、11時間くらいかかるかもしれません。


### 有害スコアに基づく分類

最後に、`toxic_scores1/`、`toxic_scores2/`、`toxic_scores3/`にある有害スコアに基づいて、`ja_cc1/CC-MAIN-2013-2016.jsonl.gz`などにある各`text`をフィルタリング（有害か無害かに分類）します。

```sh
> bash parallel_classify.sh ja_cc1/ toxic_scores1/ ja_cc1_toxic/ ja_cc1_toxicity_filtered/ 70 8.4
> bash parallel_classify.sh ja_cc2/ toxic_scores2/ ja_cc2_toxic/ ja_cc2_toxicity_filtered/ 70 8.4
> bash parallel_classify.sh ja_cc3/ toxic_scores3/ ja_cc3_toxic/ ja_cc3_toxicity_filtered/ 70 8.4
```

`ja_cc[123]_toxic/`に有害な`text`が、`ja_cc[123]_toxicity_filtered/`に無害な`text`が出力されます。`70`は使用するCPU core数です。CPU core数が`70`だと各々1時間程度かかるかもしれません。`8.4`は分類閾値です。閾値を上げるとprecisionが高くなりrecallが低くなる傾向にあり、閾値を下げるとprecisionが低くなりrecallが高くなる傾向にあります。


## 補足: DeBERTaモデルのHuggingFace形式からTensorRT形式への変換

`deberta.onnx`は、`convert_to_onnx.py`を使って、HuggingFace形式のDeBERTaモデル (`final_model/`) をTensorRT形式に変換することで生成しました。正確には、TensorRTをExecution Providerとして使ってNVIDIAハードウェアに最適化されたONNX形式にエクスポートしました。

```sh
> python3 convert_to_onnx.py
```

この結果、`deberta.onnx`が出来上がります。これを`batch_inference_onnx_logits.py`からを使うことで`text`に有害スコアを付与できます。


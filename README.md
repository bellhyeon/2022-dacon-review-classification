# 쇼핑몰 리뷰 평점 분류 경진대회
Dacon Basic | NLP | Accuracy<br>
쇼핑몰 리뷰 데이터셋을 이용하여 상품의 평점 (1점, 2점, 4점, 5점)을 분류
<br>[Competition Link](https://dacon.io/competitions/official/235938/overview/description)
* 주최/주관: Dacon
* **Private 4th, Score 0.71056**
***
## Structure
Train/Test data folder and sample submission file must be placed under **dataset** folder.
```
repo
  |——dataset
        |——train.csv
        |——test.csv
        |——sample_submission.csv
  train.py
  inference.py
  requirements.txt
```
***
## Development Environment
* Windows 10
* i9-10900X
* RTX 2080Ti 1EA
* CUDA 11.4
***
## Environment Settings

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-385/)

### Requirements
```shell
pip install --upgrade -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
***
## Train
```shell
python train.py
```
***
## Inference
```shell
python inference.py
```
***
## Solution
### 1. Preprocessing
```python
def clean(review: str) -> str:
    BMP_pattern = re.compile(
        "[" "\U00010000-\U0010FFFF" "]+", flags=re.UNICODE  # BMP characters 이외
    )

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )

    review = re.sub("([ㄱ-ㅎ ㅏ-ㅣ]+)", " ", review)  # 단일 자음, 모음 제거
    review = re.sub(
        "[-=+,#/\?:^$.@*& ​\^\"※ᆢㆍ;~♡!』<>.,‘|\(\)\[\]`'…》\”\“\’·]", " ", review
    )  # 특수문자 제거

    review = BMP_pattern.sub(r"", review)
    review = emoji_pattern.sub(r"", review)
    review = " ".join(review.split())
    return review
```
### 2. Training Details
* backbone: kykim/electra-kor-base
* 5 fold(stratified)
* lr: 1e-5
* epochs: 10
* batch size: 32
* max length: 128
* optimizer: AdamW
* scheduler: cosine schedule with warmup
* dropout rate (classifier): 0.3
* label smoothing: 0.1
***
## Tried Techniques
* back-translation
* data-augmentation (random swap)
* spell / spacing correction
***
## Reference
[kykim/electra-kor-base](https://github.com/kiyoungkim1/LMkor)
```
@misc{kim2020lmkor,
  author = {Kiyoung Kim},
  title = {Pretrained Language Models For Korean},
  year = {2020},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/kiyoungkim1/LMkor}}
}
```
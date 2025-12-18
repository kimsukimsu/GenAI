## Project description

This project is for "Generative AI" class 2025-2

https://github.com/iontail/gdl_term

## 📦 Installation

0.  *** puzzlemix repo 참고 ***
1.  **리포지토리 클론:**
    ```bash
    git clone https://github.com/kimsukimsu/GenAI.git
    cd GenAI
    ```

2.  **Conda 환경 생성 및 PyTorch 설치:**
    이 코드는 `Python 3.10` 및 `CUDA 12.1` 환경에서 테스트되었습니다.

    ```bash
    # 1. Conda 환경 생성
    conda create -n gdtp python=3.10 -y
    
    # 2. 환경 활성화
    conda activate gdtp
    
    # 3. PyTorch (CUDA 12.1) 설치
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

3.  **추가 라이브러리 설치:**
    ```bash
    pip install gco-wrapper matplotlib numpy six
    pip install transformers ftfy regex scipy matplotlib scikit-learn #for pca_clip.py
    ```

---

## 👟 Training

아래는 `preactresnet18` 아키텍처를 사용하여 CIFAR-100 데이터셋으로 모델을 학습시키는 예시 명령어입니다.

Mixing Strategies : ['warmup', 'linear', 'step', 'concat', 'no_aug']

```
bash script/train.sh
```

```
python main.py --dataset cifar100 \
    --train_org_dir "original cifar 100 train dir" \
    --train_aug_dir "custom diffusemix dir (blended)" \
    --test_dir "original cifar100 test dir" \
    --root_dir output/test \
    --labels_per_class 500 \
    --arch preactresnet18 \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --decay 0.0001 \
    --epochs 300 \
    --schedule 100 200 \
    --gammas 0.1 0.1 \
    --mix_strategy concat
    --train vanilla
```

아래는 "openai/clip-vit-base-patch32"를 이용하여 original을 기준으로 pca시각화 하는 코드입니다.


```
python pca_clip.py \
    --dirs /path/to/original /path/to/blended /path/to/generated \
    --labels Original Blended Generated \
    --output distribution_graph.png
```



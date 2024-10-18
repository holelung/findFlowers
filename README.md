
# findFlowers

## Streamlit 링크
https://findflowers-eenqfbmjgz6xk5nadmbunf.streamlit.app/

![image](https://github.com/user-attachments/assets/4d74d677-44de-4b1d-b4c3-fa01a28cf30c)

---

## 프로젝트 소개
findFlowers는 PyTorch의 ResNet-50 모델을 사용하여 꽃의 종류를 분류하는 어플리케이션입니다. 사용자는 사진을 업로드하면 해당 꽃의 종류를 예측할 수 있습니다.

### Colab 링크
ResNet-50을 사용한 꽃 분류 모델을 학습시키는 코드와 관련된 Colab 파일은 [여기]( https://colab.research.google.com/drive/1GhHZq4Ce8eWOfYxL2nZBQnfHVuvYJbGz?usp=sharing )에서 확인할 수 있습니다.

## 설치 및 실행 방법

### Backend 설치
1. Python과 PyTorch를 설치합니다.
2. `requirements.txt` 파일에 있는 종속성 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```
3. Streamlit 앱을 실행합니다:
   ```bash
   streamlit run streamlit_app.py
   ```

### 모델 파일
1. 사전 학습된 ResNet-50 모델이 `models` 디렉토리에 저장되어 있으며, 이를 통해 꽃 이미지를 분류합니다.

---

## 기술 스택
- **Library**: PyTorch
- **Model**: ResNet-50 (사전 학습된 모델 사용)
- **Frontend**: Streamlit
- **Language**: Python

---

## 주요 기능 설명
- 사용자 업로드 이미지의 꽃 종류 분류
- ResNet-50을 이용한 이미지 분석 및 예측
- Streamlit을 통해 사용자에게 인터페이스 제공

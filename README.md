# MNIST 손글씨 인식 시스템 (MNIST Handwriting Recognition System)

MNIST 데이터셋을 활용해 신경망을 학습시키고, 실시간으로 손글씨를 인식하는 기능을 제공하는 종합 딥러닝 애플리케이션입니다.

## 주요 기능 (Features)

- **MNIST 데이터셋 학습**: 60,000장의 학습 이미지를 로드하여 학습 진행
- **신경망 아키텍처**: 드롭아웃(Dropout) 규제가 포함된 4계층 신경망 구조
- **모델 지속성 (Model Persistence)**: 한 번 학습한 모델을 저장하여 향후 재사용
- **대화형 드로잉 캔버스**: 마우스로 숫자를 그리고 즉시 인식 결과 확인
- **실시간 예측**: 예측된 숫자와 함께 신뢰도(확률) 점수 출력
- **학습 시각화**: 정확도 및 손실(Loss) 곡선 그래프 저장

## 요구 사항 (Requirements)

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
opencv-python>=4.5.0
Pillow>=8.0.0
```

## 사용 방법 (Usage)

### 기본 실행

```bash
python mnist_handwriting_recognition.py
```

프로그램 실행 흐름:
1. 학습된 모델이 존재하는지 확인
   - 모델이 없으면 MNIST 데이터셋으로 새로 학습 진행 (CPU 기준 약 2~3분 소요)
   - 학습된 모델을 향후 사용을 위해 자동 저장
2. 손글씨 인식을 위한 대화형 드로잉 캔버스 실행

### 드로잉 조작법

드로잉 캔버스가 나타나면 다음 단축키를 사용할 수 있습니다:
- **마우스 왼쪽 클릭 후 드래그**: 숫자(0-9) 그리기
- **스페이스바 (SPACE)**: 그린 숫자 인식하기
- **C 키**: 캔버스 지우기 (다시 그리기)
- **Q 키**: 프로그램 종료

### 생성되는 출력 파일

- `mnist_model.h5` - 학습된 신경망 모델 파일
- `training_history.png` - 학습 정확도 및 손실 곡선 이미지
- `prediction_result.png` - 각 예측 결과 이미지

## 프로그램 구조 (Program Structure)

### 파트 1: 데이터 로딩 (Data Loading)
- MNIST 데이터셋 로드 (학습용 60,000장, 테스트용 10,000장)
- 픽셀 값을 [0, 255]에서 [0, 1] 범위로 정규화
- 28×28 이미지를 784차원 벡터로 평탄화(Flatten)
- 라벨을 원-핫 인코딩(One-hot encoding)으로 변환

### 파트 2: 모델 구조 (Model Architecture)
```
입력층 (784 뉴런)
    ↓
밀집층 (Dense, 128 뉴런, ReLU) + Dropout(0.2)
    ↓
밀집층 (Dense, 64 뉴런, ReLU) + Dropout(0.2)
    ↓
밀집층 (Dense, 32 뉴런, ReLU) + Dropout(0.2)
    ↓
출력층 (Dense, 10 뉴런, Softmax)
```

### 파트 3: 모델 학습 (Training)
- Adam 옵티마이저를 사용한 15 에포크(Epoch) 학습
- 배치 크기(Batch size): 128
- 손실 함수: 범주형 교차 엔트로피 (Categorical Crossentropy)
- 검증 데이터: 테스트셋을 검증 데이터로 활용

### 파트 4: 모델 평가 (Evaluation)
- 테스트셋을 통한 정확도 및 손실 측정
- 학습 곡선 시각화 및 저장

### 파트 5: 대화형 인식 (Interactive Recognition)
- OpenCV를 활용한 커스텀 드로잉 인터페이스 제공
- 그려진 이미지를 MNIST 표준 형식에 맞게 전처리
- 숫자 예측 및 신뢰도 출력

### 파트 6: 메인 실행 (Main Execution)
- 모델 생명주기 관리 (학습/로드/저장)
- 전체 파이프라인 조율

## 예상 성능 (Expected Performance)

학습 완료 후:
- **테스트 정확도**: 약 97~98%
- **학습 시간**: CPU 기준 약 2~3분

## 인식 작동 방식 (How the Recognition Works)

1. **사용자 입력**: 캔버스에 숫자를 마우스로 작성
2. **전처리**: 
   - 28×28픽셀로 크기 조절 (MNIST 표준)
   - 색상 반전 (흰 바탕 → 검은 바탕)
   - 픽셀 값 [0, 1] 정규화
3. **예측**: 전처리된 이미지를 학습된 신경망에 입력
4. **출력**: 예측된 숫자와 신뢰도 퍼센트 표시

## 예시 실행 화면 (Example Session)

```
==============================================================
MNIST HANDWRITING RECOGNITION SYSTEM READY
==============================================================

============================================================
HANDWRITING RECOGNITION MODE
============================================================
Instructions:
  - Left-click and drag to draw a digit
  - Press 'SPACE' to recognize the digit
  - Press 'C' to clear the canvas
  - Press 'Q' to quit
============================================================

[드로잉 캔버스 창이 열림 - 사용자가 숫자 "5"를 그림]

[스페이스바 누름 후]
============================================================
Predicted Digit: 5
Confidence: 0.9995 (99.95%)
============================================================

Draw another digit? (yes/no): no

Thank you for using MNIST Handwriting Recognition System!
```

## 문제 해결 (Troubleshooting)

**문제**: 캔버스 창이 열리지 않음
- **해결책**: OpenCV가 올바르게 설치되었는지 확인하세요: `pip install opencv-python`

**문제**: 모델 학습이 너무 오래 걸림
- **해결책**: GPU 가속이 지원되는 텐서플로우를 사용하세요: `pip install tensorflow[and-cuda]`

**문제**: 예측 정확도가 낮음
- **해결책**: 숫자를 더 선명하고 캔버스 중앙에 크게 그려보세요.

## 커스터마이징 (Customization)

`main()` 함수 내부에서 다음 파라미터를 수정할 수 있습니다:

```python
# 학습 에포크(Epoch) 수 변경
history = train_model(model, x_train, y_train, x_test, y_test, epochs=20)

# 배치 크기(Batch size) 변경
history = train_model(model, x_train, y_train, x_test, y_test, batch_size=64)

# build_model()에서 모델 구조 변경
# 레이어 추가, 뉴런 수 변경, 드롭아웃 비율 조정 등
```

## 성능 팁 (Performance Tips)

1. **최초 실행**: 모델 학습에 시간이 걸리지만 한 번 저장되면 이후에는 재사용됩니다.
2. **필기 요령**: 캔버스 중앙에 크고 명확하게 숫자를 적어주세요.
3. **다중 예측**: 저장된 모델을 재사용하므로 다시 학습할 필요가 없습니다.
4. **정확도**: MNIST 데이터셋 기준 약 97% 이상의 테스트 정확도를 달성합니다.

## 참고 자료 (References)

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Deep Learning Basics: Neural Networks, Backpropagation, Dropout Regularization

---

**작성자 (Author)**: AI Assistant  
**생성일 (Created)**: 2024  
**라이선스 (License)**: Open Source
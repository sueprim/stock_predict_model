# LSTM 모델을 활용한 주가 예측

## **프로젝트 개요**
이 프로젝트는 (제가 jyp 주식을 가지고 있는데 어쩌구 저쩌구~ 스토리 쓰기)**JYP 엔터테인먼트 (KOSDAQ: 035900)** 의 주가를 예측하기 위해 LSTM(Long Short-Term Memory) 모델을 활용합니다. 데이터는 **2000-01-01부터 현재까지**의 기간을 포함하며, Yahoo Finance API를 통해 다운로드되었습니다. 모델은 **테스트 데이터셋(`Predicted Price`)** 과 **미래 주가(`Future Prediction`)** 를 각각 예측합니다. 특히, 미래 예측은 2024-01부터 2025-12까지의 기간을 대상으로 수행됩니다.

---

## 예측 결과
![image](https://github.com/user-attachments/assets/0c88dd23-2a2f-440f-af66-a2851d7e7779)

## **주요 코드**

### **1. 데이터 수집 및 전처리
```python3
stock_symbol = '035900.KQ'  # JYP 엔터테인먼트 KOSDAQ 코드
data = yf.download(stock_symbol, start='2000-01-01', end=today)

data = data[['Close']]  # 'Close' 열만 사용
data.dropna(inplace=True)  # 결측치 제거
```
- **설명**
  - Yahhoo Finance API를 사용하여 2000년부터 현재까지의 주가 데이터를 다운로드하였습니다.
  - 종가(Close)만을 사용하여 분석의 일관성을 유지했습니다.
  - 데이터셋에서 결측값을 제거하여 학습 및 예측의 정확도를 보장했습니다
---
### **2. 데이터 전처리
```python3
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
```
- **설명**
  -	데이터를 0과 1 사이로 정규화(Min-Max Scaling)하여 LSTM 모델 학습에 적합하도록 변환했습니다.
  -	정규화를 통해 학습 안정성을 높이고 모델 성능을 개선했습니다.
---
### **3. 학습 및 테스트 데이터 분리
```python3
train_data_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_data_len]
test_data = scaled_data[train_data_len:]

X_train, y_train = create_dataset(train_data, time_step=60)
X_test, y_test = create_dataset(test_data, time_step=60)
```
- **설명**
  -	전체 데이터의 80%를 학습 데이터로, 20%를 테스트 데이터로 분리했습니다.
  -	60일의 데이터(time_step=60)를 기반으로 다음 날의 주가를 예측하도록 데이터셋을 구성했습니다.
---
### **4. 모델 생성
```python3
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
```
- **설명**
  -	LSTM 레이어를 두 개 사용하여 시계열 데이터를 처리하도록 설계했습니다.
  -	첫 번째 LSTM 레이어: 100개의 뉴런과 출력 시퀀스를 유지하는 return_sequences=True 옵션.
  -	두 번째 LSTM 레이어: 50개의 뉴런과 출력 시퀀스를 제거하는 return_sequences=False 옵션.
  -	Dense 레이어: 마지막 출력을 단일 값으로 변환합니다.
  -	Adam 옵티마이저와 mean_squared_error 손실 함수를 사용하여 학습 효율성을 극대화했습니다.
---


## **주요 특징**

### **1. Predicted Price**
- **설명**:
  - `Predicted Price`는 학습 데이터 중 80%를 학습에 사용하고, 나머지 20%를 테스트 데이터로 분리하여 예측한 결과입니다.
- **관찰된 특징**:
  - 과도한 학습으로 인해 **over fitting**이 발생해 더 적은 epoch 때보다 예측 성능이 떨어진다
  - 주가의 **전체적인 추세**(상승/하락)는 비교적 잘 반영되었습니다.
  - 급격한 상승 또는 하락과 같은 **단기적인 변동성**을 제대로 포착하지 못하는 경향이 있습니다.
  - 실제 데이터보다 약간 느리게 반응(지연 효과)이 나타나는 경우가 있습니다.
- **장점**:
  - **중장기적인 추세**를 예측하는 데 적합하며, 대규모 패턴 변화를 잘 따라갑니다.
- **단점**:
  - 세부적인 변동이나 단기적인 변화 예측에는 한계가 있습니다.

---

### **2. Future Prediction**
- **설명**:
  - `Future Prediction`은 학습된 LSTM 모델을 기반으로 2024-01부터 2025-12까지 24개월 동안의 미래 주가를 예측한 결과입니다.
- **관찰된 특징**:
  - 미래 예측값이 **평탄화(flatline)** 되는 경향을 보입니다.
  - 모델이 학습 데이터의 **최근 패턴에 과도하게 의존**하여 장기적으로 안정적이나 단조로운 결과를 제공합니다.
  - 주가의 변동성(volatility)이 제대로 반영되지 않았습니다.
- **단점**:
  - 모델이 학습 데이터 외부의 새로운 패턴을 예측하지 못하는 한계를 드러냈습니다.
  - 현실적으로 발생할 수 있는 급격한 상승 또는 하락을 반영하지 못했습니다.
  - 장기적인 예측에서는 불확실성이 커지며, 신뢰성이 떨어진다.

---

## **한계와 보완점**

### **한계**
1. **Predicted Price**:
   - 단기적인 급격한 변동성을 제대로 반영하지 못함.
   - 모델의 응답이 실제 데이터보다 느린 경우 발생.

2. **Future Prediction**:
   - 장기 예측에서 데이터가 평탄화되는 경향이 뚜렷함.
   - 모델이 최근 데이터를 지나치게 학습하여, 새로운 패턴을 반영하지 못함.

### **보완점**
1. **모델 개선**:
   - Attention Mechanism을 추가하여 데이터의 중요한 특징을 더 잘 학습할 수 있도록 개선.
   - Transformer 기반 모델 활용을 고려해 장기 시계열 예측 성능을 강화.
2. **데이터 보강**:
   - 외부 요인(거래량, 경제 지표, 시장 뉴스 등)을 추가하여 학습 데이터를 확장.
3. **예측 전략 변경**:
   - 단기적인 예측을 반복적으로 수행하여 변동성을 반영하는 방식으로 예측 구조를 변경.
4. **하이퍼파라미터 튜닝**:
   - epoch 횟수, 레이서 수와 같은 **하이퍼 파라미터**값 최적화.
5. **더 큰 데이터셋 사용**:
   - 더 긴 기간의 데이터를 사용

---

## **결론**
이 프로젝트는 LSTM 모델을 활용하여 주가를 예측하는 기본적인 방법을 구현하였습니다. `Predicted Price`는 테스트 데이터 내에서 비교적 정확한 추세를 반영했지만, 단기적인 세부 변동을 예측하지 못하는 단점이 있었습니다. `Future Prediction`은 장기적으로 단조로운 결과를 보이며, 현실적인 변동성을 반영하지 못했습니다.

향후, 모델 구조와 데이터 보강을 통해 예측 성능을 개선하고, 실질적인 주가 예측에서 활용 가능성을 높일 수 있을 것입니다.

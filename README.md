# 거시경제 레짐 탐지를 활용한 전술적 자산배분

## 개요

이 프로젝트는 Oliveira et al. (2025)의 방법론을 기반으로 거시경제 레짐 탐지를 활용한 전술적 자산배분 전략을 구현합니다. 시스템은 FRED-MD 데이터에서 서로 다른 경제 레짐을 식별하고, 이러한 레짐을 활용하여 ETF 자산 전반의 포트폴리오 배분을 최적화합니다.

## 프로젝트 구조

### 핵심 구현 파일

1. **Section3.py** - 레짐 분류 시스템
   - Stage 1: L2 k-means 클러스터링(k=2)으로 이상치 월 식별
   - Stage 2: 일반적인 월에 대한 코사인 k-means와 엘보우 방법으로 최적 k 결정
   - 로그 스케일 매핑을 사용한 확률적 레짐 분포
   - 레짐 전이 확률 행렬 계산

2. **Section5_step0_ETF_Loader.py** - ETF 데이터 로딩 및 전처리
   - ETF 수익률 데이터 로드 및 정렬
   - 데이터 품질 및 정렬 문제 처리

3. **Section5_step1.py** - 동적 레짐 확률 추정
   - 시변 레짐 확률을 위한 식(6) 구현
   - 레짐 전이 행렬 처리

4. **Section5_step2.py** - 리지 회귀 예측
   - 레짐별 리지 회귀 모델
   - 람다 매개변수 선택을 위한 교차 검증
   - 표본 외 예측

5. **Section5_step3.py** - 예측 집계
   - 레짐 확률을 사용하여 레짐별 예측 결합
   - 최종 수익률 예측 생성

6. **Section5_step4.py** - 포트폴리오 가중치 최적화
   - 롱온리 포트폴리오 제약
   - 예측된 수익률 기반 가중치 최적화

7. **Section5_step5.py** - 백테스팅
   - 전략의 성과 평가
   - 샤프 비율, 최대 낙폭 및 기타 지표 계산

### 유틸리티 스크립트

- **check_pca_components.py** - PCA 분석 및 시각화
- **etc_regime_visualization.py** - 레짐 분석 시각화
- **etc_regime_stats_transformed.py** - 변환된 레짐 데이터의 통계 분석
- **etc_visualize_backtest.py** - 백테스팅 결과 시각화
- **etc_visualize_transition_matrix.py** - 전이 행렬 히트맵 시각화

## 데이터 요구사항

- **FRED-MD_2024m12.csv** - 연방준비제도 경제 데이터(월별)
- **etf_bom_returns_*.csv** - ETF 수익률 데이터 파일
- **FRED-MD_updated_appendix.pdf** - FRED-MD 데이터 변환 문서

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

분석 파이프라인을 순서대로 실행:

```bash
# 1. 레짐 분류
python Section3.py

# 2. ETF 전략 구현
python Section5_step0_ETF_Loader.py
python Section5_step1.py
python Section5_step2.py
python Section5_step3.py
python Section5_step4.py
python Section5_step5.py

# 3. 시각화 (선택사항)
python etc_visualize_backtest.py
python etc_regime_visualization.py
python etc_visualize_transition_matrix.py
```

## 출력 파일

파이프라인은 여러 중간 및 최종 출력 파일을 생성합니다:

- **section3_1_regime_labels.csv** - 각 월의 레짐 분류
- **section3_2_regime_probabilities.csv** - 확률적 레짐 할당
- **section3_3_transition_matrix.csv** - 레짐 전이 확률
- **section5_step*_*.csv** - 포트폴리오 최적화의 다양한 중간 결과
- **backtest_metrics_*.csv** - 최종 백테스팅 성과 지표
- **.png 파일** - 다양한 시각화 출력

## 방법론

프로젝트는 2단계 접근법을 따릅니다:

1. **레짐 탐지**: PCA 변환된 거시경제 지표에 대한 비지도 클러스터링을 사용하여 구별되는 경제 레짐 식별
2. **포트폴리오 최적화**: 레짐 의존적 예측 모델을 적용하여 ETF 배분 가중치 최적화

## 의존성

- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0

## 한계점

이 구현은 다음과 같은 한계점을 가지고 있습니다:

1. **극단적 이상치 처리**: 2020년 4월의 극단적 이상치를 시계열 분석에서 제외했습니다. 이는 COVID-19 팬데믹으로 인한 비정상적인 시장 상황을 반영한 것이지만, 완전한 데이터를 사용하지 못한 한계가 있습니다.

2. **데이터 기간 제약**: 적합한 결측치 처리 방법에 대한 지식적 한계로 인해 결측치가 없는 1992년부터의 데이터만 사용했습니다. 이로 인해 이전 33년치의 귀중한 역사적 데이터를 불가피하게 활용하지 못했습니다.

3. **거래비용 미반영**: 백테스팅 과정에서 거래비용, 슬리피지, 시장 충격 비용 등 실제 거래에서 발생하는 비용을 고려하지 못했습니다. 이는 실제 성과가 백테스트 결과보다 낮을 수 있음을 의미합니다.

4. **이상치 감지 민감도**: 2011-2012년 기간의 이상치 감지가 원논문 대비 더 민감하게 작동했습니다. 이는 파라미터 튜닝이나 데이터 전처리 과정의 미세한 차이에서 기인할 수 있습니다.

5. **실시간 구현 제약**: 이 구현은 과거 데이터를 활용한 백테스팅에 중점을 두고 있어, 실시간 거래 시스템으로의 전환을 위해서는 추가적인 개발이 필요합니다.

6. **Ridge 모형의 조건부확률(EQ.12)이 Regime_t인지, Regime_t+1인지 불분명해서 Regime_t로 설정함**: 학습 데이터의 손실 때문에 논문에서는 Regime_t라고 두고 모형을 작성한 것처럼 보이기는 한다. 관련 지식이 부족해서 모르겠다..

## 참고문헌

"Tactical Asset Allocation with Macroeconomic Regime Detection" (Oliveira et al., 2025)에 설명된 방법론 기반

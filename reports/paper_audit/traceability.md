# 논문과 코드의 대응표

기준 커밋: `9439107b31f20c4b731ec1dc29f13c178beb311d`. 이 표는 최종 감사 과정에서 실행 근거와 함께 갱신한다. 개별 수식 일치는 전체 파이프라인의 정확성을 뜻하지 않는다.

| 항목 | 원문 | 코드 위치 | 기대 동작과 현재 관측 | 판정 | 근거 |
|---|---|---|---|---|---|
| DATA | 4.1~4.2 p.7 | Section3.py:10~95; Step1:14~68; Step2:23~78 | 변환·표준화·PCA는 존재. 기간·그룹6 유지·결측 삭제 정책이 원문과 다름 | 부분일치 | INVENTORY |
| ALG1 | Algorithm 1 p.5 | Section3.py:172~207; Step1:179~198 | 두 단계 군집 구현. 전표본 elbow와 rolling r=5의 적용 범위 구분 필요 | 부분일치 | INVENTORY |
| EQ01 | 식 (1) p.6 | Section3.py:271~280; Step1:127~135 | 거리에서 정규화 확률로 변환. 정상·퇴화 입력 실험 대상 | 검증불가 | INVENTORY |
| EQ02 | 식 (2) p.6 | Section3.py:315 | Pmax 계산 | 일치 | INVENTORY |
| EQ03 | 식 (3) p.6 | Section3.py:316~317; Step1:212~215 | 경계 확률 클리핑으로 극한을 근사 | 부분일치 | INVENTORY |
| EQ04 | 식 (4) p.6 | Section3.py:317~323; Step1:215~217 | log2 매핑 후 합 정규화 | 부분일치 | INVENTORY |
| EQ05 | 식 (5) p.6 | Section3.py:501~515; Step1:219~230 | 전표본은 전체 빈도, rolling은 출발 전이 수로 정규화 | 논문모호 | AMB01 |
| EQ06 | 식 (6) p.13 | Step1:216~217 | 원소 합 정규화 | 일치 | INVENTORY |
| EQ07 | 식 (7) p.13 | Step1:232~234 | p@E 연산과 행렬 방향 | 부분일치 | INVENTORY |
| EQ08 | 식 (8) p.14 | 해당 Naive/BL 실행 경로 없음 | 다음 레짐 argmax를 해당 예측기에 연결해야 함 | 미구현 | INVENTORY |
| EQ09 | 식 (9) p.14 | 없음 | Naive 조건부 Sharpe 예측 | 미구현 | INVENTORY |
| EQ10 | 식 (10) p.14 | 없음 | Naive 조건부 평균/표준편차 | 미구현 | INVENTORY |
| EQ11 | 식 (11) p.14 | 없음 | BL 조건부 view | 미구현 | INVENTORY |
| EQ12 | 식 (12) p.14 | Step2:247~287 | regime_tau의 PCA_tau로 R_(tau+1) 학습 | 논문모호 | AMB03 |
| EQ13 | 식 (13) p.14 | Step2:169~199,263~287 | Ridge와 절편, lambda LOOCV 추가 | 부분일치 | INVENTORY |
| EQ14 | 식 (14) p.15 | Step3:329~343; Step4:241~253 | R1~R5의 확률 가중 합. 단계 간 번호 일치 확인 필요 | 부분일치 | AMB04, INVENTORY |
| EQ15 | 식 (15) p.15 | Step4:256~270 | lo 분모만 구현 | 부분일치 | INVENTORY |
| EQ16 | 식 (16) p.15 | 없음 | lns | 미구현 | INVENTORY |
| EQ17 | 식 (17) p.15 | 없음 | los | 미구현 | INVENTORY |
| EQ18 | 식 (18) p.15 | Step4:256~285 | 양수 상위 l개 정규화. 비유한 입력 검사 필요 | 부분일치 | INVENTORY |
| EQ19 | 식 (19) pp.15~16 | 없음 | BL posterior/배분 | 미구현 | INVENTORY, AMB08 |
| REGIME_ANALYSIS | 4.3~4.6, Figures 2~6, Table 1 | etc_regime_stats_transformed.py; etc_regime_visualization.py | 지표 변환·평균·min-max·레짐 시각화 일부. GMM/NBER 비교 없음 | 부분일치 | INVENTORY |
| CONDITIONAL_TRANSITION | 4.6, Figures 5~6 | etc_visualize_transition_matrix.py | E 표시와 지속성 막대만 존재. 조건부 전이 정규화와 네트워크 없음 | 미구현 | INVENTORY |
| MATCHING | p.2 | Step1:149~165; Step2:151~165 | 창별 PCA 좌표를 그대로 cosine 비교 | 검증불가 | INVENTORY, AMB05 |
| MODELS | 5.2 | Step2~3 | Ridge만 구현 | 부분일치 | INVENTORY |
| SIZING | 5.3 | Step4 | lo l=2/3/4만 구현, mx 없음 | 부분일치 | INVENTORY |
| ETF_DATA | Section6,Table2 p.16 | ETF Loader:42~66,92~158 | Yahoo 조정가격으로 월초 forward return 생성, WRDS와 다른 자료 | 부분일치 | INVENTORY |
| ROLLING | Section6 p.16 | Step1:175; Step2:229~287 | 48행 sliding, 날짜·공개 시점 확인 필요 | 부분일치 | INVENTORY |
| METRICS | 5.4,Tables4~6 | Step5:51~69 | 다섯 지표 구현, 정의·경계·단위 확인 필요 | 부분일치 | INVENTORY |
| CONTROLS | Figures7~9,Table3 | 없음 | 무작위 레짐·paired t-test·Nemenyi | 미구현 | INVENTORY |
| BENCHMARKS | Tables4~6 | Step5:43~44,73~74 | SPY/EW만 있으며 MVO 없음 | 부분일치 | INVENTORY |
| VOL_SCALING | Figures10~13 | Step5:77 comment; etc_visualize_backtest.py:41~45 | 10% 조정은 주석뿐이며 그림은 원수익률 누적곱 | 미구현 | INVENTORY |

`StepN`은 `Section5_stepN.py`를 뜻한다. `ETF Loader`는 `Section5_step0_ETF_Loader.py`다. 실제 파일·함수의 정확한 범위와 SHA-256은 `evidence/inventory/manifest.json` 및 개별 실험 기록에 보존한다. AMB 항목은 `methodology.md`의 원문 불명확함 목록이다.

## 데이터 흐름과 실행 분기

1. Section3는 FRED를 전표본으로 전처리·PCA·군집하고 CSV/NPZ를 쓴다. 최상위 실행이므로 import도 전체 계산과 그림을 실행한다.
2. ETF 로더는 Section3의 달력 CSV를 우선 읽고 없으면 FRED 날짜로 대체한다. yfinance가 없으면 pip 설치를 시도한다. `_make_synth_prices`라는 이름과 달리 현재 함수는 실제 Yahoo 가격을 가져온다.
3. Step1은 ETF와 교집합하지 않은 거시경제 달력에서 별도로 rolling 군집·확률을 계산한다.
4. Step2는 ETF 달력과 먼저 교집합하고 별도 seed·별도 초기 매칭 상태로 rolling 군집·회귀를 계산한다. 두 단계가 같은 R1을 공유한다는 보장이 저장되지 않는다.
5. Step3는 기존 Step1/2 파일을 우선 읽는다. Step2가 없으면 최근 10개 창을 재생성하며 ETF가 없으면 합성 수익률을 저장한다. Step2만 있고 Step1은 없으면 `macro_ctx=None`이 후속 함수로 전달되는 경로가 있다.
6. Step4는 집계 파일이 없으면 자체 Step1/2/집계를 만든다. ETF 부재 시 합성 수익률을 만든다. 정상 입력 경로에서는 lo 비중만 계산한다.
7. Step5는 비중의 tp1에 해당하는 ETF 수익률을 곱한다. etc_visualize_backtest는 이 출력과 다른 파일명 및 다른 메트릭 열 이름을 읽는다.

## 보존된 산출물의 한계

추적 NPZ 두 개에는 원 입력 해시·생성 커밋·난수 설정·PCA 좌표 정의를 함께 저장한 메타데이터가 없다. 존재한다는 이유만으로 지금 기준 커밋의 결과라고 단정할 수 없다. CSV/PNG는 기존 `.gitignore`에 의해 대체로 추적되지 않는다. 논문과 같은 WRDS ETF 수익률은 제공되지 않았다.

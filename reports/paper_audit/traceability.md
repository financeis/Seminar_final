# 논문과 코드의 최종 대응표

기준 커밋: `9439107b31f20c4b731ec1dc29f13c178beb311d`. 사용자 지정 PDF의 전체 방법론을 32개 ID(식19·알고리즘1·방법12)에 연결했다. `StepN`은 `Section5_stepN.py`다. 줄은 원본 파일의 1기반 번호이며 해시는 `environment.json`, 더 세밀한 실행 범위는 개별 증거의 source_refs에 있다. `<module>`은 최상위 실행문이다.

판정은 해당 행의 범위에만 적용한다. 정상 정의역의 식 계산이 **일치**해도 단계 간 정체성·입력·경계가 맞는다는 뜻은 아니다. **논문모호**는 원문만으로 하나의 의도/정책을 확정하기 어려운 항목이다. 미구현 부재는 이름 검색만이 아니라 원11파일의 함수·실행·입출력을 조사한 INVENTORY와 보고서의 범위 근거를 사용한다.

| ID | 원문 | 기대 동작 | 코드 위치 또는 부재 증명 | 실제 관측 | 판정 | 실험·발견 |
|---|---|---|---|---|---|---|
| DATA | p.7 §4.1–4.2 | 그룹6 제외·지정 기간·변환·95% PCA | Section3.py:10–95 prepare_data_if_needed; Step1:17–82; Step2:23–90; Step3:60–102; Step4:39–75 | 주 전처리·tcodes 일치, 변수121/그룹6금리17 유지와 기간·전기간 선별은 변경 | 부분일치 | F02 F04 F07 F17 F22; DATA-GROUP6 DATA-TCODE DATA-FUTURE-SELECTION PAPER-COVERAGE |
| ALG1 | pp.4–5 §3.1 Algorithm1 | L2 k2 작은 군집 R0, 나머지 cosine/elbow | Section3.py:176–207 <module>; Step1:179–199 <module> | 전표본 elbow·rolling r5 구현, 동일크기 선택 일치; 경계 초기화 실패, 단계별 분할 다름 | 부분일치 | F01 F18; REG-008 REG-009 REG-018 REG-020 |
| EQ01 | p.6 식1 | 거리별 (1−d/Σd)/(K−1) | Section3.py:271–279 probs_from_distances; Step1:127–135 fuzzy_probs_from_dist | 정상거리/L2 제곱근 일치; 영거리·K1은 원식 미정 | 일치 | F20; REG-002 REG-003 REG-006 |
| EQ02 | p.6 식2 | 일반 레짐 Pmax=max P_i | Section3.py:315 <module>; Step1:212 <module> | 최대값 계산 일치 | 일치 | REG-004 |
| EQ03 | p.6 식3 | q0의 0/.5/1 경계 조건 | Section3.py:316–317 <module>; Step1:213–215 <module> | 보통 입력 조건 일치, p0=1에서 유한 클리핑 | 부분일치 | F20; REG-004 REG-005 |
| EQ04 | p.6 식4 | q0=−Pmax log2(1−p0), 합 정규화 | Section3.py:317–323 <module>; Step1:215–217 <module> | 정상 log2 매핑 일치, 끝점 극한은 근사 | 부분일치 | F20; REG-004 REG-005 |
| EQ05 | p.6 식5 | N(i→j)/N(i), 마지막 관측 분모 문제 명시 | Section3.py:503–517 <module>; Step1:220–229 <module> | 전표본 전체 발생 분모 vs rolling 출발 횟수 분모/빈행 자기전이 | 논문모호 | F18 F19; REG-010 REG-012 |
| EQ06 | p.13 식6 | 원소 합으로 확률 정규화 | Step1:216–217 <module> | 정상 입력의 비음수·합1 일치 | 일치 | REG-004 |
| EQ07 | p.13 식7 | 출발행·도착열 E에 p@E | Step1:231–232 <module> | 방향·rolling 확률 보존 일치; 확률 의미 정합성은 F01 별도 | 일치 | F01 F19; REG-011 REG-022 |
| EQ08 | p.14 식8 | Naive/BL용 다음 레짐 argmax | 없음: 원11파일 전체 함수·실행·입출력 목록 INVENTORY | 현재 hard label argmax는 있으나 Naive/BL 예측 연결 없음 | 미구현 | F10; INVENTORY PAPER-COVERAGE |
| EQ09 | p.14 식9 | Naive 조건부 Sharpe 예측 | 없음: 원11파일 전체 조사 INVENTORY | Naive 예측기 없음 | 미구현 | F10; INVENTORY |
| EQ10 | p.14 식10 | 조건부 평균/표준편차 | 없음: 원11파일 전체 조사 INVENTORY | Naive 조건부 estimator 없음 | 미구현 | F10; INVENTORY |
| EQ11 | p.14 식11 | BL의 조건부 평균 view | 없음: 원11파일 전체 조사 INVENTORY | BL view 경로 없음 | 미구현 | F10 F25; INVENTORY |
| EQ12 | p.14 식12 | 단일 다음 수익 예측; X/y/레짐 시점 확정 필요 | Step2:258–286 <module>; README.md:124 | x_tau→R_(tau+1), regime_tau; 최근 R_t 사용. P_(t+1) 이후 결정이면 목표 상한 정상 | 논문모호 | F24 F25; FOR-007 DATA-RETURN-CAUSALITY |
| EQ13 | p.14 식13 | Ridge 제곱오차+λ계수벌점 | Step2:163–191 ridge_precompute_B_Hdiag/ridge_select_lambda_and_predict; Step2:258–286 | 고정λ 정규방정식 일치; 추가된 analytic LOOCV와 조건부 fallback에 문제 | 부분일치 | F03 F09; FOR-002 FOR-003 FOR-004 FOR-005 FOR-006 FOR-008 |
| EQ14 | p.15 식14 | 같은 레짐의 확률×예측을 R1~r 합산 | Step3:329–341 <module>; Step4:241–253 <module> | 보통 산식 일치; 공유 분할 없음, R0=1 fallback·날짜 계약 문제 | 부분일치 | F01 F05 F21; FOR-009 FOR-010 FOR-011 FOR-012 FOR-013 REG-022 |
| EQ15 | p.15 식15 | H∪L의 예측 절댓값 합 | Step4:256–270 long_only_weights | lo의 양수 H 분모만 구현; L 없음 | 부분일치 | F10; FOR-015 INVENTORY |
| EQ16 | p.15 식16 | lns 양/음수 각각 최대l 선택 | 없음: 원11파일 전체 조사 INVENTORY | lns 없음 | 미구현 | F10; INVENTORY |
| EQ17 | p.15 식17 | los 절댓값 상위l 선택 | 없음: 원11파일 전체 조사 INVENTORY | los 없음 | 미구현 | F10; INVENTORY |
| EQ18 | p.15 식18 | lo 양수 상위최대l 정규화 | Step4:256–270 long_only_weights | 정상 양수·부족·동률 노출 일치. 비양수면 현금; 비유한 입력 정책 별도 | 일치 | F23; FOR-015 FOR-016 |
| EQ19 | pp.15–16 식19 | BL posterior mean/비중화 설정 | 없음: 원11파일 전체 조사 INVENTORY | BL 없음; 원문의 allocation 해석·설정 미정 | 미구현 | F10 F25; INVENTORY PAPER-COVERAGE |
| REGIME_ANALYSIS | pp.7–13 §4.3–4.6 Figures2–6/Table1 | GMM/NBER 비교·지표 평균/min-max·레짐 해석 | etc_regime_stats_transformed.py:98–147 <module>; etc_regime_visualization.py:41–87 <module> | 지표 통계·레짐 산점도 일부, GMM/NBER 비교 없음 | 부분일치 | F12 F22; DATA-TCODE INVENTORY |
| CONDITIONAL_TRANSITION | pp.10–13 §4.6 Figures5–6 | offdiagonal/(1−e_ii), 대각0와 네트워크 | etc_visualize_transition_matrix.py:10–111 <module>; 조건부 산식은 원11파일에서 부재 | E heatmap·자기전이 막대만; 조건부 정규화/네트워크 없음 | 미구현 | F12 F19; INVENTORY REG-010 |
| MATCHING | p.2 서론 | 창 사이 레짐 의미·공통 좌표를 유지 | Step1:149–164 match_centroids_cosine; Step2:145–160 같은 함수; Step3:39–50 같은 함수; Step4:138–149 같은 함수(정적) | PCA 부호·회전·차원 절단에서 동일 중심 매칭 실패 | 불일치 | F08 F01; REG-013 REG-014 REG-015 REG-016 REG-017 REG-021 INVENTORY |
| MODELS | p.14 §5.2 | Naive/Ridge/BL | Step2:163–286; Step3:165–199 | Ridge만; normal/fallback grid·LOOCV 차이 | 부분일치 | F03 F09 F10; FOR-002 FOR-004 FOR-008 INVENTORY |
| SIZING | p.15 §5.3 | lo/lns/los/mx, l2/3/4 | Step4:256–290 long_only_weights 및 실행 | lo2/3/4만, mx 없음 | 부분일치 | F10 F23; FOR-015 FOR-016 INVENTORY |
| ETF_DATA | p.16 §6 Table2 | WRDS10ETF 월초수익·2000-02~2022-12 | Section5_step0_ETF_Loader.py:36–67 compute_bom_returns_from_daily; :98–139 _fetch_real_prices | 10ticker/보통월초수익 산식 일치, Yahoo/기간변경·실자료미제공 | 부분일치 | F04 F16 F24; DATA-BOM DATA-WRDS-YAHOO MET-016 |
| ROLLING | p.16 §6 | 연속48개월·월별 한달앞 예측·당시 정보 | Step1:175–242 <module>; Step2:225–286 <module> | 창내적합48행 확인, 결측삭제시50개월·미래변수screen. 공개/체결시각 미정 | 부분일치 | F02 F04 F24; DATA-CALENDAR DATA-FIT-SCOPE DATA-FUTURE-SELECTION DATA-RELEASE-VINTAGE |
| METRICS | pp.16–21 §5.4 Tables4–6 | Sharpe/Sortino/AvgDD/MaxDD/양수비율 | Step5:51–67 compute_drawdowns/perf_metrics | 지표5개 있음; 초기손실누락·관례차이·결측평가문제 | 부분일치 | F05 F06 F23; MET-004 MET-005 MET-006 MET-009 |
| CONTROLS | pp.17–19 §6 Figures7–9/Table3 | 무작위레짐·paired t-test·Nemenyi | 없음: 원11파일 전체 조사 INVENTORY | 모든 대조/유의성 실험 부재 | 미구현 | F11; INVENTORY PAPER-COVERAGE |
| BENCHMARKS | pp.19–21 Tables4–6 | SPY/EW/MVO 같은 평가기간 | Step5:43–44,73–74 <module>; MVO 부재는 원11파일 INVENTORY | SPY/EW 보통입력 손계산 일치; MVO 없고 결측집합 정책 차이 | 부분일치 | F05 F10 F24; MET-007 MET-010 MET-018 INVENTORY |
| VOL_SCALING | pp.21–23 Figures10–13 | 10%변동성 조정·누적로그수익 | Step5:77 주석; etc_visualize_backtest.py:41–44 <module>; 원11파일실행부재 | 주석뿐이며 실제 그림은 unscaled cumprod | 미구현 | F13 F14; MET-014 INVENTORY PAPER-COVERAGE |

## 원문 그림·표까지의 범위

PAPER-COVERAGE (`evidence/paper/coverage.json`)는 원문1–26쪽 읽기/시각 검토, 식1–19·Algorithm1을 기록한다. Figure1→DATA, Figures2–4→REGIME_ANALYSIS/확률식, Figures5–6→EQ05/CONDITIONAL_TRANSITION, Figures7–9→CONTROLS, Figures10–13→VOL_SCALING/MODELS/BENCHMARKS다. Table1→REGIME_ANALYSIS, Table2→ETF_DATA, Table3→CONTROLS, Tables4–6→METRICS/SIZING/MODELS다. 이 매핑은 검토 범위이며 그림 수치의 실증 재현 판정이 아니다.

## 코드 흐름과 실행 분기

1. Section3는 전표본 분석이다. 표준화·PCA·군집을 전체389행에 적합하고 CSV/NPZ를 쓴다. rolling 전략의 창내 적합과 구별한다.
2. ETF loader는 Section3 달력 또는 FRED 달력을 사용하고 Yahoo 가격을 받는다. `_make_synth_prices`는 현재 실제 다운로드에 위임한다. 이번 검증은 다운로드를 실행하지 않았다.
3. Step1은 거시경제 달력에서 확률을 계산한다. Step2는 ETF 달력과 먼저 교집합하고 별도 군집·초기 매칭 이력을 만든다. 같은 Ri를 보장할 공유 결과물이 없어 F01이다.
4. Step3는 Step1/2의 날짜 키로 합치고 Ri끼리 곱한다. 한쪽 날짜 누락/중복 검증이 없고, Step2만 있으면 macro_ctx=None 복구 실패가 난다.
5. Step3/4 fallback은 ETF가 없으면 합성자료를 공통 CSV명에 저장한다. 별도 출처표시가 없다. 실제 prefix를 실행했으며 이후 전체10창 fallback 학습은 수행하지 않았다.
6. Step4 정상 경로는 lo 비중을 계산한다. Step5는 sasdate_tp1 수익과 곱해 SPY/EW를 함께 평가한다. missing date와 active NaN 정책은 F05, 지표는 F06/F23이다.
7. etc_visualize_backtest는 Step5와 파일명·열명 계약이 달라 원 출력으로 실행 실패한다. 파일 연결을 통과하는 MET-017은 전체 논문 방법의 정확성 증명이 아니다.

## 검증 불가능한 실증 범위

DATA-RELEASE-VINTAGE, DATA-WRDS-YAHOO, MET-018은 SKIP이다. 실제 당시 이용가능성·WRDS 동일 가격·논문 표 전체 수익 및 유의성은 검증불가다. 관련 방법 행은 구현된 부분과 함께 부분일치로 표시했으며 이 SKIP를 성공으로 바꾸지 않았다. 추적 NPZ는 원 입력/생성 커밋/PCA 정체성 계보가 없어 지금 코드의 생성 증거로 간주하지 않는다. 전체 증거 ID별 사용처는 `evidence_index.md`에 있다.

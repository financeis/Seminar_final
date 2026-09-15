# 증거 색인

78개 실행 기록과 2개 문서 근거를 모두 연결한다. FAIL/SKIP는 발견 항목이 있어야 한다. PASS의 미연결 항목은 정상 성질 또는 독립 기준 자체검사이며 방법론 대조에 사용한다.

| ID | status | 위치 | 내용 | 사용처 |
|---|---|---|---|---|
| INVENTORY | PASS | `evidence/inventory/manifest.json` | 원11파일·입출력·부재 범위 | F08, F10, F11, F12, F13, F16, F24, F25 |
| PAPER-COVERAGE | PASS | `evidence/paper/coverage.json` | 원문26쪽·식19·알고리즘·그림13·표6 | F07, F10, F11, F12, F13, F22, F25 |
| DATA-SELF | PASS | `evidence/data_timing/results.json` | 독립 oracle 자체 검사 | F22 |
| DATA-TCODE | FAIL | `evidence/data_timing/results.json` | t-code 1~7 원 구현과 부록 산식 | F22 |
| DATA-PREP-3 | PASS | `evidence/data_timing/results.json` | Section3.py 전처리 일치 | F07 |
| DATA-PREP-5_step1 | PASS | `evidence/data_timing/results.json` | Section5_step1.py 전처리 일치 | F07 |
| DATA-PREP-5_step2 | PASS | `evidence/data_timing/results.json` | Section5_step2.py 전처리 일치 | F07 |
| DATA-PREP-5_step3 | PASS | `evidence/data_timing/results.json` | Section5_step3.py 전처리 일치 | F07 |
| DATA-PREP-5_step4 | PASS | `evidence/data_timing/results.json` | Section5_step4.py 전처리 일치 | F07 |
| DATA-GROUP6 | FAIL | `evidence/data_timing/results.json` | 논문 그룹6 제외와 실행 변수 집합 | F07 |
| DATA-CALENDAR | FAIL | `evidence/data_timing/results.json` | 48행 창과 연속 48개월·다음 달 구분 | F04 |
| DATA-README | FAIL | `evidence/data_timing/results.json` | README 기간·삭제 설명과 실제 전처리 | F04 |
| DATA-QUALITY-ORDER | PASS | `evidence/data_timing/results.json` | 변수 선별→보호변수 유지→관측 삭제 순서 | F02 |
| DATA-INFINITY | FAIL | `evidence/data_timing/results.json` | complete-case 결과의 유한값 보장 | F17 |
| DATA-CONSTANT | PASS | `evidence/data_timing/results.json` | rolling 표준화의 상수열 제거 | F17 |
| DATA-BOM | PASS | `evidence/data_timing/results.json` | 월초 가격과 수익률 저장 라벨 | F04 |
| DATA-BOM-GAP | FAIL | `evidence/data_timing/results.json` | 가격 달력이 통째로 비면 다개월을 한 달로 표시 | F04 |
| DATA-FUTURE-SELECTION | FAIL | `evidence/data_timing/results.json` | 미래 raw 결측만 바꾸면 과거 레짐 결정이 변하는가 | F02 |
| DATA-FUTURE-PROTECTED | PASS | `evidence/data_timing/results.json` | 보호변수 미래 결측 대조 실험 | F02 |
| DATA-RETURN-CAUSALITY | PASS | `evidence/data_timing/results.json` | 보유 시작 이후 가격과 이전 예측의 독립성 | F24 |
| DATA-FIT-SCOPE | PASS | `evidence/data_timing/results.json` | Section3 전체표본과 Step1/2 창내 적합 범위 구분 | F02 |
| DATA-RELEASE-VINTAGE | SKIP | `evidence/data_timing/results.json` | 관측 기준월·공개일·실시간 빈티지 일치 | F24 |
| DATA-WRDS-YAHOO | SKIP | `evidence/data_timing/results.json` | WRDS 논문 가격과 Yahoo 구현 자료 대조 | F24 |
| DATA-SYNTHETIC-LINEAGE | PASS | `evidence/data_timing/results.json` | 실제·합성 ETF 입력 계보 표시 | F16 |
| REG-001 | PASS | `evidence/regimes/results.json` | Independent equations (1), (4), (5), (7) hand self-check | F20 |
| REG-002 | PASS | `evidence/regimes/results.json` | Equation (1): finite nonnegative normalized probabilities on valid distances | F20 |
| REG-003 | PASS | `evidence/regimes/results.json` | L2 distance is square-rooted before probability conversion | F20 |
| REG-004 | PASS | `evidence/regimes/results.json` | Equations (2)-(4), (6): ordinary p0 and equality at 0.5 | F20 |
| REG-005 | PASS | `evidence/regimes/results.json` | p0 endpoints: epsilon clipping differs from the exact limit | F20 |
| REG-006 | PASS | `evidence/regimes/results.json` | Identical centers and insufficient normal clusters: unspecified Eq (1) boundaries | F20 |
| REG-007 | PASS | `evidence/regimes/results.json` | Zero-vector cosine and hard-label ties are deterministic conventions | F20 |
| REG-008 | PASS | `evidence/regimes/results.json` | Algorithm 1 equal-sized first-stage groups and normal-count padding | F20 |
| REG-009 | FAIL | `evidence/regimes/results.json` | Section3 k-means++ initialization on identical vectors | F18 |
| REG-010 | PASS | `evidence/regimes/results.json` | Eq (5): counts, last-month denominator and empty row in both source paths | F12, F19 |
| REG-011 | PASS | `evidence/regimes/results.json` | Eq (7): row-origin/column-destination direction and probability conservation | F19 |
| REG-012 | FAIL | `evidence/regimes/results.json` | Section3 absent intermediate regime creates nonfinite transition row | F18 |
| REG-013 | FAIL | `evidence/regimes/results.json` | PCA sign_flip: identical physical centroids retain identity only after basis alignment | F08 |
| REG-014 | FAIL | `evidence/regimes/results.json` | PCA quarter_turn: identical physical centroids retain identity only after basis alignment | F08 |
| REG-015 | FAIL | `evidence/regimes/results.json` | Changed PCA dimension: truncation does not establish a common coordinate system | F08 |
| REG-016 | PASS | `evidence/regimes/results.json` | Dimension reduction may also destroy identity information | F08 |
| REG-017 | PASS | `evidence/regimes/results.json` | Eq (14) simultaneous permutation invariance and one-sided permutation counterexample | F01, F08 |
| REG-018 | FAIL | `evidence/regimes/results.json` | Actual identical FRED 48-row inputs: Step1/2 regime identifiers and partitions | F01, F09 |
| REG-019 | FAIL | `evidence/regimes/results.json` | Original Ridge and Step3 aggregation: consequence of unmapped regime identifiers | F01, F09 |
| REG-020 | PASS | `evidence/regimes/results.json` | Actual bounded windows: returned cluster labels agree with returned centers | F01 |
| REG-021 | PASS | `evidence/regimes/results.json` | Calendar clipping and independent matching histories have different first anchors | F01 |
| REG-022 | FAIL | `evidence/regimes/results.json` | Actual source connection does not carry partition identity across Step1→Step2→Step3 | F01 |
| FOR-001 | PASS | `evidence/forecasting/results.json` | 독립 정규방정식·LOOCV·식14·식18 손계산 자체검사 | F03 |
| FOR-002 | PASS | `evidence/forecasting/results.json` | Step2 고정 λ 계수·복원 절편·예측은 독립 정규방정식과 일치 | F03 |
| FOR-003 | PASS | `evidence/forecasting/results.json` | Step2 부분집합 중심화와 절편 복원은 평행이동에 일관적 | F03 |
| FOR-004 | FAIL | `evidence/forecasting/results.json` | Step2 analytic LOOCV는 절편 포함 한 행씩 재적합과 불일치 | F03 |
| FOR-005 | FAIL | `evidence/forecasting/results.json` | 절편 누락은 선택 λ와 예측을 바꾸는 합성 반례를 만든다 | F03 |
| FOR-006 | PASS | `evidence/forecasting/results.json` | 원 레짐 표본 0·5개는 전체 47행 fallback, 6개부터 조건부 학습 | F09 |
| FOR-007 | PASS | `evidence/forecasting/results.json` | 원 Step2 학습 목표 상한은 R_t이며 R_(t+1) 이후 변경에는 불변 | F24 |
| FOR-008 | PASS | `evidence/forecasting/results.json` | Step3 fallback RidgeCV는 명시적 음의 MSE scoring과 절편 LOO를 사용 | F03 |
| FOR-009 | PASS | `evidence/forecasting/results.json` | 정상 Step3 집계는 인쇄 식14의 R1~R5 합이며 R0 예측을 제외 | F10, F21 |
| FOR-010 | PASS | `evidence/forecasting/results.json` | 일반 레짐 확률 재정규화의 양의 공통 배율은 lo 비중에서 상쇄 | F21 |
| FOR-011 | FAIL | `evidence/forecasting/results.json` | R0 확률 1인 경계에서 Step3 균등 fallback은 인쇄 식14의 0 예측과 다름 | F21 |
| FOR-012 | FAIL | `evidence/forecasting/results.json` | Step3 inner merge는 한쪽 날짜 누락을 조용히 삭제 | F05 |
| FOR-013 | FAIL | `evidence/forecasting/results.json` | Step3 중복 날짜 merge는 3×2 카테시안 행 증식을 허용 | F05 |
| FOR-014 | PASS | `evidence/forecasting/results.json` | 누락 레짐 예측은 해당 자산 NaN으로 전달되며 확률 0이어도 유지 | F05 |
| FOR-015 | PASS | `evidence/forecasting/results.json` | 원 lo 함수: 양수 상위 l·양수 부족·전부 비양수·동률·노출 검사 | F10, F21, F23 |
| FOR-016 | PASS | `evidence/forecasting/results.json` | lo NaN/Inf 경계: NaN·음의 Inf 제외, 양의 Inf 포함 시 전체 현금 | F23 |
| MET-001 | PASS | `evidence/metrics_integration/results.json` | Independent scalar metrics, wealth, portfolio and sizing hand controls | F06 |
| MET-002 | PASS | `evidence/metrics_integration/results.json` | Original metrics agree with independently calculated implemented conventions across six cases | F06, F23 |
| MET-003 | PASS | `evidence/metrics_integration/results.json` | Original drawdown routine fills NaN with zero; metric wrapper drops NaN before evaluation | F06, F23 |
| MET-004 | FAIL | `evidence/metrics_integration/results.json` | A first-month loss is included when peak starts at initial wealth 1 | F06 |
| MET-005 | PASS | `evidence/metrics_integration/results.json` | Sortino denominator is negative-subset sample SD; reference uses downside RMS over all months | F23 |
| MET-006 | PASS | `evidence/metrics_integration/results.json` | Stored drawdowns and positive-return proportion use fractions, despite percent column title | F23 |
| MET-007 | PASS | `evidence/metrics_integration/results.json` | Actual Step5 loop multiplies weights by sasdate_tp1 returns; SPY and EW hand controls | F05 |
| MET-008 | FAIL | `evidence/metrics_integration/results.json` | Missing holding-month return date is surfaced rather than silently dropped | F05 |
| MET-009 | FAIL | `evidence/metrics_integration/results.json` | An actively held missing return is not reported as zero portfolio return | F05 |
| MET-010 | PASS | `evidence/metrics_integration/results.json` | EW mean skips a missing ETF and changes denominator from ten to nine | F05 |
| MET-011 | FAIL | `evidence/metrics_integration/results.json` | Visualization accepts the filenames produced by Step5 directly | F14 |
| MET-012 | FAIL | `evidence/metrics_integration/results.json` | After filename-only fixture aliases, visualization accepts Step5 metric schema | F14 |
| MET-013 | FAIL | `evidence/metrics_integration/results.json` | Missing annual return and volatility columns remain after diagnostic Strategy mapping | F14 |
| MET-014 | FAIL | `evidence/metrics_integration/results.json` | Step5 implements the commented 10 percent volatility target and cumulative log-return figures | F13 |
| MET-015 | FAIL | `evidence/metrics_integration/results.json` | Existing Step2 with missing Step1 can auto-generate probabilities | F15 |
| MET-016 | FAIL | `evidence/metrics_integration/results.json` | Missing ETF auto-generation leaves explicit synthetic provenance in downstream CSV artifacts | F16 |
| MET-017 | PASS | `evidence/metrics_integration/results.json` | One real-FRED window with declared synthetic ETF data traverses original Step1→2→3→4→5 files/dates/numbers | F01, F24 |
| MET-018 | SKIP | `evidence/metrics_integration/results.json` | Actual WRDS/ETF empirical performance replication | F24 |

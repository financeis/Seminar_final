# 논문 구현 진단 보고서

## 결론

현재 코드는 논문의 **Ridge + long-only 일부 방법을 구현**했지만, 논문 전체를 정확하게 재현했다고 판단할 수 없다. 핵심 수식의 정상 입력 계산은 여러 곳에서 일치한다. 그러나 단계 사이의 레짐 정의가 서로 다르고, 미래 자료를 이용한 변수 선별과 Ridge 교차검증 산식 오류가 확인됐다. 성과 계산·파일 연결에도 독립적으로 재현한 문제가 있다.

검토 기준은 사용자 지정 `idea_paper.pdf` 26쪽과 기준 커밋 `9439107b31f20c4b731ec1dc29f13c178beb311d`다. 원 연구 코드와 입력 파일은 보존했다. 이번 결과는 진단 보고서·검증 실험·수정안이며, 실제 소스 수정 및 논문의 전체 성과표 재현은 포함하지 않는다.

## 검증 결과

| 범위 | PASS | FAIL | ERROR | SKIP |
|---|---:|---:|---:|---:|
| data_timing | 13 | 7 | 0 | 2 |
| regimes | 14 | 8 | 0 | 0 |
| forecasting | 11 | 5 | 0 | 0 |
| metrics_integration | 8 | 9 | 0 | 1 |
| TOTAL | 46 | 29 | 0 | 3 |

PASS는 해당 실험의 명시된 성질이 성립했다는 뜻이다. 관례를 정확히 관측한 PASS도 포함한다. FAIL은 원문·수학적 조건·파일 계약과 다른 동작이 확인된 경우다. 경계 입력에서만 확인한 FAIL을 실제 역사 자료의 발생 빈도로 해석하지 않는다. ERROR는 진단 도구 실행 실패, SKIP은 필요한 실증 자료가 없는 항목이다. 정상 진단의 종료 코드는 FAIL이 있어도 0이다.

전체 수식·알고리즘·방법 대조는 `methodology.md`, `traceability.md`에, 실험별 입력·기대값·관측값·독립 기준·허용오차는 `evidence/*/results.json`에 있다. `evidence/run_summary.json`이 최신 통합 실행의 명령·환경·종료 코드·원본 전후 해시의 기준 기록이다.




## 확인된 정상 구현

- 정상 거리에서 식1, L2 거리의 제곱근, 식2–4·6의 log2 매핑과 합 정규화, Step1의 식7 p@E는 독립 손계산과 일치했다.
- 주 전처리 5개 경로와 t-code1–7은 일치했다. 보조 통계 함수의 결측 t-code7만 별도 차이가 있었다. 121개 CSV t-code도 제공 부록과 모두 일치한다.
- 고정 λ Ridge의 계수·절편·예측과 중심화 복원은 맞는다. Step3 fallback RidgeCV는 명시적 음의 MSE를 사용한다.
- 일반 입력의 식14 가중합과 식18의 양수 상위 l개 비중은 맞는다. 일반 확률을 양의 공통 배율로 조정해도 lo 비중은 같다.
- 실제 FRED 첫 48행과 명시한 합성 ETF(seed20260916)로 원 Step1/2 첫 반복 → 원 Step3/4/5 전체를 연결했다. 1996-02 결정, 1996-03 수익 라벨 1행과 지표5행이 전달됐다. lo2/3/4 수익은 각각 0.0031753711 / 0.0065052052 / 0.0056231149다. 이 PASS는 파일·날짜·수치 전달을 뜻하며 F01의 공유 레짐 불일치를 해결하지 않는다.

## 발견 목록

아래 25개 항목은 29개 FAIL을 원인별로 묶고, 확인된 정책·미구현·미확정 사항을 추가한 목록이다. severity는 수정 우선도, confidence는 관측 근거의 수준이다. confirmed도 합성 경계 반례인지 실제 입력 반례인지 각 한계를 함께 읽어야 한다. 구현 오류와 원문 명세 위반, 원문 모호함은 category로 구분한다. 원 JSON과 표의 ID·제목·심각도·확신도가 같다.

| ID | 제목 | category | severity | confidence |
|---|---|---|---|---|
| F01 | Step1 확률과 Step2 예측이 같은 레짐을 가리키지 않는다 | implementation_error | high | confirmed |
| F02 | 전기간 결측률로 변수를 골라 미래 자료가 과거 결정에 영향을 준다 | statistical_validity | high | confirmed |
| F03 | Ridge analytic LOOCV에서 절편의 hat diagonal 1/n이 빠졌다 | implementation_error | high | confirmed |
| F04 | 행 삭제 후 다음 행과 48행을 다음 달·48개월로 취급한다 | paper_deviation | high | confirmed |
| F05 | 날짜·수익 결측과 중복이 집계·백테스트에서 조용히 통과한다 | implementation_error | high | confirmed |
| F06 | 첫 달 손실이 낙폭 계산에서 사라진다 | implementation_error | high | confirmed |
| F07 | 입력 변수·기간이 논문의 데이터 설정과 다르다 | paper_deviation | medium | confirmed |
| F08 | 서로 다른 PCA 좌표의 중심을 직접 비교해 레짐을 매칭한다 | statistical_validity | high | confirmed |
| F09 | 소표본 레짐 회귀가 전체 창 회귀로 바뀌어도 유효 표본수가 저장되지 않는다 | paper_deviation | medium | confirmed |
| F10 | Naive·BL·MVO 및 lns·los·mx 전략이 구현되지 않았다 | missing_method | high | confirmed |
| F11 | 무작위 레짐 대조와 유의성 검정이 구현되지 않았다 | missing_method | medium | confirmed |
| F12 | GMM·NBER 비교와 조건부 전이 네트워크가 구현되지 않았다 | missing_method | medium | confirmed |
| F13 | 10% 변동성 조정과 누적 로그 수익 그림이 없다 | missing_method | medium | confirmed |
| F14 | Step5 출력과 백테스트 시각화의 파일·열 계약이 맞지 않는다 | implementation_error | medium | confirmed |
| F15 | Step2만 남은 상태에서 Step3 자동 복구가 실패한다 | implementation_error | medium | confirmed |
| F16 | ETF 자동 합성 파일에 출처와 생성 계보가 남지 않는다 | reproducibility | high | confirmed |
| F17 | complete-case 전처리가 무한대를 제거하지 못한다 | implementation_error | medium | confirmed |
| F18 | Section3의 퇴화 군집 입력에 명시적 처리가 없다 | implementation_error | medium | confirmed |
| F19 | 전이행렬의 분모에 원문과 코드 경로 간 모호함이 있다 | paper_ambiguity | medium | confirmed |
| F20 | 확률 변환의 끝점·영거리 정책이 원식의 극한과 다르다 | paper_deviation | low | confirmed |
| F21 | R0 확률 1에서 일반 레짐을 균등하게 되살린다 | paper_deviation | medium | confirmed |
| F22 | 통계 보조 스크립트의 t-code7만 결측을 채워 계산한다 | implementation_error | low | confirmed |
| F23 | 성과지표·결측·비유한 비중 정책은 논문과 동일한 정의인지 미정이다 | paper_ambiguity | medium | supported |
| F24 | 실제 ETF·공개일·실시간 빈티지가 없어 실증 재현을 확인할 수 없다 | reproducibility | high | unresolved |
| F25 | 원문 자체의 시점·BL 명세·표본 수에는 해결되지 않은 모호함이 있다 | paper_ambiguity | info | unresolved |

### F01 — Step1 확률과 Step2 예측이 같은 레짐을 가리키지 않는다

**기대:** 확률 p_i와 조건부 예측 yhat_i는 같은 창·같은 레짐 분할·같은 식별자를 사용해야 한다.

**관측:** 실제 FRED 1992-03~1996-02의 같은 48×121 입력에서 두 단계의 48개 레이블 중 27개가 다르다. 일반 레짐 번호를 최적으로 맞춰도 10개가 다르고, 이상치 소속도 2개가 다르다. 별도 seed/초기화/군집과 별도 ETF 달력·매칭 이력을 사용하면서 Step3는 날짜와 Ri 열 이름만으로 곱한다.

**영향:** 확률과 조건부 예측의 의미가 어긋나므로 식14의 숫자 연산이 맞아도 의도한 조건부 예측이 아니다. 원 Ridge와 합성 목표(seed714)를 연결한 첫 창에서는 매핑 전 0.0004440491, 진단용 매핑 후 0.0021490163으로 달라졌다.

**근거:** REG-017, REG-018, REG-019, REG-020, REG-021, REG-022, MET-017. 코드: `Section5_step1.py:182–246` (<module>); `Section5_step2.py:234–286` (<module>); `Section5_step3.py:329–341` (<module>). 원문: p.15 5.2.3 식14.

**수정안·후속 검사:** 우선순위 1. Step1/2에서 창·전처리·PCA·군집 레이블을 한 번 생성하는 공통 결과물을 만들고 partition_id와 입력 해시를 두 출력에 보존한다. Step3에서 일치와 날짜 유일성을 강제한다. 동일 창의 소속 완전 일치 및 양측 동시 순열 불변성을 재검사한다. 모든 예측·비중을 재생성해야 한다.

**판정 한계:** 3개 제한된 실제 거시경제 창을 검사했다. 민감도 수익은 합성이며 역사적 성과 편향의 크기가 아니다. 최적 레이블 매핑도 다른 분할을 복구하지 못한다. REG-018 내부 학습 집계의 일부 날짜는 합성 위치 진단용이며 실제 창 날짜는 바깥 dates를 사용한다.

### F02 — 전기간 결측률로 변수를 골라 미래 자료가 과거 결정에 영향을 준다

**기대:** 이전 시점의 입력·결정은 그 이후 원자료만 바꾸어도 변하지 않아야 한다.

**관측:** 2015-01~2016-12의 RPI 24셀만 NaN으로 바꾸고 1996-02 이전 원자료는 그대로 두었다. 전기간 내부 결측률 2% 선별로 변수 수가 121→120이 되고 최초 과거 p_next 최대 차이 4.978347e-5가 생겼다. 보호변수 FEDFUNDS를 바꾼 대조는 차이 0이다. 창내 PCA/표준화 자체는 48행에 적합한다.

**영향:** 백테스트 정보 집합에 미래 결측 여부가 들어간다. 창내 모델 적합만으로 이 선별 단계의 미래 의존성이 없어지지 않는다.

**근거:** DATA-FUTURE-SELECTION, DATA-FUTURE-PROTECTED, DATA-FIT-SCOPE, DATA-QUALITY-ORDER. 코드: `Section3.py:10–95` (prepare_data_if_needed); `Section5_step1.py:22–68` (<module>); `Section5_step2.py:29–69` (<module>); `Section5_step3.py:60–102` (load_step2_or_autogen); `Section5_step4.py:39–75` (<module>). 원문: p.16 6.

**수정안·후속 검사:** 우선순위 1. 선별을 각 의사결정 이전 정보로 제한하거나 사전에 고정한 변수 목록을 사용한다. 모든 전처리 경로에 같은 정책을 적용하고 미래 원자료 교란에 대한 과거 결정 불변성을 다시 확인한다.

**판정 한계:** 정확한 역사 수익률 왜곡 방향·크기는 계산하지 않았다. 논문은 결측 선별 정책을 상세히 정하지 않으므로 인과적 타당성 기준의 진단이다.

### F03 — Ridge analytic LOOCV에서 절편의 hat diagonal 1/n이 빠졌다

**기대:** 절편을 복원하는 Ridge의 한 행 제외 검증은 H=11ᵀ/n+Xc(XcᵀXc+λI)⁻¹Xcᵀ의 대각으로 계산하거나 실제로 한 행씩 재적합해야 한다.

**관측:** 원 코드는 중심화한 X의 hat diagonal만 사용한다. 독립적인 절편 비벌점 정규방정식으로 각 행을 제외해 재적합하면 n=8 반례에서 λ=13.8949549437인데 원 코드는 1.9306977289를 고른다. 예측도 2.8806423114 대 4.1849911948로 다르다. 1/n 보정을 더한 analytic 점수는 독립 재적합과 일치한다.

**영향:** λ 선택과 그 뒤 예측·비중이 달라질 수 있다. 고정 λ의 계수·절편·예측은 이미 정상이며 오류 범위는 analytic 검증 점수다.

**근거:** FOR-001, FOR-002, FOR-003, FOR-004, FOR-005, FOR-008. 코드: `Section5_step2.py:163–176` (ridge_precompute_B_Hdiag); `Section5_step2.py:178–191` (ridge_select_lambda_and_predict); `Section5_step2.py:258–286` (<module>). 원문: p.14 5.2.3 식13.

**수정안·후속 검사:** 우선순위 2. Step2의 hdiag에 절편 항 1/n을 반영하거나 절편 포함 명시적 LOO로 통일한다. 동일 후보별 MSE·선택 λ·예측을 독립 재적합과 비교한다. Step3/4 fallback의 후보 grid도 명시하고 동등 설정 비교를 수행한다.

**판정 한계:** 반례는 seed4의 작은 합성 회귀이고 수치가 실제 ETF 수익률은 아니다. 논문은 λ 선택법을 지정하지 않는다. Step3 fallback은 neg_mean_squared_error를 명시하므로 기본 R² scoring 오류가 없다. LOO를 썼다는 사실 자체를 미래 OOS 자료 누출로 판정하지 않는다.

### F04 — 행 삭제 후 다음 행과 48행을 다음 달·48개월로 취급한다

**기대:** 월별 이동 실험은 48개의 연속 기준월과 다음 달 목표를 유지하고 누락 월을 명시해야 한다. 가격 자료도 월간 구간 경계를 검증해야 한다.

**관측:** 전처리 후 389개월(1992-03~2024-09)이 남는다. raw CSV의 2020-04 CP3Mx·COMPAPFFx 결측 및 CP3Mx 차분의 전파로 2020-05도 삭제된다. 341개 창 중 47개는 48행이 50개월에 걸친다. 2020-03의 다음 retained label은 2020-06이다. 별도 합성 가격 달력에서 2월이 없으면 로더는 1월→3월 수익을 1월 월간 수익으로 표시한다.

**영향:** 학습 기간과 예측 달력의 의미가 원문의 매월 이동과 달라진다. README의 4월 한 달 제외 설명보다 삭제 영향이 길다.

**근거:** DATA-CALENDAR, DATA-README, DATA-BOM, DATA-BOM-GAP. 코드: `Section5_step1.py:59–68` (<module>); `Section5_step1.py:175–180` (<module>); `Section5_step2.py:60–77` (<module>); `Section5_step2.py:225–230` (<module>); `Section5_step0_ETF_Loader.py:36–67` (compute_bom_returns_from_daily); `README.md:114–116` (문서). 원문: p.16 6.

**수정안·후속 검사:** 우선순위 1. 월간 PeriodIndex를 유지하고 누락 월 처리 방식을 먼저 결정한다. 48개월 연속성·target=t+1개월을 검사하며 다개월 가격 구간은 월간 수익으로 저장하지 않는다. ETF 교집합 이후에도 같은 검사를 수행한다.

**판정 한계:** March→June은 거시경제 다음 행의 라벨 문제다. 이것을 곧바로 3개월 보유수익으로 해석하면 안 된다. 실제 R_June은 로더가 정의한 June 월초 가격 구간이다. April raw 결측을 누가 언제 만들었는지는 제공 파일만으로 모른다.

### F05 — 날짜·수익 결측과 중복이 집계·백테스트에서 조용히 통과한다

**기대:** 의사결정 키는 유일해야 하고 양쪽 날짜 불일치나 보유 자산 수익 결측은 명시적으로 실패·보류되어야 한다.

**관측:** Step3 inner merge는 3개 예측 날짜를 2개로 줄이고 중복 3×2 키를 6행으로 증식시킨다(전체 8행). Step5는 보유월 날짜 누락 시 4결정→3행으로 조용히 건너뛴다. 비중 1인 SPY 수익이 NaN일 때 np.nansum이 포트폴리오 수익 0을 만든다. EW는 결측 ETF를 제외해 0.1/9를 계산한다.

**영향:** 평가 표본·종목 집합이 경로마다 달라지고 미관측 손익이 0으로 보일 수 있다. 누락·중복 자료가 있는 경우 성과 비교의 전제가 무너진다.

**근거:** FOR-012, FOR-013, FOR-014, MET-007, MET-008, MET-009, MET-010. 코드: `Section5_step3.py:329–341` (<module>); `Section5_step5.py:33–49` (<module>). 원문: p.16 6.

**수정안·후속 검사:** 우선순위 1. 날짜 키 uniqueness, merge validate=one_to_one, outer 비교 후 불일치 보고를 추가한다. 비중≠0인 자산의 수익 유한성을 검사하고 평가용 고정 달력·basket 정책을 통일한다. 누락 날짜·중복·활성 NaN 반례가 명시적 진단을 내는지 확인한다.

**판정 한계:** 작은 입력 반례로 확인한 동작이며 실제 ETF 자료의 누락 빈도는 확인하지 못했다. EW의 skipna 자체는 원문이 정하지 않은 정책이지만 고정 10종목 비교와 구분해야 한다.

### F06 — 첫 달 손실이 낙폭 계산에서 사라진다

**기대:** 투자 전 초기 자산 1을 최고점 후보로 넣어 첫 달부터의 손실을 계산해야 한다.

**관측:** 수익 [-0.1,0.05]의 자산은 [0.9,0.945]다. 초기 1을 포함하면 MaxDD=-0.1, 음수 기간 평균 AvgDD=-0.0775다. 원 Step5는 첫 수익 후의 0.9부터 cummax를 잡아 둘 다 0을 반환한다. 시각화도 누적곱 이후부터 고점을 잡는다.

**영향:** 처음 손실이 나고 기존 초기 자산을 회복하기 전 구간의 낙폭이 작게 보인다.

**근거:** MET-001, MET-002, MET-003, MET-004. 코드: `Section5_step5.py:51–54` (compute_drawdowns); `Section5_step5.py:56–67` (perf_metrics); `etc_visualize_backtest.py:52–65` (<module>). 원문: p.16 5.4.

**수정안·후속 검사:** 우선순위 2. 초기 wealth=1을 포함한 고점으로 원 함수와 시각화 계산을 통일한다. 첫 손실·첫 이익·무손실·한 번 손실의 독립 손계산과 비교한다.

**판정 한계:** 논문은 AvgDD 평균 범위를 자세히 정하지 않는다. 여기의 AvgDD=-0.0775는 원 코드와 같은 음수 기간 평균에 초기 고점만 고친 기준이다. 실제 표 전체의 영향은 미측정이다.

### F07 — 입력 변수·기간이 논문의 데이터 설정과 다르다

**기대:** 원문은 127개 FRED-MD 변수와 그룹6 제외, 1959-12~2023-01을 기술한다. 재현용 입력 명세를 해당 설정에 맞추거나 변경으로 명시해야 한다.

**관측:** 제공 CSV는 121변수이고 모든 변수가 유지된다. 부록 그룹6의 금리 17개도 모델 입력에 남는다. 1978-07~2024-09를 먼저 자른 뒤 변환·complete-case를 적용한다. 주 전략 5개 전처리는 서로 같은 389×121 값을 낸다. CSV의 121 t-code는 부록과 모두 일치한다(120개 exact, IPB51222s 대소문자 별칭 1개).

**영향:** 원문과 같은 샘플·변수·PCA·군집·수익률을 기대할 수 없다. 이는 재현 설정의 차이이며 각 변환 산식이 전부 잘못된 것은 아니다.

**근거:** DATA-GROUP6, DATA-PREP-3, DATA-PREP-5_step1, DATA-PREP-5_step2, DATA-PREP-5_step3, DATA-PREP-5_step4, PAPER-COVERAGE. 코드: `Section3.py:10–95` (prepare_data_if_needed); `Section5_step1.py:17–68` (<module>); `Section5_step2.py:23–69` (<module>); `Section5_step3.py:60–102` (load_step2_or_autogen); `Section5_step4.py:39–75` (<module>). 원문: p.7 4.1–4.2.

**수정안·후속 검사:** 우선순위 1. 논문 재현과 확장 실험의 변수 목록·기간을 별도 설정으로 고정하고 원자료/선별 목록 해시를 저장한다. 그룹6 제외와 기대 월 목록을 독립 검증한다. PCA는 누적 설명력95%로 선택하고 rolling에 61개를 강제하지 않는다.

**판정 한계:** 원문 61개 PCA는 특정 전표본의 결과다. 중심화한 48행의 rank≤47이므로 rolling의 29개 등을 단순 오류라고 할 수 없다. 논문의 금리 사후 그림이 모델 입력에 금리를 넣었다는 증거는 아니다.

### F08 — 서로 다른 PCA 좌표의 중심을 직접 비교해 레짐을 매칭한다

**기대:** 창 사이 동일한 물리적 중심을 비교하려면 공통 변수·표준화·좌표계가 필요하다. PCA 부호·회전·차원 수의 변화만으로 정체성이 바뀌어서는 안 된다.

**관측:** 동일한 중심을 PCA 부호만 뒤집은 좌표로 주면 원 매칭은 [1,0,2,3], 90도 회전이나 차원 절단에서는 [2,3,1,0]을 선택한다. 공통 좌표로 맞춘 독립 비교는 [0,1,2,3]이다. 두 행렬의 최소 열 수로 자르는 방식은 공통 공간을 만들지 않는다. Step4:138–149도 동일한 최소열절단 비교를 하며 fallback의 175/209에서 사용한다(이 경로는 정적 확인).

**영향:** 창 사이 번호의 경제적 의미를 유지한다는 전제가 깨진다. 단계 간 별도 분할 문제 F01과 함께 확인해야 한다.

**근거:** REG-013, REG-014, REG-015, REG-016, REG-017, INVENTORY. 코드: `Section5_step1.py:149–164` (match_centroids_cosine); `Section5_step2.py:145–160` (match_centroids_cosine); `Section5_step3.py:39–50` (match_centroids_cosine); `Section5_step4.py:138–149` (match_centroids_cosine). 원문: p.2 서론.

**수정안·후속 검사:** 우선순위 1. 공통 원 변수 공간에서 비교하거나 명시된 공통 변환으로 중심을 옮기고, 잃은 차원 정보·표준화 차이도 처리한다. 부호·회전·차원 변화 불변성과 실제 연속 창의 의미 일관성을 검사한다.

**판정 한계:** 논문은 정확한 매칭 알고리즘을 주지 않는다. 이 반례는 일반 좌표 타당성 검사다. 양측 확률·예측에 같은 순열을 적용하면 식14는 불변이므로 번호 변화만으로 포트폴리오 수익이 바뀐다고 단정하지 않는다.

### F09 — 소표본 레짐 회귀가 전체 창 회귀로 바뀌어도 유효 표본수가 저장되지 않는다

**기대:** 조건부 회귀의 실제 학습 집합과 fallback 여부를 명시해야 한다. 원문이 정하지 않은 표본 부족 정책은 추가 가정으로 기록해야 한다.

**관측:** 원 레짐 표본 [0,5,6,10,26]에서 실제 학습 행 수는 [47,47,6,10,26]이다. n_Ri는 fallback 전 개수다. 실제 FRED 1999-01~2002-12 진단 창은 일반 레짐 5개 중 4개가 전체 창 회귀로 대체되어 합성 목표에 대한 예측도 동일하다.

**영향:** 레짐별 모형이라는 해석과 실제 조건부 정보가 약해지고 출력만으로 사용한 학습 집합을 확인하기 어렵다.

**근거:** FOR-006, REG-018, REG-019. 코드: `Section5_step2.py:258–286` (<module>); `Section5_step3.py:171–199` (load_step2_or_autogen); `Section5_step4.py:162–194` (<module>). 원문: p.14 5.2.3 식13.

**수정안·후속 검사:** 우선순위 2. min_samples·대체모형 정책을 설정으로 선언하고 raw_count/effective_count/fallback/model_id를 저장한다. 전체 창·조건부 회귀 비교와 표본 0/5/6 경계 테스트를 수행한다.

**판정 한계:** 전체 창 fallback이 항상 나쁜 예측이라는 뜻은 아니다. 원문에 없는 선택이며 성과 우열을 검증하지 않았다.

### F10 — Naive·BL·MVO 및 lns·los·mx 전략이 구현되지 않았다

**기대:** 논문 전체 구현은 Naive/Ridge/BL, lo/lns/los/mx와 MVO 대조를 포함한다.

**관측:** 전체 원 연구 11개 파일(etc_regime_stats_transformed.py, etc_regime_visualization.py, etc_visualize_backtest.py, etc_visualize_transition_matrix.py, Section3.py, Section5_step0_ETF_Loader.py, Section5_step1.py, Section5_step2.py, Section5_step3.py, Section5_step4.py, Section5_step5.py)의 함수·최상위 실행·입출력 및 README를 조사했다. Ridge와 lo l=2/3/4, SPY/EW 경로만 있다. Naive 조건부 Sharpe(식8–10), BL view/posterior(식11/19), lns/los(식16/17), mx 전환 및 MVO 최적화 함수·출력·실행 분기가 없다.

**영향:** 논문의 전략별 성과표와 모형 비교 전체를 재현할 수 없다.

**근거:** INVENTORY, PAPER-COVERAGE, FOR-009, FOR-015. 원문: p.14 5.2–5.3.

**수정안·후속 검사:** 우선순위 3. 공통 데이터·레짐·평가 계약을 먼저 고친 뒤 누락 모형과 비중 규칙을 추가한다. 각 수식의 독립 작은 예제, 순노출·총노출, mx 전환 조건 및 비교표의 전략 목록을 검사한다.

**판정 한계:** BL의 실제 P/Omega/tau·최종 비중화는 원문만으로 충분히 결정되지 않는다. 저자 명세 확인이나 명시적 가정이 필요하다.

### F11 — 무작위 레짐 대조와 유의성 검정이 구현되지 않았다

**기대:** 레짐 탐지의 추가 효과를 평가하려면 무작위 레짐 대조·paired t-test·Nemenyi 결과를 생성해야 한다.

**관측:** 전체 원 연구 11개 파일(etc_regime_stats_transformed.py, etc_regime_visualization.py, etc_visualize_backtest.py, etc_visualize_transition_matrix.py, Section3.py, Section5_step0_ETF_Loader.py, Section5_step1.py, Section5_step2.py, Section5_step3.py, Section5_step4.py, Section5_step5.py)의 함수·최상위 실행·입출력 및 README를 조사했다. 실제 실행·함수·출력에 대조군 반복 생성, paired t-test, Nemenyi 비교가 없다. 군집 초기화의 난수는 이 대조 실험에 해당하지 않는다.

**영향:** 전략 수익이 레짐 구조에서 나온다는 논문의 통계적 주장까지 검증되지 않는다.

**근거:** INVENTORY, PAPER-COVERAGE. 원문: p.17 6, Figures 7–9 / Table 3.

**수정안·후속 검사:** 우선순위 3. 동일 평가기간·모형·비용 조건을 고정한 대조 실험을 설계하고 seed·반복 수·짝짓기 단위·검정 가정을 저장한다. 샘플 순서·짝 불일치와 알려진 작은 검정 예제로 확인한다.

**판정 한계:** 원문은 반복·무작위화·짝짓기 세부가 부족하다. 이번 진단에서 논문의 p-value나 유의성을 재현하지 않았다.

### F12 — GMM·NBER 비교와 조건부 전이 네트워크가 구현되지 않았다

**기대:** 레짐 해석은 GMM/NBER 대조와, 자기 전이를 제외한 조건부 전이 e_ij/(1-e_ii), 대각0의 네트워크를 포함한다.

**관측:** 전체 원 연구 11개 파일(etc_regime_stats_transformed.py, etc_regime_visualization.py, etc_visualize_backtest.py, etc_visualize_transition_matrix.py, Section3.py, Section5_step0_ETF_Loader.py, Section5_step1.py, Section5_step2.py, Section5_step3.py, Section5_step4.py, Section5_step5.py)의 함수·최상위 실행·입출력 및 README를 조사했다. 지표 평균·row min-max·레짐 산점도·전이 히트맵·자기 전이 막대는 있다. etc_regime_visualization.py:41의 NBER 주석과 etc_visualize_transition_matrix.py:45의 network-style 주석 뒤 실제 실행에는 NBER 자료·음영 비교나 조건부 off-diagonal 정규화/네트워크가 없다. GMM 적합 경로도 없다.

**영향:** 논문의 레짐 경제적 해석과 전이 경로 비교가 일부만 구현되어 있다.

**근거:** INVENTORY, PAPER-COVERAGE, REG-010. 원문: p.10 4.3–4.6.

**수정안·후속 검사:** 우선순위 3. 해당 보조 분석을 추가하고 NBER 구간 출처, GMM 설정, 조건부 전이 분모·흡수상태 정책을 명시한다. 대각0/조건부 행합1 및 e_ii=1 경계를 손계산과 비교한다.

**판정 한계:** 현재 히트맵과 지속성 막대 자체가 잘못됐다는 진단은 아니다. 논문에서 요구하는 추가 분석이 없다.

### F13 — 10% 변동성 조정과 누적 로그 수익 그림이 없다

**기대:** 원문 그림은 10% 변동성으로 조정한 누적 로그 수익을 사용한다.

**관측:** 전체 원 연구 11개 파일(etc_regime_stats_transformed.py, etc_regime_visualization.py, etc_visualize_backtest.py, etc_visualize_transition_matrix.py, Section3.py, Section5_step0_ETF_Loader.py, Section5_step1.py, Section5_step2.py, Section5_step3.py, Section5_step4.py, Section5_step5.py)의 함수·최상위 실행·입출력 및 README를 조사했다. Step5:77의 optional 주석 외 실행되는 변동성 조정은 없다. etc_visualize_backtest.py:41–44는 원 수익의 (1+r).cumprod()를 그린다. 목표 변동성 추정·lag·레버리지 경로 및 누적 로그 계산이 없다.

**영향:** 현재 그림을 원문 그림의 위험 조정 비교로 볼 수 없다.

**근거:** MET-014, INVENTORY, PAPER-COVERAGE. 원문: p.21 Figures 10–13.

**수정안·후속 검사:** 우선순위 3. 사전 정보만으로 추정하는 목표 변동성 정책·상한을 선언하고 조정 수익과 log wealth를 별도 산출한다. 일정 변동성 예제·lag 교란·0변동성 경계를 검증한다.

**판정 한계:** 원문은 추정 창·lag·상한·표의 조정 여부를 명확히 정하지 않아 임의 정책의 결과를 원문과 동일하다고 부르면 안 된다.

### F14 — Step5 출력과 백테스트 시각화의 파일·열 계약이 맞지 않는다

**기대:** 시각화는 앞 단계가 실제로 저장한 파일명·모델명·열을 그대로 읽을 수 있어야 한다.

**관측:** 원 Step5 출력만 제공하면 다른 파일명을 읽어 FileNotFoundError가 난다. 임시 파일명만 맞추면 Model 대신 Strategy를 요구해 KeyError, 진단용 이름 매핑 후에도 Ann.Return와 Ann.Vol 누락 오류가 순서대로 발생했다.

**영향:** README의 순차 실행 지침으로 최종 성과 그림을 완성할 수 없다.

**근거:** MET-011, MET-012, MET-013. 코드: `Section5_step5.py:19–20` (<module>); `Section5_step5.py:65–75` (<module>); `etc_visualize_backtest.py:10–15` (<module>); `etc_visualize_backtest.py:108–114` (<module>). 원문: p.16 5.4 / Figures 10–13.

**수정안·후속 검사:** 우선순위 2. 출력 스키마를 한 곳에 정의하고 시각화의 입력명·모델명·필수 지표를 통일한다. Step5가 쓴 원 파일을 그대로 다음 스크립트에 주는 작은 연결 검사를 수행한다.

**판정 한계:** Ann.Return=0 임시 열은 다음 열 접근을 드러내는 진단 fixture였으며 추정 성과가 아니다. 그림의 전체 디자인 재현은 검사하지 않았다.

### F15 — Step2만 남은 상태에서 Step3 자동 복구가 실패한다

**기대:** Step1 파일을 재생성하려면 필요한 macro context를 확보하거나 명시적인 누락 오류를 반환해야 한다.

**관측:** Step2가 있으면 loader는 macro_ctx=None을 반환한다. Step1이 없으면 다음 loader가 이를 unpack하면서 TypeError: cannot unpack non-iterable NoneType object를 낸다.

**영향:** 중간 파일 일부만 존재하는 일반적인 재실행 상태에서 집계가 중단된다.

**근거:** MET-015. 코드: `Section5_step3.py:53–205` (load_step2_or_autogen); `Section5_step3.py:207–320` (load_step1_or_autogen). 원문: p.15 5.2.3 식14.

**수정안·후속 검사:** 우선순위 2. macro context 로딩과 결과 파일 로딩을 분리한다. Step1/Step2 존재의 네 조합에서 생성·재사용·명시적 실패 정책을 확인하고 실패 시 빈 결과를 성공으로 기록하지 않는다.

**판정 한계:** 전체 10창 fallback을 돌린 실증 검사가 아니라 실제 로더 분기를 분리 실행한 재현이다.

### F16 — ETF 자동 합성 파일에 출처와 생성 계보가 남지 않는다

**기대:** 실제 가격 기반 결과와 합성 진단은 입력 파일에서 출처·생성 seed·해시로 구분되어야 한다.

**관측:** ETF 파일 부재 분기에서 원 Step3/4 prefix를 실행하면 seed123의 합성 389행이 실제 loader와 같은 etf_bom_returns_aligned_demo.csv 이름으로 저장된다. CSV와 sidecar에 합성 표시는 없다. Step0의 _make_synth_prices는 이름과 달리 Yahoo 실제 다운로드 함수에 위임한다.

**영향:** 기존 파일이 있으면 이후 단계는 이를 재사용하므로 파일명만으로 실제 데이터 실험인지 알 수 없다. 추적 NPZ도 원 입력 해시·생성 커밋·PCA 정체성 메타데이터가 부족하다.

**근거:** DATA-SYNTHETIC-LINEAGE, MET-016, INVENTORY. 코드: `Section5_step3.py:28–37` (synthesize_etf_bom_returns); `Section5_step3.py:104–110` (load_step2_or_autogen); `Section5_step4.py:14–22` (synthesize_etf_bom_returns); `Section5_step4.py:77–84` (<module>). 원문: p.16 6 / Table 2.

**수정안·후속 검사:** 우선순위 1. 기본 경로는 실자료 부재를 명확히 알리고 합성 실행을 명시적으로 선택하게 한다. source_kind·seed·입력 hash·생성 commit·partition_id를 별도 manifest에 저장하고 소비 단계에서 검증한다. 동일 입력의 정상/합성 재사용 분리를 검사한다.

**판정 한계:** 합성 생성 함수 자체의 확률 모형을 논문 오류라고 평가하지 않았다. 제공된 과거 NPZ가 어떤 실행에서 나왔는지는 확인할 수 없다.

### F17 — complete-case 전처리가 무한대를 제거하지 못한다

**기대:** 모델 입력의 수치 유효성은 NaN뿐 아니라 ±Inf도 검사해야 한다.

**관측:** t-code7에서 분모가 0인 합성 입력은 dropna 이후에도 무한대 2셀을 남긴다. 상수열 제거는 독립 예제와 일치하지만 비유한 입력 전체를 보장하지 않는다.

**영향:** 이런 입력에서는 표준화·SVD 이후 계산이 실패하거나 유효하지 않은 값으로 이어질 수 있다.

**근거:** DATA-INFINITY, DATA-CONSTANT. 코드: `Section5_step1.py:52–68` (<module>); `Section5_step1.py:71–75` (zscore_window). 원문: p.7 4.2.

**수정안·후속 검사:** 우선순위 2. 변환 이후와 각 모델 경계에서 isfinite를 검사하고 0분모·비양수 로그·결측 정책을 문서화한다. 0/NaN/Inf 입력에서 명확한 제외 사유나 오류를 검증한다.

**판정 한계:** 무한대 반례는 합성이며 제공 FRED 실제 창에서 발생했다고 주장하지 않는다. 동일 전처리 정책을 복제한 다른 단계도 수정 검토 대상이다.

### F18 — Section3의 퇴화 군집 입력에 명시적 처리가 없다

**기대:** 동일 벡터·빈 레짐 같은 퇴화 입력에서 군집 수 축소 또는 선언된 오류/전이 정책을 사용해야 한다.

**관측:** 모든 벡터가 같으면 초기화 확률 분모가 0이 되어 ValueError: probabilities contain NaN이 난다. 레이블 [0,2,0]처럼 중간 레짐1이 없으면 Section3 전이행렬의 그 행은 NaN이다.

**영향:** 입력 다양성이나 레짐 수가 줄어든 경우 전표본 분석 실행과 확률 행렬의 유효성이 깨진다.

**근거:** REG-009, REG-012. 코드: `Section3.py:107–117` (kmeanspp_init_l2); `Section3.py:141–151` (kmeanspp_init_cosine); `Section3.py:503–517` (<module>). 원문: p.5 3.1 / 3.3.

**수정안·후속 검사:** 우선순위 2. 고유 벡터 수·잔여 초기화 가중치 합을 확인하고 빈 레짐의 전이 의미를 정의한다. 동일 벡터·빠진 중간 레이블·단일 레짐 예제를 검증한다.

**판정 한계:** 두 FAIL은 경계 입력으로만 재현했다. 실제 FRED 전체 분석에서 해당 실패가 발생했다는 증거는 없다. 논문도 이 경계 정책은 정하지 않는다.

### F19 — 전이행렬의 분모에 원문과 코드 경로 간 모호함이 있다

**기대:** 전체 발생 횟수와 나가는 전이 횟수 중 무엇으로 나누는지 명시하고 확률 보존 요구와 함께 정의해야 한다.

**관측:** 원문 식5와 Section3는 전체 발생 횟수 N(i)를 분모로 사용한다. [0,1,2,0,0]에서 레짐0 행합은 2/3이다. Step1은 실제 출발 전이 수로 나누고 빈 행을 자기전이로 채워 행합1을 만든다. Step1 p@E 방향과 확률 보존은 독립 손계산과 일치한다.

**영향:** 두 출력 E는 같은 정의의 확률행렬이 아니다. 문자 그대로 원문을 따르면 마지막 관측 때문에 확률 질량이 줄어드는 문제가 생긴다.

**근거:** REG-010, REG-011. 코드: `Section3.py:503–517` (<module>); `Section5_step1.py:220–232` (<module>). 원문: p.6 3.3 식5.

**수정안·후속 검사:** 우선순위 2. 출력에 denominator_policy를 저장하고 count/occurrence/outgoing을 함께 기록한다. 원문 문자식과 확률 보존 보정식을 구별하고 마지막 관측·빈 행 예제를 검사한다.

**판정 한계:** Step1의 stochastic 보정은 수학적으로 타당한 선택이다. 이를 단순한 잘못된 정규화라고 판정하지 않는다. 논문 의도의 최종 선택은 미정이다.

### F20 — 확률 변환의 끝점·영거리 정책이 원식의 극한과 다르다

**기대:** 정상 거리에는 식1/4를 적용하고 p0=1 극한 및 0거리·K=1처럼 원식 미정인 경계는 별도 정책을 표시해야 한다.

**관측:** 정상 거리·L2 거리·log2 매핑·합 정규화는 일치한다. p0=1, 일반 확률 [.6,.4] 반례에서 클리핑 후 R0는 0.9598682095이고 극한은 1이다. 모든 거리=0이면 Step1은 [1,0,0], Section3는 균등분포를 낸다. K1·zero-vector·동률도 결정적 구현 정책이 있다.

**영향:** 이상치가 확실한 경계와 퇴화 거리에서 정책에 따라 확률이 달라진다.

**근거:** REG-001, REG-002, REG-003, REG-004, REG-005, REG-006, REG-007, REG-008. 코드: `Section3.py:271–279` (probs_from_distances); `Section5_step1.py:127–135` (fuzzy_probs_from_dist); `Section5_step1.py:211–217` (<module>). 원문: p.6 3.2 식4.

**수정안·후속 검사:** 우선순위 2. p0 끝점을 분기 처리할지 클리핑을 유지할지 문서화하고 정상 입력과 퇴화 입력의 정책을 통일한다. p0=0/.5/1과 영거리/K1을 검사한다.

**판정 한계:** 이 차이가 일반 실제 창에서 발생하는 빈도·성과는 평가하지 않았다. 영거리와 K1은 원문 식1 자체가 정의되지 않으므로 특정 fallback만 정답이라고 할 수 없다.

### F21 — R0 확률 1에서 일반 레짐을 균등하게 되살린다

**기대:** 인쇄 식14는 R1~Rr의 가중합이므로 일반 확률이 전부0이면 예측0이다.

**관측:** Step3에서 일반 확률 합이0이면 균등분포로 바꾼다. p_R0=1 반례의 예측은 원식 [0,0] 대신 [.018,.03]이다. 보통 양의 일반 확률에서는 식14와 일치하며 공통 양의 재정규화 배율은 lo에서 상쇄된다.

**영향:** 위기만 확실한 경계에서 일반 레짐 예측으로 포지션을 만들 수 있다.

**근거:** FOR-009, FOR-010, FOR-011, FOR-015. 코드: `Section5_step3.py:329–341` (<module>); `Section5_step4.py:241–253` (<module>). 원문: p.15 5.2.3 식14.

**수정안·후속 검사:** 우선순위 2. zero-normal-mass 정책을 현금/위기모형/명시적 오류 중 명세에 따라 정한다. R0=1 및 직전 경계의 예측·비중을 함께 검사한다.

**판정 한계:** 원문은 R0 제외와 위기 활용의 관계가 불명확하다. 일반 확률 재정규화 부재를 별도 lo 비중 오류로 세지 않는다. 정상 Step1의 클리핑 때문에 이 정확한 끝점의 자연 발생은 별도로 확인해야 한다.

### F22 — 통계 보조 스크립트의 t-code7만 결측을 채워 계산한다

**기대:** 같은 t-code7은 결측치가 있는 시점에도 주 전처리와 동일한 산식을 사용해야 한다.

**관측:** 6개 함수×7코드×4입력의 168비교에서 불일치는 etc_regime_stats_transformed.py의 t-code7 결측 입력에 한정된다. pct_change 기본 fill 동작으로 원 결측 뒤 변화율을 계산한다. 주 전략5개 전처리의 t-code1~7은 비교 기준과 일치한다.

**영향:** 일부 사후 지표 통계가 전략 입력의 변환값과 다르게 해석될 수 있다.

**근거:** DATA-SELF, DATA-TCODE, PAPER-COVERAGE. 코드: `etc_regime_stats_transformed.py:36–56` (transform_series). 원문: p.7 4.2.

**수정안·후속 검사:** 우선순위 2. pct_change(fill_method=None) 등 명시한 무보간 산식으로 통일하고 결측 앞뒤·0분모의 독립 변환 예제를 재검사한다.

**판정 한계:** 선택된 사후 지표가 이 경계를 실제 사용한 빈도나 그림 변화는 검사하지 않았다. 전체 전략의 변환 오류로 확대하지 않는다.

### F23 — 성과지표·결측·비유한 비중 정책은 논문과 동일한 정의인지 미정이다

**기대:** Sharpe의 rf/연율화, Sortino 분모, AvgDD 평균 범위, 단위, 결측·비유한 예측 처리 기준을 선언해야 한다.

**관측:** Sharpe는 rf0, sqrt12, ddof1이다. Sortino는 음수 월만의 표본 SD라 손실0/1회이면 NaN이다. 선언한 전체월 downside RMS 기준과 ordinary fixture가 0.408248 대 0.316228로 다르다. AvgDD는 음수 낙폭만 평균하고 % Positive Ret.도 저장값은 비율이다. lo는 양수 Inf 포함 시 모든 비중0이나 n_selected=2 메타데이터를 남긴다.

**영향:** 다른 정의로 계산한 논문/라이브러리 값과 직접 비교하면 해석이 어긋날 수 있다.

**근거:** MET-002, MET-003, MET-005, MET-006, FOR-015, FOR-016. 코드: `Section5_step5.py:56–67` (perf_metrics); `Section5_step5.py:51–54` (compute_drawdowns); `Section5_step4.py:256–270` (long_only_weights). 원문: p.16 5.4.

**수정안·후속 검사:** 우선순위 2. 지표 정의·단위·결측·비유한 예측 정책을 설정/메타데이터로 저장한다. 고정 작은 수익열에서 손계산과 비교하고 음수 월0/1/2개·상수·Inf 예제를 포함한다.

**판정 한계:** 논문이 분모와 경계 정책을 충분히 정하지 않아 Sortino 차이를 확정적 논문 수식 위반으로 세지 않았다. PASS는 해당 관례가 정확히 관측됐다는 뜻이다.

### F24 — 실제 ETF·공개일·실시간 빈티지가 없어 실증 재현을 확인할 수 없다

**기대:** 논문 WRDS 가격과 당시 이용 가능한 거시경제 빈티지, 의사결정·체결 시각이 있어야 실제 성과와 정보 시점을 검증할 수 있다.

**관측:** 실제 ETF CSV/WRDS 입력은 없다. 원 loader는 Yahoo 조정가격을 사용하고 기간도 다르다. FRED-MD_2024m12.csv에는 공개일·as-of 열이 없고 공식 빈티지와의 동일성도 미확인이다. 따라서 3개 검사는 SKIP이다. R_t=P(first trade t+1)/P(first trade t)-1의 목표 상한은 조건부로 정상이다.

**영향:** 논문 성과표·통계적 유의성 및 실시간 이용가능성의 최종 검증을 할 수 없다.

**근거:** DATA-RELEASE-VINTAGE, DATA-WRDS-YAHOO, MET-018, DATA-RETURN-CAUSALITY, FOR-007, MET-017, INVENTORY. 코드: `Section5_step0_ETF_Loader.py:98–139` (_fetch_real_prices); `Section5_step1.py:17–68` (<module>); `Section5_step2.py:17–19` (<module>). 원문: p.16 6 / Table 2.

**수정안·후속 검사:** 우선순위 1(명세), 3(실증). 원 WRDS/ETF 자료와 수정주가 정책·시각·공개/빈티지 자료를 확보하고 해시를 고정한다. 이후 시간 인과성·출처 일치·동일 표본 평가를 재검사한다.

**판정 한계:** 결정이 P_(t+1) 관측 이후라면 학습의 최신 R_t는 이미 알려진다. 목표 shift 자체를 누출로 판정하지 않는다. 동일 가격에서 즉시 체결 가능한지와 거시경제 자료 공개 지연은 미확정이다. 한 실제 FRED창+합성 ETF 연결 PASS는 실증 성과 재현이 아니다.

### F25 — 원문 자체의 시점·BL 명세·표본 수에는 해결되지 않은 모호함이 있다

**기대:** 완전 재현 명세는 학습 x/y·레짐 시점, BL 비중화 설정, 실제 월 목록·지표 정의를 확정해야 한다.

**관측:** 식12의 beta·X_(1:t) 표기는 단일 예측 차원/학습 시점과 불명확하다. 식19는 posterior expected return 형태지만 allocation이라 부르며 설정이 부족하다. p16의 2000~2022년 746개월 서술은 최대276개월과 양립하지 않는다. p20의 ridge_lo_2 최저 MaxDD 문장도 Table6 lo_3/lo_4 수치와 맞지 않는다. methodology의 AMB01–09에 원문 쟁점을 모두 기록했다.

**영향:** 임의 선택을 저자 의도의 정답으로 간주하면 완전 재현 판정이 과장된다.

**근거:** PAPER-COVERAGE, INVENTORY. 원문: p.14 5.2.3 식12; p.16 5.3–6 식19; p.20 Table 6 논의.

**수정안·후속 검사:** 우선순위 1. 저자 코드·데이터·정정으로 확인할 항목과 프로젝트의 명시적 가정을 구분해 고정한다. 원문 모호함이 남은 항목은 여러 합리적 정책의 민감도를 별도로 보고한다.

**판정 한계:** 명확한 표본 수 모순은 확인했지만 올바른 원 표본은 모른다. 본 감사는 2026년 출판본으로 기준 PDF를 대체하지 않는다. README는 이미 regime_t/t+1 선택의 불명확함을 인정했다.

## 수정 순서와 완료 조건

1. **연결·입력 명세:** F01/F02/F04/F05/F07/F08/F16/F24를 먼저 처리한다. 날짜·변수·자료 출처·공유 partition_id가 고정되어야 후속 회귀·비중의 뜻이 안정된다. 동일 창 소속 일치, 미래 입력 교란 불변성, 달력 연속성, 결측·중복 거부를 완료 조건으로 둔다.
2. **회귀·평가·실행:** F03/F06/F09/F14/F15/F17–F23을 처리한다. 독립 LOO와 후보별 점수, 초기 자산 포함 DD, fallback 유효 표본 수, CSV 생산자→소비자 연결을 확인한다. 정책 차이를 정한 뒤 모든 중간 결과를 새로 생성한다.
3. **논문 전체 범위:** F10–F13의 모형·전략·대조·그림을 추가하고 실제 자료가 확보되면 동일 표본의 전체 성과표를 재실행한다. p-value·BL·변동성 조정의 미정 설정은 가정으로 표시한다.

현재 연구 소스의 수정은 수행하지 않았다. 위 순서는 제안이며, 모든 FAIL을 감추기 위해 허용오차나 기대값을 바꾸는 방식은 완료 조건이 아니다.

## README가 이미 밝힌 한계와 이번에 확인한 내용

README는 1992년부터의 데이터, 2020-04 제외, 거래비용 미반영, 2011–2012 민감도, 실시간 거래 미지원, 식12의 regime_t/t+1 불명확함을 이미 설명한다. 이번에는 **실제 2020-05 추가 삭제·48행/달력 불일치**, 미래 결측 선별의 과거 영향, 같은 입력의 단계별 분할 불일치, analytic LOOCV 절편 누락, 최초 손실 DD 누락, 실제 파일 계약 오류 등을 작은 재현으로 확인했다. 거래비용·실시간 지원 부재를 새로운 확정 버그처럼 포장하지 않는다. 비용과 체결 설정은 실증 재현 전에 명시해야 할 조건이다.

## 재실행과 증거 해석

저장소 루트에서 Python3.11 환경에 `audit_tests/requirements-audit.txt`의 고정 의존성을 준비한 뒤 실행한다. 이 작업에서 보존한 실행기는 `venv/Scripts/python.exe`다. Windows에서는 아래 `python` 자리에 해당 경로를 사용한다. `-X utf8`은 한글 로그 인코딩을 고정한다.

```powershell
python -X utf8 audit_tests/run_audit.py --output reports/paper_audit/evidence
python -X utf8 audit_tests/check_artifacts.py --report-dir reports/paper_audit
```

첫 명령의 정상 종료는 0(FAIL 포함), 실행 도구 오류는 2다. 둘째는 문서·스키마·증거·해시가 일치하면0, 누락/모순이면1이다. checker는 현재 실행 코드의 해시까지 비교하므로 테스트 또는 checker 변경 후에는 runner를 다시 실행해야 한다. 이 보고서의 46/29/0/3은 78개 실험 기록만 세며 INVENTORY/PAPER-COVERAGE 두 개의 문서 근거는 포함하지 않는다.

지정 PDF는 저장소 부모의 `idea_paper.pdf`에서 해시로 찾는다. 다른 위치라면 `AUDIT_PAPER_PATH`에 정확한 파일을 지정한다. SHA-256은 `1dfd2574208a52effd3fa195e6005e33122b54f19220b4fb974cae603fc1aaab`이며 다른 PDF는 거부한다. 실행은 네트워크를 차단하고 원 함수/AST 및 원 스크립트를 임시 폴더에서 사용한다. 원 Step0 다운로드나 전체 실제 ETF 실증은 실행하지 않는다.

`environment.json`과 최신 `evidence/run_summary.json`은 기준 커밋·원본18파일 해시·실행 환경을 기록한다. `evidence/*/execution.json` 및 `verification.json`의 과거 작업폴더·옛 실행기 경로는 그때의 기록이다. 그 임시 폴더가 없어져도 최신 통합 실행의 재현에 필요하지 않으며, 최신 실행 증거로 오인하면 안 된다. 실험 입력·seed·독립 기준·오차는 각 results/fixtures 및 연결 CSV에 보존한다. 비유한 수는 JSON 문자열로 기록한다.

## 전체 범위와 남은 제한

`methodology.md`는 원문 26쪽, 모든 식1–19, Algorithm1과 각 방법의 기대 동작·모호함을 정리한다. `traceability.md`는 32개 방법/수식 ID를 코드·실험·발견에 연결한다. Figure1–13, Table1–6의 방법 매핑과 전쪽 시각 검토는 PAPER-COVERAGE에 있다. `evidence_index.md`는 PASS를 포함한 80개 근거 ID와 사용처를 열거한다.

실제 ETF/WRDS와 실시간 공개일·빈티지·체결 자료가 없어 원문의 수익률·통계적 유의성·실시간 투자 가능성을 확인하지 못했다. 작은 독립 산식과 제한된 실제 거시경제 창의 진단은 구현 결함의 존재를 보여 주지만 모든 역사 창의 빈도·성과 영향은 보여 주지 않는다. 논문 원문 자체가 불충분하게 정한 내용도 남아 있다. 따라서 **진단 작업의 완료**와 **연구 구현의 정확성 확보**를 구분한다.

# 자료·시간축 감사

## 범위와 증거

원 연구 코드·자료는 수정하지 않았다. 최상위 다운로드·원 위치 파일 저장을 실행하지 않았다. 원본 함수 및 전처리/단일 rolling 반복문 AST를 그대로 메모리에서 실행했다. `results.json` 각 항목의 `source_refs`는 실제 소스 SHA-256과 원래 줄 범위다. 원자료 해시는 `7494b30c6a3acbe46a9f9029e63db7cca6a5dd9dac0111e2f8d02ef62482f195`이다. 독립 산식 자체검사 9개를 먼저 통과한 뒤 원 구현과 비교했다. FAIL은 반례 발견이며 감사 도구 실패가 아니다. 전체 백테스트·수익률 성과 재현은 범위 밖이다.

## 실제 전처리 결과

- 원 CSV는 1959-01~2024-11, 791개월·121변수다. 실행은 변환 **전에** 1978-07~2024-09의 555행으로 자른다. 따라서 최초 1/2개월 차분 자료를 경계 밖 역사로 채우지 않는다.
- 그룹 처리 후 후보 121변수 → 내부 결측 비율 2% 초과 비보호변수 제거 → 남은 변수 121개 → complete-case 389개월이다. 시작 1992-03, 끝 2024-09. 삭제된 달 전체 166개와 달별 결측 원인은 `panel.json`에 있다.
- 실제 기간 내부 삭제월: 2020-04, 2020-05. README의 1992년 시작·2020년 4월 설명과 실제 결과를 분리해야 한다. 코드에 2020년 4월만 수동 삭제하는 문장은 없다. 제공 CSV의 4월 CP3Mx·COMPAPFFx가 결측이며, CP3Mx의 t-code 2 차분이 5월까지 결측을 전파한다. 해당 raw 결측을 누가 언제 만들었는지는 이 스냅샷만으로 확정할 수 없다.
- 그룹6 중 실제 유지 17개: FEDFUNDS, CP3Mx, TB3MS, TB6MS, GS1, GS5, GS10, AAA, BAA, COMPAPFFx, TB3SMFFM, TB6SMFFM, T1YFFM, T5YFFM, T10YFFM, AAAFFM, BAAFFM. 논문 p.7의 그룹6 제외와 다르다.
- 내부 gap 정책으로 제거된 변수: 없음.
- 실제 유지 변수: RPI, W875RX1, DPCERA3M086SBEA, CMRMTSPLx, RETAILx, INDPRO, IPFPNSS, IPFINAL, IPCONGD, IPDCONGD, IPNCONGD, IPBUSEQ, IPMAT, IPDMAT, IPNMAT, IPMANSICS, IPB51222S, IPFUELS, CUMFNS, HWI, HWIURATIO, CLF16OV, CE16OV, UNRATE, UEMPMEAN, UEMPLT5, UEMP5TO14, UEMP15OV, UEMP15T26, UEMP27OV, CLAIMSx, PAYEMS, USGOOD, CES1021000001, USCONS, MANEMP, DMANEMP, NDMANEMP, SRVPRD, USTPU, USWTRADE, USTRADE, USFIRE, USGOVT, CES0600000007, AWOTMAN, AWHMAN, HOUST, HOUSTNE, HOUSTMW, HOUSTS, HOUSTW, PERMIT, PERMITNE, PERMITMW, PERMITS, PERMITW, ACOGNO, AMDMNOx, ANDENOx, AMDMUOx, BUSINVx, ISRATIOx, M1SL, M2SL, M2REAL, BOGMBASE, TOTRESNS, NONBORRES, BUSLOANS, REALLN, NONREVSL, CONSPI, S&P 500, S&P div yield, S&P PE ratio, FEDFUNDS, CP3Mx, TB3MS, TB6MS, GS1, GS5, GS10, AAA, BAA, COMPAPFFx, TB3SMFFM, TB6SMFFM, T1YFFM, T5YFFM, T10YFFM, AAAFFM, BAAFFM, WPSFD49207, WPSFD49502, WPSID61, WPSID62, OILPRICEx, PPICMM, CPIAUCSL, CPIAPPSL, CPITRNSL, CPIMEDSL, CUSR0000SAC, CUSR0000SAD, CUSR0000SAS, CPIULFSL, CUSR0000SA0L2, CUSR0000SA0L5, PCEPI, DDURRG3M086SBEA, DNDGRG3M086SBEA, DSERRG3M086SBEA, CES0600000008, CES2000000008, CES3000000008, UMCSENTx, DTCOLNVHFNM, DTCTHFNM, INVEST, VIXCLSx.
- 모든 유지 날짜, 삭제 날짜, 변수별 처음/마지막 유효월·내부 결측률·비양수 원값·NaN/inf 개수는 `panel.json`에 저장했다. Step2의 실 ETF 교집합 달력은 실제 ETF 파일이 없어 확정할 수 없다.

## t-code와 특수값

6개 구현의 t-code 1~7을 양수·0·음수·내부 결측·상수 fixture 168회 비교했다. 핵심 경로 5개 구현은 모두 일치했다. 유일한 변환 불일치는 `etc_regime_stats_transformed.py`의 code7에서 `pct_change` 기본 결측 채움이 관측된 경우다(현재 pandas 실행 환경); 이 utility 분기 결과를 핵심 전략 변환 오류로 확대하지 않는다. code2/5는 첫 1행, code3/6/7은 첫 2행을 잃는다. 내부 결측/비양수 로그는 차분 횟수만큼 후속 행에 전파된다. 보호변수는 내부 gap이 커도 유지되어 해당 관측을 제거한다. 끝의 결측과 시작 전 결측은 내부 비율에 포함되지 않으므로 시작이 늦은 변수는 표본 전체 시작을 뒤로 미룬다. 상수는 complete-case에서는 유지되고 rolling 표준화에서 제외된다. code7의 0 분모는 inf를 만들고 dropna만으로는 제거되지 않는 반례를 확인했다. 이 특수값 요건은 일반 수치 타당성이며 원문의 구체 결측 정책은 아니다. 후속 zscore는 NaN 표준편차의 열을 묵시적으로 제외할 수 있으므로 열 제거와 행 제거는 다르다.

## 48행과 48개월

총 341개 macro rolling 창 중 47개가 연속 48개월이 아니다. 길이별 개수: {48: 294, 50: 47}. 다음 retained row를 다음 달로 부르는 다개월 이동 1개를 발견했다. 원문 Section 6의 48개월·1개월 앞 예측과 구분해야 한다. 같은 삭제 달력에서 전이 추정도 인접 행을 인접 월로 취급한다. ETF 로더 역시 전체 가격 panel에서 한 달이 통째로 없으면 shift(-1)가 다개월 가격변화를 한 달 라벨로 저장한다. 실제 Yahoo 자료에서 그 반례가 발생했는지는 일별 자료 부재로 미확정이다.

## 가격·목표·결정·보유 시점

월초 저장 라벨 R_t는 P(t+1)/P(t)-1이다. Step2의 x_tau→R_(tau+1), 마지막 학습 tau=t-1은 R_t를 사용한다. R_t는 P(t+1)을 관측해야 끝난다. Step5는 sasdate_tp1 라벨 수익률, 즉 정상 달력에서는 P(t+1)~P(t+2)를 보유 수익률로 쓴다. **결정을 P(t+1) 관측 후로 해석하면 target의 shift만으로 미래누출이라고 할 수 없다.** 같은 가격으로 주문·체결 가능한지는 구현에 없는 실행 가정이며 x_t 공개 가능성도 별도로 남는다. sasdate_t 자체를 매매 시각이라고 읽으면 R_t는 그때 알려져 있지 않다.

| 거시 기준월 t | 거시 실제 공개시점 | ETF 가격 관측 | 최신 학습 목표 가격구간 | 결정 시점 | 실제 보유 가격구간 |
|---|---|---|---|---|---|
| 1996-02-01 | 변수별 공개일 없음 | 첫 거래일 adjusted Close | first trading day 1996-02 to first trading day 1996-03 | 미명시; 정상 달력은 P(t+1) 관측 이후 해석 가능 | first trading day 1996-03 to first trading day 1996-04 |
| 2020-03-01 | 변수별 공개일 없음 | 첫 거래일 adjusted Close | first trading day 2020-03 to first trading day 2020-04 | 미명시; 정상 달력은 P(t+1) 관측 이후 해석 가능 | first trading day 2020-06 to first trading day 2020-07 |

각 열의 기계 판독본은 `timeline.json`이다. 첫 정상 예시의 macro는 제공 원자료에서, ETF 가격은 명시적 합성 fixture에서 가져왔다. 2020년 점프 예시는 실제 macro 날짜지만 ETF 가격구간은 로더/Step5 정의에 따른 상징적 해석이며 실제 가격 관측을 주장하지 않는다.

## 미래 변경 실험과 적합 범위

2015-01~2016-12 RPI raw 24셀만 NaN으로 바꾸고 **전체 원 전처리**부터 첫 rolling 반복문까지 재실행했다. 과거 raw는 동일하고 최초 결정일도 1996-02로 같은데 과거 변수집합 및 다음 레짐 확률이 변했다(최대 절대차 4.97834717338e-05). 이는 전체 표본의 >2% 내부 결측률로 변수를 고르는 경로의 일반 시계열 누출 반례다. 실제 과거 정보시점 기준 변수집합을 고정하거나 매 시점 선택하는 정책은 원문이 충분히 정하지 않는다. 보호변수 FEDFUNDS의 같은 미래 결측 변경은 미래 관측만 제거하고 해당 과거 결정은 바뀌지 않는 대조 실험을 수행했다.

Step1/2 표준화·PCA·군집은 각 48행 창에 적합한다. 상수열 제외도 창 안에서 일어난다. 그러나 그 앞의 변수/관측 선별은 전체 기간에 의존한다. Ridge의 최대 47개 학습 설명변수 행, 각 regime 표본선택, 목표 중심화와 lambda 선택의 범위를 원 loop에서 추적했다. lambda는 8개 후보의 analytic LOOCV이며 PCA/군집/중심화 등을 각 held-out fold에서 다시 적합하는 시간순 forward validation은 아니다. 이는 원문 미명세 추가 선택으로 기록한다.

원 Step2 첫 loop에 실제 macro와 합성 ETF를 넣고 P(t+1) 이후 가격만 1.7배 변경한 결과 이전 예측 최대 변화 0, lambda 변화 없음이다. 반면 마지막 학습 target R_t를 0.5 바꾸면 예측 최대 변화 0.114753813143다. 이는 target의 정보 상한을 확인하며 공개지연 문제를 해결하지 않는다. Section3는 전체표본 표준화/PCA인 회고 분석이고 미래 RPI 값만 바꿔도 과거 z-score가 변한다. 이를 그대로 실시간 신호로 재사용하는 경우와 Step1/2의 재적합을 구분한다.

## 공개시점·빈티지·자료 제공자

제공 입력은 `FRED-MD_2024m12.csv`로 명명된 단일 저장소 스냅샷이다(공식 원 빈티지와 byte/내용 동일성 미확인). 거시 기준월만 있고 각 값의 발표일/수정일/당시 빈티지 및 결정 시각이 없다. 과거 거래 시점에 사용할 수 있었던 최신 값인지 검증하는 as-of 비교는 SKIP이다. [St. Louis Fed의 FRED-MD 안내](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)는 자료 수정과 역사 빈티지를 설명한다. [FRED 실시간 기간 문서](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)는 관측 기간과 실시간 기간을 구별한다. [BLS CPI 발표 일정](https://www.bls.gov/schedule/news_release/cpi.htm)은 기준월과 발표 날짜/시각을 별도 열로 제시한다. 따라서 기준월과 첫 거래일을 같게 보는 것만으로 자료 가용성을 보장하지 못한다. 모든 변수에 일률적인 한 달 lag를 충분한 것으로 가정하지 않았다.

논문은 WRDS, 10 ETF, 2000-02~2022-12를 기술한다. 코드는 Yahoo/yfinance, auto_adjust=True의 Close, 1993-01-01~2024-10-31 다운로드를 요청한다. 소스/조정 방식/수집 빈티지/거래일 달력/커버리지의 수치 차이를 확정할 실제 가격 입력은 없다. 새 다운로드로 원 자료를 가장하지 않았고 실제 수익률 기간/결정 수/성과는 SKIP이다. 논문의 746개월 수치는 적힌 기간과 산술적으로 맞지 않는다.

## 합성 자료 계보

Step0의 `_make_synth_prices`는 이름과 달리 실제 Yahoo 함수로 위임한다. Step3/4는 선행 예측 부재 시 자동생성하며 ETF CSV도 없으면 seed=123 합성 수익률을 만든다. 두 경로가 `etf_bom_returns_aligned_demo.csv`라는 같은 파일명에 저장하므로 이후 파일명으로 실제/합성을 판별할 수 없다. 감사는 함수만 불러 표본을 확인하고 파일 저장은 실행하지 않았다. 모든 합성 fixture와 원본 분기의 계보는 `input_lineage.json` 및 개별 inputs에 표시했다.

## 판정 해석

- 논문 충실도: 그룹6 제외, 48개월/한 달 forecast 의무를 실행값과 비교했다.
- 일반 타당성: 미래 결측에 의한 과거 결정 변화, inf 잔존, 월 누락 가격의 잘못된 1개월 표시는 반례로 FAIL이다.
- 문서 일치: README 기간/삭제 설명은 별도 기준이다.
- 재현 가능성 한계: release/vintage와 WRDS/Yahoo 원 가격은 SKIP이다. 이들은 논문/코드의 구체적 결정 시점 미명세와 입력 부재를 숨기지 않는다.
- 현재 결과: {'PASS': 13, 'FAIL': 7, 'SKIP': 2}. 원 코드를 고쳐 FAIL을 제거하지 않았다.

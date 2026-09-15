# 성과지표·파이프라인 감사 (T6)

## 실행과 한계

결과: {'PASS': 8, 'FAIL': 9, 'ERROR': 0, 'SKIP': 1}. 원 연구 코드·원 CSV·NPZ·PDF·README·requirements는 수정하지 않았다. 원 함수/AST와 전체 Step3/4/5 및 시각화 스크립트를 사용했고, 모든 쓰기는 직렬 임시 폴더에서 수행했다. 네트워크를 차단했고 Step0 다운로드 함수는 호출하지 않았다. 원 코드의 기대 예외는 FAIL이며 감사 도구 예외 ERROR와 구분한다.

## 성과지표

초기 wealth=1을 포함한 손계산 [-0.1,0.05]는 wealth [0.9,0.945], drawdown [-0.1,-0.055], AvgDD=-0.0775, MaxDD=-0.1이다. 원 Step5는 첫 수익 후 wealth부터 고점을 잡아 **AvgDD=0, MaxDD=0**을 반환했다(MET-004).

Sharpe는 rf=0, sqrt(12)×월평균/표본표준편차(ddof=1)이다. Sortino는 음수 월만의 표본표준편차를 분모로 쓴다. 손실 0/1회이면 Sortino가 NaN이고 상수 양수 수익은 Sharpe도 NaN이다. 선언한 비교 기준 downside RMS=sqrt(sum(min(r,0)^2)/전체 월수)와 수치가 다르다. **원문은 Sortino 분모를 충분히 정하지 않아 이 차이를 원문 수식 위반으로 판정하지 않는다.** AvgDD는 음수 drawdown 기간만 평균한다. DD와 `% Positive Ret.` 저장값은 비율이며 0.75는 75%다.

## 원 백테스트·파일 계약

- 실제 Step5에서 [0.6,0.4]×[0.1,-0.05]=0.04, 다음 달 -0.2를 확인했다.
- 보유월 날짜가 없으면 입력 결정 4개 중 3개만 출력한다. 경고 없이 continue한다(MET-008).
- 비중 1인 SPY의 수익이 NaN이어도 np.nansum으로 포트폴리오 수익 0을 만든다(MET-009).
- EW는 NaN 종목을 제외해 10종목 중 1종목 누락 시 0.1/9=0.011111…를 반환한다. 0 대체 기준 0.01과 다르며, 논문 미정 결측 정책으로 기록했다.
- 시각화는 Step5와 다른 파일명을 읽어 FileNotFoundError를 낸다. 임시 복사본 이름만 맞추면 `Model` 대신 `Strategy`를 요구해 KeyError를 낸다. 진단용 열·모델명만 매핑한 뒤에도 `Ann.Return`, 이어서 `Ann.Vol`의 누락 오류가 실제 발생했다(MET-011~013). 임시 Ann.Return=0은 다음 열 접근을 노출하기 위한 fixture이며 수익률 추정이 아니다.
- Step5의 10% vol-target/log-return 그림은 마지막 주석뿐이다. 시각화도 원 수익을 바로 cumprod한다. 실행되는 스케일링·로그 그림 코드는 없다(MET-014).

## 입력 조합·합성 계보

Step2만 존재하고 Step1이 없으면 원 Step2 loader가 macro_ctx=None을 반환한다. 원 Step1 loader가 이를 unpack하면서 TypeError를 낸다(MET-015). 별도로 Step3/4의 실제 FRED 전처리와 missing-ETF 생성·저장 prefix AST를 실행했다. seed=123 합성 389행이 공통 이름 `etf_bom_returns_aligned_demo.csv`으로 저장되고 CSV에 합성 출처 표시나 sidecar가 없다(MET-016). 그 뒤 10창 RidgeCV 자동생성은 수행하지 않았다.

## 제한된 연결 실행

실제 FRED 첫 49 retained rows 중 48행(1992-03~1996-02)을 원 Step1/2 함수·첫 반복 AST에 넣고, seed=20260916의 명시적 합성 ETF 49×10을 사용했다. 생성된 선행 CSV를 임시 폴더에 모두 준 뒤 원 Step3→4→5 전체를 실행했다. 결정 1996-02, 보유 수익 라벨 1996-03의 1행, 지표 5행을 얻었다. 독립 식14·lo 비중·wR·날짜 연결이 일치했다(MET-017). 정확한 수치와 CSV는 connected_fixture 및 fixtures.json에 있다.

**연결 PASS는 파일·날짜·수치 전달만 뜻한다.** REG-018/019/022의 실제 Step1/2 레짐 분할·식별자 불일치 FAIL을 해소하지 않는다. 한 달의 합성 수익으로 논문 성과나 통계적 유의성을 재현했다고 주장하지 않는다. WRDS/실제 ETF 원 입력이 없으므로 실증 재현은 SKIP이다.

## 증거와 재실행

results.json은 원소스 SHA-256·심볼·줄범위, 입력·seed, expected/observed/oracle/tolerance를 기록한다. script_executions.json은 전체 스크립트의 실제 명령·stdout·stderr·종료 코드를 보존한다. execution.json은 진단 실행과 원본 18개 파일 전후 해시를 기록한다. 비유한 수는 JSON 문자열로 보존한다.

`python -X utf8 -m audit_tests.test_metrics_integration` 또는 `python -X utf8 audit_tests/run_audit.py --output reports/paper_audit/evidence`로 재실행한다. 정상 진단 완료는 FAIL이 있어도 종료 0이며 실행자 오류는 2다.

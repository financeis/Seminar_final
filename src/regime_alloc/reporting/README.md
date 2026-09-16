# 저장 증거의 보고서와 검증

`generate_report(run, output=None)`는 단일 실행 디렉터리, suite 디렉터리,
또는 아래 JSON을 받는다. 출력은 입력 실행 밖의 새 디렉터리여야 한다.
기본값은 UTC 시각을 붙인 형제 디렉터리다. 기존 출력에 덮어쓰지 않으며
원 실행과 자식 manifest는 수정하지 않는다. 실패하면 새 폴더에 failed
report manifest를 남긴다. 재시도는 다른 출력 경로를 사용한다.
요약 전 각 입력 실행의 succeeded 상태와 전체 산출물 목록·해시를 확인한다.
독립 smoke 월은 연결된 NAV 곡선 대신 월별 순수익 점으로 표시한다.

```json
{
  "kind": "research_evidence_bundle",
  "runs": [{"path": "../existing-run", "role": "baseline", "execution_scope": "saved candidate / 239 months"}],
  "references": [{"path": "../analysis/summary.json", "role": "full sample diagnostic"}],
  "unrun": ["100 controls over the full period"]
}
```

상대 경로는 bundle 파일 기준이다. references는 파일만 받으며 새 보고서의
evidence 폴더로 복사하고 원본/사본 해시를 기록한다. 데이터가 실제 수행한
범위를 execution_scope에 명시한다. 여러 버전의 증거를 합쳤다는 이유로
완료된 단일 suite로 바꾸지 않는다. 전표본 분석은 저장 그림을 재사용한다.

`verify_run(run, report=None)`는 JSON 직렬화 가능한 기록을 반환한다.
status가 succeeded이면 종료코드0, failed이면4로 처리한다. 기록 파일은
호출자가 입력 디렉터리 밖에 별도로 저장한다. 보고서는 선택사항이므로
smoke → verify → report 순서도 가능하다.

검증은 파일 목록과 해시, suite 예정 자식, 월·전략 패널, 저장 예측에서
기본 비중, 양쪽 곡선의 거래 비용·드리프트·자산·낙폭·지표를 독립 계산한다.
`REGIME_DATA_ROOT` 또는 suite의 기록된 data root가 있으면 고정 ETF 원가격도
대조한다. 없으면 저장 수익 패널 기준이라고 명시한다. 이 결과는 모델
재학습·변동성 추정기·통계 재표집·논문 전체의 인증이 아니다. 계산 당시
code_revision과 검사기 소스 해시를 구분한다. 과거 버전이라는 이유만으로
현 체크아웃 소스 해시와 일치시키거나 자동 복구하지 않는다.

보고서 검증은 파일·입력 해시와 EQ01~19, ALG1, FIG01~13, TABLE01~06,
F01~25 목록 및 실제 참조 파일을 확인한다. 빈 값의 경제적 의미와 미실행
범위는 본문과 metrics_reasons.json을 함께 읽는다. JSON은 UTF-8이며
NaN/Infinity를 허용하지 않는다.

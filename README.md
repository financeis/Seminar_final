# 거시경제 레짐 기반 ETF 자산배분

논문의 방법을 실제 Yahoo ETF·FRED-MD 자료로 다시 구현한 개인 연구용 Python 패키지입니다. 데이터 준비, 공통 레짐 추정, 예측, 비중 계산, 백테스트, 보고서를 역할별로 나눴습니다.

## 이번에 확인한 범위

- 두 자료 방식에서 2003-02~2022-12 **239개월 × 50개 전략**을 실행했습니다. 이 전체 기간 결과의 코드 버전은 `1541930`입니다.
- 후속 대조·민감도 구현 `ac069ce`에서는 두 자료 방식의 38개 짧은 실행과 8개 연속 구간 실행·재실행을 확인했습니다.
- 사용자 요청에 따라 100회 대조군의 전체 기간 실행, 모든 민감도의 전체 기간 실행, 최종 버전의 전체 묶음 반복은 생략했습니다. 따라서 논문의 모든 실험을 최종 코드로 완전히 재현했다는 의미는 아닙니다.
- 원 감사와 발견 25개는 `reports/paper_audit/`에 그대로 보존했습니다. 새 결과는 `reports/reproduction/`에서 읽습니다.

## 빠른 시작

저장소 루트의 PowerShell에서 실행합니다. 현재 준비된 독립 환경과 원자료를 사용합니다.

```powershell
& '.\.venv-research\Scripts\python.exe' -X utf8 -m regime_alloc --help
& '.\.venv-research\Scripts\python.exe' -X utf8 -m regime_alloc run --config configs/vintage_lagged.toml --output artifacts/my-smoke --smoke
& '.\.venv-research\Scripts\python.exe' -X utf8 -m regime_alloc report --run artifacts/my-smoke --output artifacts/my-smoke-report
& '.\.venv-research\Scripts\python.exe' -X utf8 -m regime_alloc verify --run artifacts/my-smoke --report artifacts/my-smoke-report
```

출력 폴더는 매번 새 이름을 사용합니다. `--smoke`는 최초·2020-04·마지막 월을 각각 확인하므로 이어 붙인 투자 성과가 아닙니다. 설치와 자료 확보, 전체 기간 명령은 [실행 안내](docs/operations.md)에 있습니다.

## 구조

```text
src/regime_alloc/
  data/        Yahoo·FRED·NBER 원자료, 거래일과 공개본
  features/    48개월 학습 창, 결측 처리, 표준화·PCA
  regimes/     공유 레짐 상태, 확률·전이
  models/      Naive·Ridge·BL·MVO, 검증 방식
  portfolio/   lo·lns·los·mx 비중, 변동성 조정
  backtest/    실행·회계·지표·대조·민감도
  reporting/   전체 표본 해석·보고서·산출물 점검
configs/       명시적 연구 가정
tests/         작은 수학 반례와 실제 자료 연결 검사
legacy/        원 코드의 보존 위치와 새 명령 대응
```

## 결과를 읽을 때

`fixed_snapshot`은 후대에 개정된 고정 공개본을 사용합니다. `vintage_lagged`는 과거 월별 공개본에 지연 가정을 적용합니다. 두 방식은 정보 수준이 다릅니다. Yahoo 수정주가도 논문의 WRDS와 동일한 원자료가 아닙니다.

2020년 3·4월을 제외하면 전체 표본의 R0가 1개에서 304개로 늘어나는 진단 결과가 나왔습니다. 당시 제외 의도는 이해할 수 있지만, 이는 사후 표본 진단입니다. 기본 백테스트는 중간 달을 지우지 않고 연속 48개월을 유지합니다.

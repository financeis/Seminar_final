# 원 구현 보존과 현재 실행 경로

원본 18개 파일의 경로와 승인 SHA-256은 [baseline_manifest.json](baseline_manifest.json)에 기록했다. 원본은 Git `9439107b31f20c4b731ec1dc29f13c178beb311d`에 보존되어 있다. 승인 해시는 Windows CRLF 체크아웃 기준이다. Git의 LF 텍스트는 CRLF로 복원하면 승인 해시와 정확히 일치하며, 바이너리는 변환 없이 일치한다. 각 항목에 Git blob 해시와 바이트 복원 방식을 함께 기록했다. 원 README, requirements, 자료·논문·NPZ도 이 목록에 포함한다. 큰 자료의 중복 복사본은 만들지 않았다.

최상위 Section·etc Python 스크립트는 현재 실행 경로에서 제거했다. 새 실행은 저장소 루트에서 `python -m regime_alloc`을 사용한다. 아래 대응은 역할 이관이며 옛 구현과 결과가 같다는 뜻은 아니다.

| 이전 경로 | 현재 모듈·명령 |
|---|---|
| Section3.py | src/regime_alloc/reporting/regime_analysis.py, regime_plots.py의 전체 표본 사후 진단 |
| Section5_step0_ETF_Loader.py | src/regime_alloc/data/; `data acquire --config configs/data.toml`, `data validate --config configs/data.toml` |
| Section5_step1.py | src/regime_alloc/features/, regimes/; `run --config FILE --output DIR` |
| Section5_step2.py | src/regime_alloc/models/ridge.py, window.py; 같은 run 명령 |
| Section5_step3.py | src/regime_alloc/models/와 regimes/의 확률 집계; 같은 run 명령 |
| Section5_step4.py | src/regime_alloc/portfolio/; 같은 run 명령 |
| Section5_step5.py | src/regime_alloc/backtest/; 같은 run 명령 |
| etc_regime_stats_transformed.py | src/regime_alloc/reporting/regime_analysis.py |
| etc_regime_visualization.py | src/regime_alloc/reporting/regime_plots.py |
| etc_visualize_transition_matrix.py | src/regime_alloc/regimes/transitions.py와 reporting/ |
| etc_visualize_backtest.py | src/regime_alloc/reporting/report.py; `report --run PATH --output DIR` |

## 역사 감사

`reports/paper_audit/`와 `audit_tests/`는 당시 내용을 유지한다. 원 감사의 재실행은 **별도 Git 281ca05 체크아웃에서만** 한다. 현재 루트는 옛 스크립트를 제거했으며 새 코드로 옛 해시·판정을 인증할 수 없다. 이번 마무리에서는 역사 감사를 다시 실행하지 않았다.

```powershell
git worktree add --detach ../Seminar_historical_audit 281ca05
```

새 폴더에서 해당 버전의 의존성과 논문 경로를 준비하고 `audit_tests/run_audit.py --output reports/paper_audit/evidence` 다음 `audit_tests/check_artifacts.py --report-dir reports/paper_audit`를 실행한다. 세부 조건은 [실행 안내](../docs/operations.md)에 있다. 일반적인 새 연구 실행에는 이 단계가 필요하지 않다.

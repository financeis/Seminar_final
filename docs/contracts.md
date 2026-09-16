# CLI와 결과 계약

인증이 없는 로컬 CLI다. 명령행 경로는 현재 작업 디렉터리, TOML 안의 상대 경로는 설정 파일 위치 기준이다. 파일은 UTF-8, 월은 YYYY-MM, 시각은 UTC offset을 포함하고 수익·비중은 소수 단위다.

| 명령 | 입력 | 결과와 주요 실패 |
|---|---|---|
| data acquire --config FILE | 제공자·기간·자료 루트·고정 ID | 새 원자료 manifest 또는 기존 스냅샷 검증. 다운로드·필수 자료·고정 ID 불일치 시 실패 |
| data validate --config FILE | 고정 자료 ID | 열·자산·달력·공개본·해시 검사. 누락·중복·손상 시 실패 |
| run --config FILE --output DIR | 단일 자료 방식·기간·가정 | 예측·비중·수익·지표·창별 상태·manifest. 자료 부족·수치 오류·출력 충돌 시 실패 |
| suite --config FILE --output DIR | 두 자료 방식·대조·민감도 설정 | 실행 계획·자식 결과·비교 통계. 예정 자식 실패를 숨기지 않음 |
| report --run PATH [--output DIR] | 단일 실행·묶음·연구 증거 bundle | 별도 한국어 보고서·CSV·JSON·PNG·SVG·report_manifest. 기존 출력과 손상된 입력은 실패 |
| verify --run PATH [--report DIR] | 실행·묶음·선택적 보고서 | status/checks/errors/scope JSON. supplied 결과의 해시·참조·독립 회계·지표 검증 |

run/suite의 --smoke는 떨어진 세 달의 연결 확인이다. --no-cache는 이전 실행 캐시를 읽지 않는 기본 정책이다. run은 continuous 기본 실행과 smoke를 결과에 구분한다. suite는 실행 전에 planned_runs/execution_plan을 기록한다.

종료코드는 성공 0, 설정 오류 2, 자료·실행 오류 3, 검증 실패 4다. 구조화한 오류는 error_code/reason/details를 담는다. verify의 실패는 errors 목록으로 반환한다. 지원 코드에는 invalid_config, missing_data, duplicate_key, nonfinite_value, calendar_gap, hash_mismatch, partition_mismatch, degenerate_partition, insufficient_history, numerical_failure, run_conflict, verification_failed가 있다.

실행 상태는 running/succeeded/failed다. run_manifest에는 코드·환경·설정·입력 ID·실제 기간·전략·파일 해시가 있다. CSV 키는 창/모형/자산 또는 월/전략이며 유일해야 한다. weights에는 CASH가 있고 returns에는 순수익·비용·자산·낙폭이 있다. metrics의 정의 불가 값은 비표준 NaN JSON 대신 사유 있는 null로 표시한다. 창별 NPZ는 pickle 없이 저장한다.

verify는 supplied 산출물과 산술을 검사한다. 모델 재적합·변동성 추정기 재검증·미실행 실험 인증은 하지 않는다. REGIME_DATA_ROOT가 있으면 고정 원가격까지 확인하며, 없으면 저장 자산수익을 입력으로 사용했다는 범위를 결과에 명시한다. 보고서 생성도 모델을 재학습하지 않는다.

현재 CLI는 regime_alloc만 사용한다. 구 감사 실행기는 현재 체크아웃에 포함하지 않는다.

# 진단 CLI 구현

이 디렉터리는 원문 구현 진단의 실행기, 원 소스 로딩, 독립 기대값, 증거 기록과 산출물 점검기를 소유한다. 일반적인 테스트 보조 폴더에 그치지 않고 run_audit.py와 check_artifacts.py라는 독립 CLI를 제공한다. 투자 전략의 새 구현이나 원 데이터 정정은 이 모듈의 역할이 아니다.

## 경계와 패턴

- 원 연구 .py와 원 CSV·PDF·NPZ는 읽기 대상으로 취급한다. 원 스크립트 전체 실행은 metrics_support의 임시 폴더 실행을 사용하며 원 저장소에서 최상위 모듈을 import하지 않는다.
- 원 함수/AST 호출과 독립 계산을 구분한다. 같은 이름의 중복·중첩 함수가 있으므로 실행 분기와 정확한 줄 범위를 기록한다. 편의를 위해 복사한 구현만 실행하고 원 구현 증거라고 부르지 않는다.
- 각 영역은 run(repo, output)으로 결과 목록을 반환하고 영역별 폴더에 기록한다. 같은 결과 폴더의 호출은 직렬로 유지한다. 예상한 원 코드 예외와 진단 코드 자체의 예외를 구분한다.
- common.record 형식을 유지하고 source_refs·inputs·oracle·tolerance·observed를 남긴다. seed가 없는 손계산은 그 사실을 적는다. NaN/Inf는 JSON에 비표준 숫자 리터럴로 쓰지 않는다.
- run_summary.json은 실제 통합 호출의 명령·환경·코드·결과 해시를 기록한다. 과거 개별 실행 기록을 현재 코드로 실행한 것처럼 갱신하지 않는다. 보고서 검사 증거는 evidence 밖에 두어 순환 해시를 피한다.

## 검증

원 결함을 재현한 FAIL은 유효한 진단 결과다. 감사 코드를 바꾸면 `python -X utf8 audit_tests/run_audit.py --output reports/paper_audit/evidence`와 이어서 `python -X utf8 audit_tests/check_artifacts.py --report-dir reports/paper_audit`를 실행한다. 두 명령의 정상 종료만으로 연구 구현 전체가 정확하다고 주장하지 않는다.

독립 기대값은 먼저 손계산 가능한 양성 대조로 확인한다. 데이터 변환의 결측·0분모, PCA 좌표의 부호·회전, 같은 레짐 동시 순열, Ridge 절편/LOO·표본 부족, 첫 손실·활성 NaN·누락/중복 날짜를 필요한 변경 범위에 맞춰 검증한다. 실증 자료가 없는 검사는 이유를 보존하고 합성 결과로 대체 성공시키지 않는다.

check_artifacts 변경은 정상 보고서 승인뿐 아니라 수식 누락·가짜 증거 ID·건수 불일치를 넣은 사본의 거부도 확인한다. 임시 손상본은 원본 보고서나 입력에 덮어쓰지 않는다. pinned 의존성은 requirements-audit.txt에 있으며 실험 중 네트워크를 사용하지 않는다.

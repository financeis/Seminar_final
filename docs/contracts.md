# 진단 CLI와 산출물 계약

명령은 저장소 루트에서 호출한다. 애플리케이션 인증은 없으며 로컬 파일 읽기·결과 폴더 쓰기 권한이 필요하다. 경로 옵션의 상대 경로는 현재 작업 디렉터리 기준이다. `--repo`의 기본값은 실행 스크립트가 속한 저장소다.

## 실험 실행

`python -X utf8 audit_tests/run_audit.py --output reports/paper_audit/evidence`

- 입력: 선택적 `--repo`, `--output`. 기본 출력은 reports/paper_audit/evidence다. 기준 원본 파일과 지정 PDF, 고정 패키지가 필요하다.
- 출력: 데이터·레짐·예측·성과/연결 영역의 results.json과 보조 증거, 전체 run_summary.json. stdout에는 영역별 및 전체 상태 건수를 출력한다.
- 종료: 정상 진단 완료는 0이며 FAIL을 포함할 수 있다. 실행기/영역 실행 오류, 유효하지 않은 결과 형식, 원본 해시 불일치는 2다. 원 스크립트에서 기대한 오류를 재현한 경우는 해당 성질의 FAIL로 기록한다.

## 산출물 점검

`python -X utf8 audit_tests/check_artifacts.py --report-dir reports/paper_audit`

- 입력: 선택적 `--repo`, `--report-dir`. 보고서 폴더에는 methodology.md, traceability.md, audit_report.md, findings.json, environment.json, evidence_index.md와 evidence가 있어야 한다. Git 기준 이력도 필요하다.
- 출력: stdout의 JSON 객체에 status, exit_code, errors, facts를 담는다. facts는 실제 검사 건수·발견 수·근거 수·원본 개수 등을 포함한다.
- 종료: 스키마·수식/방법 목록·증거 참조·보고서 일치·현재 소스와 원본 해시가 맞으면 0, 누락·모순·검사 실행 실패면 1이다. 연구 구현의 정확성을 인증하는 종료코드가 아니다.

## 공통 형식

실험 상태는 PASS(명시된 성질 성립), FAIL(그 성질 불성립), ERROR(검사를 끝내지 못함), SKIP(필요 자료 등 부재로 미실행)이다. 각 실험은 id, title, status, expected, observed, source_refs, inputs, oracle, tolerance, notes를 가진다. 비유한 수는 NaN/Infinity 등의 JSON 문자열로 보존한다.

findings.json은 발견 목록이다. 필수 필드는 id, title, category, severity, confidence, paper_refs, code_refs, expected, observed, impact, evidence_ids, recommendation, limitations다. limitations는 문자열 목록이고 없으면 []다. category는 implementation_error/paper_deviation/missing_method/paper_ambiguity/statistical_validity/reproducibility, severity는 critical/high/medium/low/info, confidence는 confirmed/supported/unresolved 중 하나다. 미구현은 code_refs=[]와 전체 조사 범위로 부재를 설명한다.

방법 판정은 일치·부분일치·불일치·미구현·논문모호·검증불가를 사용한다. 실험 상태와 방법 판정은 다른 분류다. 같은 ID가 가리키는 결과는 보고서와 JSON에서 같아야 한다. run_summary.json은 최신 통합 실행이며 과거 영역별 execution/verification 파일을 최신 실행으로 취급하지 않는다.

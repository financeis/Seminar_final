# 최신 프로젝트 정리

사용자 요청으로 현재 체크아웃에서 구 구현·감사 도구·자료 보존 폴더를 제거했다.

- 제거: audit_tests, legacy, reports/paper_audit, 예전 루트 CSV·NPZ·논문 사본·requirements.txt, 과거 감사 전용 결정 문서.
- 최신 Ridge 검증에 필요한 8행 예제만 tests/fixtures/ridge_loo_counterexample.json으로 옮겼다. 계산값은 바꾸지 않았다.
- 지정 원문은 docs/idea_paper.pdf로 복사했다. 최신 코드·설정·원자료·보고서·.venv-research는 유지했다.
- Git 임시 작업 복사본 두 개는 제거했다. 원격 게시·master 병합·Git 이력 삭제는 하지 않았다.
- 재귀 삭제가 자동 승인 정책에 막혀, 옛 venv·계획 파일·감사 캐시와 갱신 전 보고서는 저장소 밖 ../work/retired-project로 옮겼다. 이 항목은 디스크에서 완전히 폐기된 것이 아니다.
- 최신 보고서의 실제 계산 원본은 ../work/rebuild-execution에 계속 존재한다. 해당 자료는 이번 재구현 결과의 출처이므로 제거하지 않았다.

검증: Ridge·import 경계·보고서·검증기 테스트 17개 통과. 저장된 결과로 보고서를 갱신하고 산출물 목록·해시·참조 검사를 통과했다. 시장 실험이나 전체 테스트를 반복하지 않았다. 이전 통합 테스트 369개 기록은 당시 버전의 결과로 유지한다.

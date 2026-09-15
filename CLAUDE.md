# 거시경제 레짐 기반 자산배분 연구

미국 거시경제 자료로 레짐을 구분하고 ETF 예측·비중·백테스트를 계산하는 개인 연구용 Python 프로젝트다.
NumPy/Pandas 중심의 파일 기반 연구 코드와, 원문 방법·구현·실행 증거를 대조하는 로컬 진단 CLI가 있다. 공개 서비스나 자동 거래 시스템은 현재 제공하지 않는다.

## 프로젝트 안내

```text
Seminar_final/
├── CLAUDE.md                         → 프로젝트 개요와 작업 진입 기준
├── AGENTS.md                         → 동일한 프로젝트 기준
├── docs/
│   ├── architecture.md              → 연구 계산과 진단의 연결 구조
│   ├── business-rules.md            → 정보 시점·레짐·예측·평가의 의미
│   ├── security.md                  → 로컬 접근과 원본·출처 보호
│   ├── standards.md                 → 구현 경계와 검증 규칙
│   ├── engineering-notes.md         → 달력·PCA·LOOCV·재실행의 주의점
│   ├── operations.md                → 보존 환경과 새 환경의 실행 절차
│   ├── contracts.md                 → 두 CLI와 결과 형식
│   └── tracking/
│       ├── status.md                → 확인된 범위와 남은 연구 작업
│       ├── findings.md              → 해결하지 않은 연구 결함과 자료 한계
│       └── decisions/
│           ├── index.md             → 주요 선택의 목록
│           └── 0001-isolated-audit.md → 원 구현 격리 실행의 선택과 제약
└── audit_tests/
    └── AGENTS.md                    → 진단 CLI의 구현·검증 경계
```

## 핵심 조건

- 진단은 선택한 원본 버전의 실행 전후 해시를 보존한다. 다른 버전의 결과를 같은 기준의 증거로 인증하지 않는다.
- 합성 반례와 실제 시장 성과를 구분한다. 한 창의 연결 성공이나 정상 종료는 논문 전체 재현을 뜻하지 않는다.
- 논문 충실도·시계열 타당성·재현 가능성을 분리한다. 원문에 미정인 정책을 확정 코드 오류나 저자 의도로 단정하지 않는다.
- 코드·보고서 변경 후에는 독립 기대값과 최신 실행 증거를 대조한다. 결함을 숨기기 위해 기대값이나 허용오차를 조정하지 않는다.

## 작업 전 확인

기본적으로 docs/standards.md, docs/engineering-notes.md와 해당 모듈의 AGENTS.md를 읽는다. 감사 CLI는 audit_tests/AGENTS.md가 관할하며, 저장소 루트의 평면 연구 스크립트는 이 파일의 기준을 따른다.

- 수식·군집·Ridge를 바꾸기 전: reports/paper_audit/methodology.md의 원문 기대와 모호함, traceability.md의 실제 경로를 확인한다.
- 데이터·달력을 바꾸기 전: docs/business-rules.md의 수익 라벨과 정보 시점, evidence/data_timing/timeline.json의 구간을 확인한다.
- 원 스크립트를 실행하기 전: docs/operations.md의 진단 명령과 입력 조건을 확인한다. README의 순차 연구 실행에는 다운로드·파일 생성 부작용이 있다.
- 후속 수정의 범위를 정하기 전: docs/tracking/status.md와 reports/paper_audit/audit_report.md의 우선순위·검증 한계를 확인한다.

## 문제 처리

진단 중 원본이 바뀌거나 실자료와 합성자료가 구분되지 않거나, 미실행 결과를 실행 증거로 기록한 경우에는 해당 판정을 중단하고 사용자에게 즉시 알린다. 그 밖의 재현된 연구 결함은 영향·재현 조건·수정 방향을 기록한다. 현재 범위나 자료 부족 때문에 해결할 수 없는 문제만 사유와 함께 docs/tracking/findings.md에 남긴다.

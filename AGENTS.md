# 거시경제 레짐 기반 자산배분 연구

실제 Yahoo ETF와 FRED-MD로 레짐·예측·비중·백테스트를 계산하는 개인 연구용 Python 패키지다. NumPy/Pandas/SciPy/scikit-learn 기반의 로컬 파일 구조이며 공개 서비스나 실거래 시스템은 없다.

## 프로젝트 안내

```text
Seminar_final/
├── README.md                        → 실행 진입점과 확인 범위
├── AGENTS.md                        → 프로젝트 작업 기준
├── CLAUDE.md                        → 동일한 작업 기준
├── docs/
│   ├── architecture.md              → 계산 모듈의 연결
│   ├── business-rules.md            → 시간·레짐·예측·평가 정의
│   ├── security.md                  → 자료 접근과 보존
│   ├── standards.md                 → 구현·검증 규칙
│   ├── engineering-notes.md         → 달력·좌표·절편의 주의점
│   ├── operations.md                → 설치·실행·역사 감사
│   ├── contracts.md                 → CLI와 산출물
│   └── tracking/
│       ├── status.md                → 구현과 실제 실행 범위
│       ├── findings.md              → 남아 있는 자료·연구 한계
│       └── decisions/
│           ├── index.md
│           ├── 0001-isolated-audit.md
│           └── 0002-research-rebuild.md
├── src/regime_alloc/
│   ├── AGENTS.md                    → 공통 패키지 경계
│   ├── data/AGENTS.md               → 원자료·공개본·달력
│   ├── features/AGENTS.md           → 학습 표본과 변환
│   ├── regimes/AGENTS.md            → 공유 상태·확률·전이
│   ├── models/AGENTS.md             → 예측과 검증
│   ├── portfolio/AGENTS.md          → 비중·노출
│   ├── backtest/AGENTS.md           → 회계·실험 실행
│   └── reporting/AGENTS.md          → 설명과 산출물 점검
├── audit_tests/AGENTS.md            → 역사 감사 전용
└── legacy/README.md                 → 원 코드 보존과 새 경로 대응
```

## 작업 기준

코드 수정 전 docs/standards.md와 해당 모듈의 AGENTS.md를 읽는다. 수식·가정 변경은 reports/paper_audit/methodology.md의 원문과 모호함을 확인한다. 데이터·달력 변경은 결정·체결·목표 끝을 펼쳐 비교한다. 이전 감사의 ‘진단만’ 범위는 종료됐으며 새 패키지 수정은 허용됐다.

실제 시장 자료와 합성 반례, 코드 구현과 실제 실행을 구별한다. 오래된 결과를 최신 소스의 실행으로 인증하지 않는다. PCA 설명력을 예측 정확도라고 부르지 않는다. 원자료와 역사 감사는 그대로 보존한다.

검증은 변경에 필요한 수준으로 수행한다. 이미 통과한 검사를 이유 없이 반복하거나 장기 실험을 자동으로 늘리지 않는다. 발견한 오류는 재현 조건·영향·수정 근거를 기록하고 자료 한계는 코드 버그와 구분한다. 사용자 요청 없이 master 병합이나 push를 하지 않는다.

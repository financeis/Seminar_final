# 실행 안내

## 설치

저장소 루트에서 Python 3.11로 실행한다. 현재 컴퓨터에는 .venv-research가 준비돼 있다. 새 체크아웃에서는 다음 순서로 설치한다.

```powershell
py -3.11 -m venv .venv-research
& '.\.venv-research\Scripts\python.exe' -m pip install -r requirements-research.lock
& '.\.venv-research\Scripts\python.exe' -m pip install --no-deps --no-build-isolation -e .
```

설정 안의 상대 경로는 TOML 파일 위치 기준이다. REGIME_DATA_ROOT는 정식 자료 루트의 절대 경로를 덮어쓰는 선택적 환경변수다. 기본 자료는 data/raw 아래에 있다. 원자료는 Git에 없으므로 다른 컴퓨터에는 동일 스냅샷을 별도로 복사해야 기존 결과의 입력을 재현할 수 있다.

## 자료와 실행

```powershell
& '.\.venv-research\Scripts\python.exe' -X utf8 -m regime_alloc data validate --config configs/data.toml
& '.\.venv-research\Scripts\python.exe' -X utf8 -m regime_alloc run --config configs/vintage_lagged.toml --output artifacts/smoke-01 --smoke
& '.\.venv-research\Scripts\python.exe' -X utf8 -m regime_alloc report --run artifacts/smoke-01 --output artifacts/smoke-01-report
& '.\.venv-research\Scripts\python.exe' -X utf8 -m regime_alloc verify --run artifacts/smoke-01 --report artifacts/smoke-01-report
```

자료가 준비되지 않았으면 data acquire --config configs/data.toml을 사용한다. 기존 설정의 dataset ID는 이번에 확보한 특정 스냅샷이다. 재다운로드는 가격 개정·취득 시각 때문에 다른 ID가 될 수 있다. 새 취득에는 설정 복사본에서 세 dataset ID를 비우고 새 data root를 지정한 뒤, 반환된 ID를 후속 실험 설정에 고정한다. 과거 스냅샷과 같은 데이터라고 주장하지 않는다.

기본 전체 기간은 run 명령에서 --smoke를 빼면 실행된다. fixed_snapshot.toml은 후대 고정 공개본 비교이고 vintage_lagged.toml은 당시 공개본 지연 가정이다.

```powershell
& '.\.venv-research\Scripts\python.exe' -X utf8 -m regime_alloc suite --config configs/reproduction.toml --output artifacts/suite-01 --smoke
& '.\.venv-research\Scripts\python.exe' -X utf8 -m pytest -q
```

suite의 --smoke를 빼면 100개 대조군과 전체 민감도 묶음을 실행한다. 이번 마무리에서는 이 긴 실행을 수행하지 않았다. 결과 폴더는 매번 새 이름이어야 한다. --no-cache는 과거 실행 캐시를 읽지 않는 기본 정책을 명시한다. 실행 중에는 코드를 편집하지 않는다. 서버·배포 절차는 없다.

## 현재 프로젝트의 보존 자료

구 구현·옛 감사 실행기는 현재 체크아웃에서 제거했다. 지정 논문은 docs/idea_paper.pdf다. 최신 결과의 재검증에 필요한 원자료는 data/raw, 실제 실험 원본은 저장소 부모 work/rebuild-execution에 남아 있다. 과거 코드가 필요한 경우에만 Git 이력에서 별도로 확인한다.

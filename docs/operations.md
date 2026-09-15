# 실행 방법

## 현재 작업 폴더에서 재검증

Git 저장소와 기준 커밋, 원본 18개 파일, 지정 idea_paper.pdf, Python 3.11 환경이 필요하다. 아래는 현재 컴퓨터에 보존한 venv를 사용하는 PowerShell 명령이다.

```powershell
Set-Location 'C:\Users\imyon\Projects\renewal_seminar\Seminar_final'
& '.\venv\Scripts\python.exe' -X utf8 audit_tests/run_audit.py --output reports/paper_audit/evidence
if ($LASTEXITCODE -ne 0) { throw '진단 실행 오류: run_summary.json을 확인하세요.' }
& '.\venv\Scripts\python.exe' -X utf8 audit_tests/check_artifacts.py --report-dir reports/paper_audit
if ($LASTEXITCODE -ne 0) { throw '보고서·증거 일관성 검증 실패' }
```

첫 실행이 현재 코드의 해시와 실험 결과를 갱신하므로 순서를 바꾸지 않는다. 기본 evidence 폴더의 진단 산출물은 다시 작성되며 원 연구 입력은 그대로 읽는다. 과거 결과를 유지하려면 별도 결과 폴더를 지정하고 그 결과를 보고서에 연결해 관리한다.

## 환경을 새로 준비할 때

venv가 없는 새 체크아웃에서 먼저 Python 3.11 실행 파일을 확인한다. 현재 컴퓨터의 확인된 설치 경로를 사용하면 다음과 같다. 다른 컴퓨터에서는 첫 줄의 실행 파일 경로를 해당 Python 3.11 경로로 바꾼다.

```powershell
& 'C:\Users\imyon\.pyenv\pyenv-win\versions\3.11.9\python.exe' -m venv venv
& '.\venv\Scripts\python.exe' -m pip install -r audit_tests/requirements-audit.txt
```

의존성 설치에는 패키지 다운로드가 필요할 수 있지만 진단 실행은 네트워크를 사용하지 않는다. 현재 보존 환경은 system-site-packages를 공유하는 Python 3.11.9이며 고정 의존성 버전을 실제 import로 확인했다. 소스 복사본만 있고 기준 Git 이력이 없으면 산출물 점검기의 출처 검증이 실패한다.

## 입력과 환경 설정

- 지정 PDF는 저장소 부모 경로들에서 idea_paper.pdf를 찾아 SHA-256으로 확인한다. 해시는 1dfd2574208a52effd3fa195e6005e33122b54f19220b4fb974cae603fc1aaab이다. 저장소 내부의 다른 제목 PDF로 대체하지 않는다.
- 다른 위치에 둔 경우 현재 프로세스에만 `$env:AUDIT_PAPER_PATH = '지정 PDF의 절대 경로'`를 설정한다. 설정값이 있으면 그 파일만 검증하므로 잘못된 설정은 수정하거나 제거한다.
- `-X utf8`은 Windows의 한글·수식 로그 인코딩을 고정한다. 셸의 python 명령이 pyenv 설정 오류를 내면 보존 venv의 실행 파일을 직접 사용한다.
- 원 FRED CSV와 부록·NPZ는 Git의 기준 파일을 사용한다. 실제 ETF CSV를 새로 다운로드할 필요 없이 진단의 명시적 합성 입력으로 제한된 연결 검사를 수행한다.

최신 실행의 명령·종료·환경·해시는 reports/paper_audit/evidence/run_summary.json에 있다. 영역별 옛 execution/verification 기록의 임시 경로는 역사 기록이며 그 폴더를 복원할 필요가 없다. 별도 서버 시작, build, 배포 절차는 없다. README의 연구 파이프라인 전체 실행은 이 진단 CLI와 별개의 작업이다.

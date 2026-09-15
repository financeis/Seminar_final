# 레짐·확률·연결 감사 (S07)

## 범위와 판정

지정 원문의 해석은 `reports/paper_audit/methodology.md`를 기준으로 한다. 독립 손계산을 먼저 확인하고, 원본 함수 또는 명시된 AST 구간을 실행했다. 원 연구 파일·CSV·NPZ를 변경하거나 다운로드하지 않았다.

실행 결과: {'PASS': 14, 'FAIL': 8}. FAIL은 검사 대상 성질의 실패이며 감사 실행 오류가 아니다. undefined_boundary 분류의 FAIL은 실제 예외/NaN 관측을 뜻하며, 원문 미정 경계와 실제 FRED 발생 여부를 분리한다.

## Confirmed — 확인된 사실

- 식 (1)의 정상 입력, 실제 L2/제곱거리 구분, 식 (4)의 내부 p0 값과 합 정규화, 식 (7)의 행벡터 방향이 통과했다.
- Section3 식 (5)는 전체 출현 횟수를 분모로 쓰므로 마지막 레짐 행 합이 1 미만이다. Step1은 출발 횟수로 정규화하고 빈 행에 self-loop를 넣는다. 이는 원문의 문자 그대로 분모와 다른 명시적 보정이며, 원문 자체의 모호함과 분리한다.
- 원 매칭 함수에 같은 실물 중심의 PCA 부호·회전·차원 변화 반례를 직접 넣으면 식별 순열이 바뀐다. 중심 좌표의 앞 m개를 자르는 것만으로 서로 다른 PCA 기저가 정렬되지 않는다.
- Step1과 Step2는 실제 같은 FRED 48행 입력에서도 서로 다른 원시 레짐 식별자 및 일부 서로 다른 분할을 반환한다. 최적 정상 레짐 순열 정렬 후 차이도 REG-018에 보존했다.
- Step2는 Step1 군집 아티팩트를 읽지 않고 재군집하며 Step3는 두 날짜와 R 번호로 곱한다. REG-017의 동시 순열 불변성 및 단측 반례가 연결 의미를 검증한다.

## Supported — 범위가 제한된 영향 근거

- 실제 FRED 창과 고정 합성 자산 목표(seed 714)를 원 Ridge 루프에 넣었다. Step3의 정상 레짐 확률과 예측 열을 최적 멤버십 정렬 전후로 곱한 차이는 REG-019에 기록했다. 실제 ETF 수익·비중·성과 손실을 추정한 결과가 아니다.
- 진단용 ETF 달력에 대한 원 코드 교집합 실행은 Step2 첫 창과 매칭 시작점이 달라짐을 보인다. 같은 실제 창을 양쪽 cold start로 비교한 REG-018은 seed/n_init 차이의 실제 효과를 분리한 진단이며 전체 과거 매칭 상태의 재생은 아니다.

## Unresolved — 미확정 및 경계

- 추적된 실제 ETF CSV와 두 단계의 모델/군집 ID 계보가 없어 생산 실행의 정확한 시작 달력·전체 이전 PCA 기저·매칭 순열은 확정할 수 없다.
- 0벡터의 cosine, 모든 거리 0, 일반 군집 1개와 빈 전이 행은 원식이 유일한 답을 정하지 않는다. Step1 argmin fallback과 Section3 균등 fallback을 구현 선택으로 기록했다. p0=1의 epsilon clipping은 극한 R0=1의 유한 근사다.
- Section3 k-means++의 identical-vector 초기화 오류와 빈 중간 레짐의 NaN 전이 행은 경계 한계다. 정상 FRED 창 실패로 확대 해석하지 않는다.
- 차원 축소로 서로 다른 실물 중심이 완전히 같은 좌표가 되면 어느 알고리즘도 투영 좌표만으로 유일한 정체성을 복원할 수 없다.

## 논문 충실도와 통계적 타당성

논문 충실도: 식 (1), (4), (7)의 정상 경로는 수식과 일치한다. 식 (5)의 두 분모 구현과 p0/0거리 보정은 논문 경계·명세 문제로 별도 분류했다. 창 매칭에 특정 알고리즘을 논문 정답으로 가정하지 않았다.

통계적 타당성: 확률과 레짐별 조건부 예측은 같은 집합을 가리켜야 한다. 실제 분할 비교와 원 함수 기반 반례가 현재 연결의 미보장 상태를 확인한다. 좌표계 변환에 따른 임의 ID 변경도 별도의 수학적 실패다. 본 감사는 성과 재현이나 새 전략 설계가 아니다.

## 실제 FRED 창 수치 요약

| 창 | 원시 불일치 /48 | 최적 정상레짐 정렬 후 /48 | 정렬 전후 집계 예측 차이 |
|---|---:|---:|---:|
| 1992-03-01 ~ 1996-02-01 | 27 | 10 | 0.0017049672 |
| 1999-01-01 ~ 2002-12-01 | 7 | 7 | 0.0000000000 |
| 1999-02-01 ~ 2003-01-01 | 23 | 7 | 0.0000474746 |

집계 예측 차이는 고정 합성 목표와 원 Ridge·Step3 코드로 계산한 진단값이다. 실제 ETF 성과가 아니다.

## 증거 색인

| ID | 상태 | 분류 | 제목 |
|---|---|---|---|
| REG-001 | PASS | defined_property | Independent equations (1), (4), (5), (7) hand self-check |
| REG-002 | PASS | defined_property | Equation (1): finite nonnegative normalized probabilities on valid distances |
| REG-003 | PASS | defined_property | L2 distance is square-rooted before probability conversion |
| REG-004 | PASS | defined_property | Equations (2)-(4), (6): ordinary p0 and equality at 0.5 |
| REG-005 | PASS | implementation_choice | p0 endpoints: epsilon clipping differs from the exact limit |
| REG-006 | PASS | implementation_choice | Identical centers and insufficient normal clusters: unspecified Eq (1) boundaries |
| REG-007 | PASS | implementation_choice | Zero-vector cosine and hard-label ties are deterministic conventions |
| REG-008 | PASS | implementation_choice | Algorithm 1 equal-sized first-stage groups and normal-count padding |
| REG-009 | FAIL | undefined_boundary | Section3 k-means++ initialization on identical vectors |
| REG-010 | PASS | paper_ambiguity | Eq (5): counts, last-month denominator and empty row in both source paths |
| REG-011 | PASS | defined_property | Eq (7): row-origin/column-destination direction and probability conservation |
| REG-012 | FAIL | undefined_boundary | Section3 absent intermediate regime creates nonfinite transition row |
| REG-013 | FAIL | coordinate_invariance | PCA sign_flip: identical physical centroids retain identity only after basis alignment |
| REG-014 | FAIL | coordinate_invariance | PCA quarter_turn: identical physical centroids retain identity only after basis alignment |
| REG-015 | FAIL | coordinate_invariance | Changed PCA dimension: truncation does not establish a common coordinate system |
| REG-016 | PASS | mathematical_limit | Dimension reduction may also destroy identity information |
| REG-017 | PASS | defined_property | Eq (14) simultaneous permutation invariance and one-sided permutation counterexample |
| REG-018 | FAIL | cross_stage_identity | Actual identical FRED 48-row inputs: Step1/2 regime identifiers and partitions |
| REG-019 | FAIL | cross_stage_identity | Original Ridge and Step3 aggregation: consequence of unmapped regime identifiers |
| REG-020 | PASS | defined_property | Actual bounded windows: returned cluster labels agree with returned centers |
| REG-021 | PASS | source_path | Calendar clipping and independent matching histories have different first anchors |
| REG-022 | FAIL | cross_stage_identity | Actual source connection does not carry partition identity across Step1→Step2→Step3 |

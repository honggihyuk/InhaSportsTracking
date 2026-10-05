# 테스트 · 벤치마크 가이드

모든 명령은 저장소 루트에서 실행합니다.

## 1. 단위·통합 테스트

```bash
python -m pytest tests -q          # 68 passed (약 30 초, CPU)
```

YOLO 모델·GPU 없이 **합성 영상**과 **가짜 탐지기**로 실행됩니다.

| 파일 | 내용 |
|---|---|
| `tests/test_core.py` | 호모그래피, 탐지(클래스 ID 오프셋·COCO 필터), 추적(ID·공 ID 충돌), 파이프라인, 업로드·목록·Range, WebSocket |
| `tests/test_analysis.py` | 카메라 움직임 추정, 보정 전파, 발 위치 → 경기장 좌표, 박스 보간, 팀 분류, 분석 API |
| `tests/test_precise.py` | 고정밀 모드 — 경기장 라인 정렬, 공 타일 추론, 트랙 잇기·보간, 좌표 평활화, 골키퍼, 경기 지표, 분석 취소 |
| `tests/test_autocalib.py` | 자동 보정 — 직선 검출, 단일 프레임 보정(센터서클 포함), 거부, 교차 검증, 수동 키프레임 우선, API |

특정 파일·테스트만:

```bash
python -m pytest tests/test_autocalib.py -q
python -m pytest tests -q -k "calibrat"
```

## 2. 정량 벤치마크 — 정답이 있는 합성 중계 영상

```bash
python scripts/benchmark_precise.py --frames 300 --seed 0
```

팬·줌하는 중계 카메라 영상을 만들고(`pipeline/synthetic.py`), 정답 박스에 잡음·누락·가려짐·공 오탐을 섞은 모의 탐지기로 세 구성을 비교합니다.

| 구성 | 경기장 보정 |
|---|---|
| realtime | 수동 키프레임 1 개 (클릭 오차 ±1.5 px) |
| precise | 수동 키프레임 1 개 + 매 프레임 라인 정렬 |
| precise + 자동 보정 | **기준점 입력 없음** |

출력 지표: 보정 경기장 오차, 선수 위치 오차, 속력 오차, 정답 선수당 트랙 ID 수, 공 검출 비율 등. 결과 해석은 [4부 35절](../PROJECT_OVERVIEW_4_PRECISE_ANALYSIS.md#35-검증--정답이-있는-합성-중계-영상), [5부 42절](../PROJECT_OVERVIEW_5_AUTO_CALIBRATION.md#42-검증).

## 3. 프론트엔드

```bash
cd frontend
npm run lint      # oxlint
npm run build     # 프로덕션 빌드 (dist/)
```

## 4. 기타 점검 스크립트

| 명령 | 내용 |
|---|---|
| `python scripts/sample_test.py` | 호모그래피·3DGS 모듈 동작 확인, 2D 도식 샘플 영상 `data/raw/sample_video.mp4` 재생성 |
| `python scripts/test_roboflow_integration.py` | Roboflow API 키·다운로드·모델 로드·탐지 점검 ([Roboflow 가이드](./ROBOFLOW.md)) |
| `python -m pipeline.main_pipeline --video <영상>` | 파이프라인 CLI (앞 50 프레임) |

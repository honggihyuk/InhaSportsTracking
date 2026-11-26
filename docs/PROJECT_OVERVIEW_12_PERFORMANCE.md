# 12부. 처리 시간 단축과 측정 도구

후반기 보고서 6.2절의 향후 계획 2순위(처리 시간)와 5순위(하부리그 실증)에 대한 작업이다.

정밀 분석은 35초 영상에 728초(보정 제외)가 걸렸다. 늘어난 282초 가운데 274초가 SAM3 영상 추적이었고, 겹침 장면 31곳을 하나씩 따로 처리했다. 자동 보정은 키프레임 탐색에 60초, 앵커 추적에 20초가 들었다.

## 1. SAM3 위험 구간 묶음 처리 (`tracking/sam_video.py`)

### 문제
`repair_tracks` 는 끊김과 겹침 사건마다 다음을 했다.
- 영상을 다시 읽는다.
- `init_video_session` 으로 전처리한다.
- 모든 프레임의 ViT 영상 특징을 새로 계산하고 전파한다.

겹침 장면은 대개 몇 초 안에 몰려 있다. 그래서 같은 프레임의 영상 특징을 여러 번 계산했다. 전파 비용은 대부분 이 영상 특징 계산(ViT-L, 1008 px)이다.

### 방법
1. **구간 묶기** (`merge_windows`): 사건의 프레임 구간 `[시작, 끝]` 가운데 시간이 겹치거나 맞닿은 것을 한 덩어리로 묶는다. 덩어리 길이는 `max_span`(기본 120프레임) 이하로 둔다.
2. **한 세션에서 처리** (`Sam3VideoTracker.track_many`):
   - 덩어리마다 영상을 한 번 읽고 세션을 하나 만든다.
   - 세션 캐시를 `_FrameFeatureCache` 로 바꿔, 덩어리 안 모든 프레임의 영상 특징을 GPU 에 한 번만 계산해 둔다. transformers 기본 캐시는 1프레임만 보관한다.
   - 위치 임베딩은 프레임과 상관없이 같으므로 하나만 저장한다. 값이 다르면 그 프레임 것을 따로 저장한다.
   - 같은 프레임에서 시작하는 사건의 선수들은 한 번에 프롬프트하고 한 번 전파한다. 시작 프레임이 다르면 `reset_tracking_data()` 로 추적 상태(메모리)만 지우고 다시 전파한다. 이때 영상 특징은 그대로 재사용한다.
3. **결과가 같은 이유**
   - SAM 메모리(조건 프레임, 최근 6프레임, 객체 포인터)는 객체마다 따로 있다. 객체끼리 겹치지 않도록 막는 제약도 없다.
   - 사건마다 추적 상태를 새로 시작하므로, 각 객체의 마스크는 그 사건 구간만 잘라 따로 처리할 때와 같다.
   - ID 교체를 바로잡으면 뒤 사건의 프롬프트 박스가 바뀔 수 있다. 이 경우 그 사건만 바뀐 프롬프트로 다시 전파한다(`info['recomputed']`). 판정 순서와 규칙은 이전과 같다.
4. `track_many` 가 없는 추적기(테스트의 가짜 추적기 등)는 예전처럼 하나씩 처리한다. 설정으로 끌 수도 있다: `perf.sam_video.batch: false`.

### 검증
- 단위 테스트(`tests/test_sam_video_batch.py`): 구간 묶기 규칙과 영상 특징 캐시를 확인한다. 또 끊김 2곳, ID 교체 1곳, 교체 뒤 프롬프트가 바뀌는 겹침 1곳이 있는 장면에서 묶음 처리와 하나씩 처리의 결과를 비교한다. 이은 트랙, 교체, 고친 frames, 요약이 모두 같다. 선수 4명이 몰린 장면에서는 겹침 사건 여러 곳이 세션 1개로 처리되는 것도 확인했다.
- 실제 SAM3 스모크 검사(sports_video_1, 프레임 100~139, 사건 4개, 객체 8개):
  - 사건별 `track` 과 묶음 `track_many` 의 박스가 **227/227 모두 같았다** (IoU 1.0).
  - 시간은 57.2초에서 37.2초로 줄었다. 다만 섹터 A 작업이 GPU 를 95% 쓰던 상태라 시간 값은 참고만 한다.
  - 최대 VRAM 은 2.20 GiB 에서 2.56 GiB 로 늘었다. 영상 특징 보관에 프레임당 약 9 MB 가 들어, `max_span` 120이면 최대 약 1.1 GB 가 늘어난다.

## 2. 자동 보정 (`pipeline/autocalib.py`, `pipeline/calib_track.py`)

### 결과가 같은 최적화
| 위치 | 내용 | 효과 (sports_video_1) |
|---|---|---|
| `AutoCalibrator._stage1` | 가설 채점에 `einsum` 대신 BLAS 행렬곱 사용 (차이 1e-15) | 키프레임 탐색 `calibrate()` 71프레임: 78.4초 → 44.0초. 결과 71/71 동일 |
| `AutoCalibrator._render` | 경기장 라인을 촘촘한 점으로 바꾸는 계산을 매번 하지 않고 처음에 한 번만 함 | (위에 포함) |
| `LineScorer.dist` | 같은 마스크의 거리 변환을 다시 계산하지 않음 | 앵커 추적 약 1초 단축. 호모그래피 1054/1054, 신뢰도 동일 |

앵커 추적 시간은 대부분 `pitch.LineRefiner.refine`(ICP, 약 55%)이다. 이 파일은 섹터 B 소유가 아니어서 손대지 않았다.

### 조정 항목과 기본값
조정 항목은 `configs/model_config.yaml` 의 `perf:` 블록에 있다. 블록에 없는 키는 `video_analysis.PERF_DEFAULTS` 값을 쓴다.

| 키 | 기본값 | 근거 |
|---|---|---|
| `autocalib.every_s` | 0.5 (유지) | 1.0이면 탐색 45초 → 19초. 다만 정상 비율이 88.6% → 87.5% 로 떨어지고 최고 속력이 9.50 → 9.65 m/s 로 오른다. 영상 1편만 본 결과라 기본값은 유지하고 여러 편으로 다시 판단한다 |
| `autocalib.top_k` | 120 (유지) | 60이면 키프레임이 28개 → 23개로 줄고 정상 비율이 떨어진다 |
| `autocalib.refine_k` | **8 → 4** | 키프레임 28개와 호모그래피가 모두 같다(차이 0 m). 탐색은 45.0초 → 40.0초 |
| `calib_track.anchor_every_s` | 0.1 (유지) | 0.2면 앵커 추적이 17.8초 → 15.9초로 줄지만, 일부 프레임의 보정이 최대 2.6 m 달라진다 |

설정 비교 결과는 아래 표와 같다.
- 실행: `eval_suite recalib`, CPU, 보정 판정은 0.2초 간격으로 180프레임.
- 섹터 A 의 GPU 작업과 함께 돌린 시간이다.
- 정상 비율과 라인 오차는 eval_calibration 과 같은 기준이다.

| 설정 | 보정 정상 % | 라인 오차 px | 최고 속력 m/s | 키프레임 s | 앵커 s | 합계 s |
|---|---:|---:|---:|---:|---:|---:|
| 이전 기본 (0.5 / 120 / 8 / 0.1) | 88.6 | 2.00 | 9.50 | 45.0 | 17.8 | 62.8 |
| refine_k 4 (**새 기본**) | 88.6 | 2.00 | 9.50 | 40.0 | 17.7 | 57.7 |
| every_s 1.0 + refine_k 4 | 87.5 | 2.00 | 9.65 | 18.7 | 19.8 | 38.5 |
| anchor 0.2 | 88.6 | 2.00 | 9.48 | 43.3 | 15.9 | 59.2 |
| every_s 1.0 + refine_k 4 + anchor 0.2 | 89.8 | 2.00 | 9.85 | 18.9 | 15.6 | 34.5 |

위 표의 "이전 기본" 행에도 행렬곱 최적화가 이미 들어가 있다. 최적화 전 코드는 같은 조건에서 키프레임 탐색에 약 80초가 걸렸다.

## 3. 측정 도구 (`scripts/eval_suite.py`)

영상 × 설정마다 한 행씩 다음 지표를 잰다.
- 보정 정상 프레임 비율과 라인 오차(중앙값 px): `eval_calibration` 과 같은 판정이다.
- 공 검출 프레임, 선수 트랙 수, 화면 안 끊김 수, 등번호 확정 트랙 수: `compare_tracking` 과 같은 규칙이다.
- 최고 속력.
- 단계별 시간: 탐지·추적, 후처리, SAM3, 팀, 등번호, 키프레임, 앵커, 합계.
- 최대 VRAM.

결과는 `--out x.json` 과 `x.md`(Markdown 표)로 저장한다.

- `run`: 백엔드와 같은 순서(프로필 → 분석 → GPU 모델 내림 → 자동 보정)로 영상을 처음부터 분석한다. `--save-dir` 를 주면 결과 JSON 과 라인 마스크를 저장해 뒤에 다시 쓸 수 있다. 모델 경로는 실행 폴더 기준이므로 모델이 있는 체크아웃에서 실행한다.
- `recalib`: 저장된 결과로 자동 보정만 다시 한다(CPU).
- `eval`: 지표만 계산한다.
- `--variant 이름:구역.키=값,...` 를 여러 번 주면 설정 여러 개를 한 번에 비교한다.

단계별 시간을 재기 위해 `analyze_video` 결과에 `timings` 를 새로 기록한다(`detect_track`, `postprocess`, `sam_video`, `teams`, `identity`).

## 4. 섹터 A 병합 뒤 실행할 측정
모델과 영상이 있는 체크아웃 루트에서 실행한다.

```bash
# (1) SAM3 묶음 처리 전후: 시간, 이은 끊김, 바로잡은 교체, 트랙 수가 같은지 확인 (정밀 모드 전체, GPU)
python scripts/eval_suite.py run uploads/sports_video_1.mp4 --out data/eval/sam_batch.json \
    --save-dir data/eval/sam_batch --variant one:sam_video.batch=false --variant batch:sam_video.batch=true
python scripts/compare_tracking.py data/eval/sam_batch/one/sports_video_1.mp4.json \
    data/eval/sam_batch/batch/sports_video_1.mp4.json --labels one batch

# (2) 자동 보정 설정 비교 (CPU, 위 결과 재사용)
python scripts/eval_suite.py recalib data/eval/sam_batch/batch/sports_video_1.mp4.json --every 0.2 \
    --out data/eval/calib.json --variant base: --variant refine8:autocalib.refine_k=8 \
    --variant every1:autocalib.every_s=1.0 --variant anchor02:calib_track.anchor_every_s=0.2

# (3) 하부리그 실증 (5순위): 영상 여러 편 정밀 분석 지표
python scripts/eval_suite.py run uploads/*.mp4 --out data/eval/league.json --save-dir data/eval/league
```

`every_s: 1.0` 은 (2)와 (3)에서 여러 영상의 정상 비율과 최고 속력이 유지될 때만 기본값으로 바꾼다.

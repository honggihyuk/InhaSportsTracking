# 🔑 Roboflow API 키 설정 가이드

## 1. API 키 발급받기

1. [Roboflow Universe](https://app.roboflow.com/settings/api) 에 접속합니다.
2. 로그인 후 **API Key** 섹션에서 키를 복사합니다.

## 2. .env 파일 설정

프로젝트 루트에 있는 `.env` 파일을 열어 API 키를 입력합니다:

```bash
# .env 파일 편집
ROBOFLOW_API_KEY=실제_API_키_여기에_입력
```

**주의**: `your_roboflow_api_key_here` 를 실제 키로 교체해야 합니다!

## 3. 데이터셋 다운로드

API 키 설정 후 아래 명령어로 3개의 축구 전문 데이터셋을 다운로드합니다:

```bash
python setup_roboflow.py
```

### 다운로드되는 데이터셋

| # | 데이터셋 이름 | 용도 | 클래스 |
|---|-------------|------|--------|
| 1 | **football-players-detection-3zvbc** | 선수 탐지 | player, referee, goalkeeper |
| 2 | **football-ball-detection-rejhg** | 공 탐지 | soccer ball |
| 3 | **football-field-detection-f07vi** | 경기장 탐지 | field lines, goal, penalty area |

## 4. 다운로드 확인

다운로드가 완료되면 `datasets/` 폴더에 다음과 같이 구조가 생성됩니다:

```
datasets/
├── football-players-detection-3zvbc/
│   ├── train/
│   ├── valid/
│   └── test/
├── football-ball-detection-rejhg/
│   ├── train/
│   ├── valid/
│   └── test/
└── football-field-detection-f07vi/
    ├── train/
    ├── valid/
    └── test/
```

## 5. 모델 학습 (선택사항)

Roboflow 에서 바로 학습된 모델을 사용할 수도 있지만, 로컬에서 재학습하려면:

```bash
# YOLO11 모델로 학습 예시
yolo detect train data=datasets/football-players-detection-3zvbc/data.yaml model=yolo11s.pt epochs=100 imgsz=640
```

## 6. 문제 해결

### ❌ "Invalid API Key" 오류
- API 키를 정확히 복사했는지 확인하세요 (공백 없음)
- `.env` 파일 저장 후 터미널을 재시작하세요

### ❌ "Project not found" 오류
- 데이터셋이 공개인지 확인하세요
- 버전 번호가 올바른지 확인하세요 (v1, v2, v3...)

### ❌ "Rate limit exceeded" 오류
- Roboflow 무료 티어는 다운로드 제한이 있습니다
- 잠시 기다렸다가 다시 시도하세요

## 📚 추가 리소스

- [Roboflow 문서](https://docs.roboflow.com/)
- [YOLO11 가이드](https://docs.ultralytics.com/models/yolo11/)
- [축구 객체 탐지 튜토리얼](https://blog.roboflow.com/soccer-player-tracking/)

---

**다음 단계**: 데이터셋 다운로드 후 `python pipeline/main_pipeline.py --video <영상경로>` 실행

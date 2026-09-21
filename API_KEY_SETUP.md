# Roboflow API 키 설정 가이드

## 🔑 API 키 발급 방법

1. **Roboflow 웹사이트 접속**
   - https://app.roboflow.com/settings/api 로 이동합니다.

2. **로그인/회원가입**
   - Google, GitHub 계정으로 로그인하거나 이메일로 회원가입합니다.

3. **API 키 생성**
   - "Create New API Key" 버튼을 클릭합니다.
   - 키 이름을 입력합니다 (예: `InhaSportsTracking`).
   - 생성된 API 키를 복사합니다.

4. **.env 파일 수정**
   ```bash
   # .env 파일을 엽니다
   nano .env  # Linux/Mac
   notepad .env  # Windows
   
   # 아래 줄을 찾아 실제 API 키로 교체합니다
   ROBOFLOW_API_KEY=your_actual_api_key_here
   ```

5. **다운로드 스크립트 실행**
   ```bash
   python setup_roboflow.py
   ```

## 📦 다운로드되는 데이터셋

| 데이터셋 | 설명 | 클래스 |
|---------|------|--------|
| football-players-detection-3zvbc | 축구 선수, 심판, 골키퍼 탐지 | player, referee, goalkeeper |
| football-ball-detection-rejhg | 축구공 탐지 | ball |
| football-field-detection-f07vi | 축구장 라인, 영역 탐지 | field lines, penalty area, goal |

## ⚠️ 주의사항

- **API 키 보안**: `.env` 파일은 절대 GitHub 에 커밋하지 마세요!
- **`.gitignore` 확인**: `.env` 파일이 `.gitignore` 에 포함되어 있는지 확인하세요.
- **키 분실 시**: Roboflow 대시보드에서 키를 재생성할 수 있습니다.

## 🔍 문제 해결

### "API key does not exist" 오류
- API 키가 올바르게 복사되었는지 확인하세요 (공백 없음).
- Roboflow 대시보드에서 키 상태를 확인하세요.

### "invalid format" 오류
- 이미 자동 처리되지만, 필드 데이터셋은 `yolov8` 포맷으로 다운로드됩니다.

### 다운로드 실패
- 인터넷 연결을 확인하세요.
- Roboflow 서버 상태를 확인하세요: https://status.roboflow.com

#!/usr/bin/env python3
"""
Roboflow 데이터셋 다운로드 스크립트
.env 파일의 API 키를 사용하여 3개의 축구 전문 데이터셋을 다운로드합니다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 확인
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "your_roboflow_api_key_here":
    print("❌ 오류: .env 파일에 유효한 Roboflow API 키를 설정해주세요.")
    print("📝 발급 주소: https://app.roboflow.com/settings/api")
    print("\n💡 사용법:")
    print("1. .env 파일을 엽니다.")
    print("2. 'your_roboflow_api_key_here'를 실제 API 키로 교체합니다.")
    print("3. 이 스크립트를 다시 실행합니다.")
    exit(1)

print(f"✅ API 키가 확인되었습니다: {ROBOFLOW_API_KEY[:8]}...")

# roboflow 패키지 설치 확인
try:
    from roboflow import Roboflow
    print("✅ Roboflow 패키지가 설치되어 있습니다.")
except ImportError:
    print("❌ Roboflow 패키지가 설치되어 있지 않습니다.")
    print("📦 설치 명령: pip install roboflow")
    exit(1)

# 데이터셋 정보
DATASETS = [
    {
        "name": "football-players-detection-3zvbc",
        "workspace": "roboflow-jvuqo",
        "version": 3,
        "description": "축구 선수 탐지 모델"
    },
    {
        "name": "football-ball-detection-rejhg",
        "workspace": "roboflow-jvuqo",
        "version": 1,
        "description": "축구공 탐지 모델"
    },
    {
        "name": "football-field-detection-f07vi",
        "workspace": "roboflow-jvuqo",
        "version": 1,
        "description": "축구장 라인/영역 탐지 모델"
    }
]

# 다운로드 디렉토리 생성
DOWNLOAD_DIR = Path("datasets")
DOWNLOAD_DIR.mkdir(exist_ok=True)

print(f"\n📂 데이터셋 다운로드 위치: {DOWNLOAD_DIR.absolute()}")
print("=" * 60)

# Roboflow 초기화
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

# 각 데이터셋 다운로드
for dataset_info in DATASETS:
    try:
        print(f"\n🔍 [{dataset_info['description']}]")
        print(f"   프로젝트: {dataset_info['workspace']}/{dataset_info['name']}")
        print(f"   버전: v{dataset_info['version']}")
        
        # 프로젝트 로드
        project = rf.workspace(dataset_info["workspace"]).project(dataset_info["name"])
        
        # 버전 로드 및 다운로드
        version = project.version(dataset_info["version"])
        download_path = version.download(
            model_format="yolov11",  # 최신 YOLO11 형식
            location=str(DOWNLOAD_DIR),
            overwrite=False
        )
        
        print(f"   ✅ 다운로드 완료: {download_path}")
        
    except Exception as e:
        print(f"   ❌ 오류 발생: {str(e)}")
        print(f"   💡 해당 데이터셋의 버전 번호를 확인하거나 API 키 권한을 확인하세요.")

print("\n" + "=" * 60)
print("🎉 모든 데이터셋 다운로드가 완료되었습니다!")
print(f"\n📁 다운로드된 데이터셋은 '{DOWNLOAD_DIR}' 폴더에서 확인하실 수 있습니다.")
print("\n💡 다음 단계:")
print("1. configs/model_config.yaml 에서 모델 경로 확인")
print("2. python pipeline/main_pipeline.py --video <영상경로> 실행")

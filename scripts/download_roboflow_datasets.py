"""
Roboflow 데이터셋 다운로드 및 관리 모듈

사용법:
1. .env 파일에 ROBOFLOW_API_KEY 설정
2. download_all_datasets() 실행
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# 환경 변수 로드
load_dotenv()

class RoboflowDatasetDownloader:
    """Roboflow 축구 관련 데이터셋 다운로드 관리자"""
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Roboflow API 키 (없으면 환경 변수에서 로드)
        """
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError(
                "ROBOFLOW_API_KEY 가 설정되지 않았습니다.\n"
                "1. https://universe.roboflow.com 에서 API 키를 발급받으세요.\n"
                "2. .env 파일에 ROBOFLOW_API_KEY=your_key 로 설정하세요."
            )
        
        self.rf = Roboflow(api_key=self.api_key)
        self.datasets_dir = Path("data/roboflow_datasets")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터셋 정보
        self.datasets = {
            "players": {
                "workspace": "roboflow-jvuqo",
                "project": "football-players-detection-3zvbc",
                "version": int(os.getenv("FOOTBALL_PLAYERS_VERSION", 3)),
                "path": self.datasets_dir / "football-players-detection",
                "description": "축구 선수 탐지 (Player, Goalkeeper, Referee)"
            },
            "ball": {
                "workspace": "roboflow-jvuqo", 
                "project": "football-ball-detection-rejhg",
                "version": int(os.getenv("FOOTBALL_BALL_VERSION", 1)),
                "path": self.datasets_dir / "football-ball-detection",
                "description": "축구공 탐지"
            },
            "field": {
                "workspace": "roboflow-jvuqo",
                "project": "football-field-detection-f07vi",
                "version": int(os.getenv("FOOTBALL_FIELD_VERSION", 1)),
                "path": self.datasets_dir / "football-field-detection",
                "description": "축구 필드 라인/영역 탐지"
            }
        }
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> Path:
        """
        특정 데이터셋 다운로드
        
        Args:
            dataset_name: 'players', 'ball', 'field' 중 하나
            force: 기존 데이터셋 있어도 다시 다운로드
            
        Returns:
            다운로드된 데이터셋 경로
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")
        
        ds_info = self.datasets[dataset_name]
        
        # 이미 다운로드된 경우 스킵 (force=False 일 때)
        if ds_info["path"].exists() and not force:
            print(f"✅ {dataset_name} 데이터셋이 이미 존재합니다: {ds_info['path']}")
            return ds_info["path"]
        
        print(f"📥 {dataset_name} 데이터셋 다운로드 시작...")
        print(f"   Workspace: {ds_info['workspace']}")
        print(f"   Project: {ds_info['project']}")
        print(f"   Version: {ds_info['version']}")
        
        try:
            # Roboflow 프로젝트 로드
            project = self.rf.workspace(ds_info["workspace"]).project(ds_info["project"])
            
            # 버전 로드
            version = project.version(ds_info["version"])
            
            # YOLOv8/v11 형식으로 다운로드
            download_path = version.download(
                model_format="yolov8",  # YOLO11 과 호환
                location=str(ds_info["path"]),
                overwrite=force
            ).location
            
            print(f"✅ {dataset_name} 데이터셋 다운로드 완료: {download_path}")
            return Path(download_path)
            
        except Exception as e:
            print(f"❌ {dataset_name} 데이터셋 다운로드 실패: {str(e)}")
            raise
    
    def download_all_datasets(self, force: bool = False) -> dict:
        """
        모든 축구 데이터셋 다운로드
        
        Returns:
            {데이터셋명: 경로} 딕셔너리
        """
        results = {}
        
        for name in self.datasets.keys():
            try:
                path = self.download_dataset(name, force=force)
                results[name] = path
            except Exception as e:
                print(f"⚠️ {name} 데이터셋 다운로드 실패: {str(e)}")
                results[name] = None
        
        return results
    
    def get_yolo_config(self, dataset_name: str) -> dict:
        """
        YOLO 모델 학습을 위한 설정 반환
        
        Args:
            dataset_name: 'players', 'ball', 'field'
            
        Returns:
            YOLO 설정 딕셔너리
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        ds_info = self.datasets[dataset_name]
        data_yaml = ds_info["path"] / "data.yaml"
        
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml 이 없습니다: {data_yaml}")
        
        import yaml
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def list_classes(self, dataset_name: str) -> list:
        """
        데이터셋의 클래스 목록 반환
        
        Args:
            dataset_name: 'players', 'ball', 'field'
            
        Returns:
            클래스명 목록
        """
        config = self.get_yolo_config(dataset_name)
        return config.get('names', [])


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🚀 Roboflow 축구 데이터셋 다운로드")
    print("=" * 60)
    
    try:
        downloader = RoboflowDatasetDownloader()
        
        print("\n📋 다운로드할 데이터셋:")
        for name, info in downloader.datasets.items():
            print(f"   - {name}: {info['description']}")
        
        print("\n⏳ 다운로드 시작...\n")
        results = downloader.download_all_datasets(force=False)
        
        print("\n" + "=" * 60)
        print("📊 다운로드 결과:")
        print("=" * 60)
        
        for name, path in results.items():
            if path:
                classes = downloader.list_classes(name)
                print(f"✅ {name}: {path}")
                print(f"   Classes: {classes}")
            else:
                print(f"❌ {name}: 다운로드 실패")
        
        print("\n💡 다음 단계:")
        print("   1. detector.py 에서 다운로드된 모델 사용")
        print("   2. 실제 축구 영상으로 테스트")
        
    except ValueError as e:
        print(f"\n❌ 오류: {e}")
        print("\n📝 해결 방법:")
        print("   1. https://universe.roboflow.com 에 접속")
        print("   2. 우측 상단 프로필 → Settings → API Key 복사")
        print("   3. .env 파일 생성 후 ROBOFLOW_API_KEY=your_key 추가")


if __name__ == "__main__":
    main()

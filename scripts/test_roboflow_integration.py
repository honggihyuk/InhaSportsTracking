"""
Roboflow 데이터셋 테스트 스크립트

API 키 설정 후 이 스크립트를 실행하여 데이터셋 다운로드 및 모델 로드를 테스트합니다.
"""

import sys
from pathlib import Path

def test_imports():
    """기본 임포트 테스트"""
    print("=" * 60)
    print("📦 1. 임포트 테스트")
    print("=" * 60)
    
    try:
        from tracking.detector import RoboflowSoccerDetector
        print("✅ RoboflowSoccerDetector 임포트 성공")
    except Exception as e:
        print(f"❌ 임포트 실패: {e}")
        return False
    
    try:
        from scripts.download_roboflow_datasets import RoboflowDatasetDownloader
        print("✅ RoboflowDatasetDownloader 임포트 성공")
    except Exception as e:
        print(f"❌ 임포트 실패: {e}")
        return False
    
    return True


def test_env_setup():
    """환경 변수 설정 테스트"""
    print("\n" + "=" * 60)
    print("🔑 2. 환경 변수 설정 테스트")
    print("=" * 60)
    
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if api_key and api_key != "your_api_key_here":
        print(f"✅ API 키 설정됨: {api_key[:10]}...{api_key[-5:]}")
        return True
    else:
        print("⚠️  API 키가 설정되지 않았습니다.")
        print("\n📝 설정 방법:")
        print("   1. .env 파일 생성 또는 편집")
        print("   2. ROBOFLOW_API_KEY=your_actual_key 추가")
        print("\n💡 계속 진행하려면 API 키가 필요합니다.")
        print("   https://universe.roboflow.com 에서 발급받으세요.")
        return False


def test_dataset_download(api_key: str):
    """데이터셋 다운로드 테스트"""
    print("\n" + "=" * 60)
    print("📥 3. 데이터셋 다운로드 테스트")
    print("=" * 60)
    
    try:
        from scripts.download_roboflow_datasets import RoboflowDatasetDownloader
        
        downloader = RoboflowDatasetDownloader(api_key=api_key)
        
        print("\n다운로드할 데이터셋:")
        for name, info in downloader.datasets.items():
            print(f"   - {name}: {info['description']}")
        
        print("\n⏳ 다운로드 시작 (첫 번째 데이터셋만 테스트)...")
        
        # 첫 번째 데이터셋만 테스트
        path = downloader.download_dataset('players', force=False)
        
        if path and path.exists():
            print(f"✅ 다운로드 완료: {path}")
            
            # 클래스 확인
            classes = downloader.list_classes('players')
            print(f"   Classes: {classes}")
            return True
        else:
            print("❌ 다운로드 실패")
            return False
            
    except Exception as e:
        print(f"❌ 다운로드 중 오류: {e}")
        return False


def test_model_loading():
    """모델 로드 테스트"""
    print("\n" + "=" * 60)
    print("🤖 4. 모델 로드 테스트")
    print("=" * 60)
    
    try:
        from tracking.detector import RoboflowSoccerDetector
        
        # 기본 모델로 테스트 (YOLO11s)
        print("기본 모델 (YOLO11s) 로딩...")
        detector = RoboflowSoccerDetector(
            confidence_threshold=0.5,
            device='cpu'
        )
        
        print("✅ 모델 로드 성공")
        print(f"   Device: {detector.device}")
        print(f"   Classes: {len(detector.class_names)}개")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return False


def test_detection():
    """탐지 테스트"""
    print("\n" + "=" * 60)
    print("🎯 5. 객체 탐지 테스트")
    print("=" * 60)
    
    try:
        import cv2
        import numpy as np
        from tracking.detector import RoboflowSoccerDetector
        
        # 테스트 이미지 생성 (초록색 배경에 흰색 원)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (0, 128, 0)  # 초록색 배경 (축구장 느낌)
        
        # 사람 모양의 간단한 사각형 그리기
        cv2.rectangle(test_image, (200, 150), (250, 350), (255, 255, 255), -1)
        cv2.circle(test_image, (225, 130), 20, (255, 255, 255), -1)  # 머리
        
        # 공 모양의 원 그리기
        cv2.circle(test_image, (400, 300), 15, (255, 255, 255), -1)
        
        print("테스트 이미지 생성 완료 (640x480)")
        
        # 모델 로드
        detector = RoboflowSoccerDetector(
            confidence_threshold=0.3,  # 낮은 임계값으로 테스트
            device='cpu'
        )
        
        # 탐지 실행
        detections = detector.detect_frame(test_image)
        
        print(f"\n✅ 탐지 완료: {len(detections)}개 객체 발견")
        
        for i, det in enumerate(detections):
            print(f"   [{i}] {det['type']}: {det['class_name']} ({det['confidence']:.2f})")
            print(f"       BBox: {det['bbox']}, Center: {det['center']}")
        
        # 시각화
        output = detector.draw_detections(test_image, detections)
        
        # 결과 저장
        output_path = Path("data/test_detection_result.jpg")
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), output)
        print(f"\n📸 결과 이미지 저장: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 탐지 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 함수"""
    print("\n" + "🚀" * 30)
    print("   Roboflow 축구 데이터셋 & 모델 테스트")
    print("🚀" * 30 + "\n")
    
    # 1. 임포트 테스트
    if not test_imports():
        print("\n❌ 임포트 테스트 실패. 의존성 패키지를 확인하세요.")
        sys.exit(1)
    
    # 2. 환경 변수 테스트
    env_ok = test_env_setup()
    
    if not env_ok:
        print("\n⚠️  API 키 없이 계속 진행합니다 (기본 YOLO11 모델만 사용).")
        print("   전체 기능을 사용하려면 API 키를 설정하세요.\n")
        
        # 기본 모델만 테스트
        if test_model_loading():
            test_detection()
        sys.exit(0)
    
    # 3. 데이터셋 다운로드 테스트
    import os
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if test_dataset_download(api_key):
        print("\n✅ 데이터셋 다운로드 성공")
    else:
        print("\n⚠️  데이터셋 다운로드 실패. 네트워크 또는 API 키를 확인하세요.")
    
    # 4. 모델 로드 테스트
    if test_model_loading():
        print("\n✅ 모델 로드 성공")
    else:
        print("\n❌ 모델 로드 실패")
        sys.exit(1)
    
    # 5. 탐지 테스트
    if test_detection():
        print("\n✅ 탐지 테스트 성공")
    else:
        print("\n❌ 탐지 테스트 실패")
        sys.exit(1)
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("🎉 모든 테스트 완료!")
    print("=" * 60)
    print("\n💡 다음 단계:")
    print("   1. 실제 축구 영상을 data/raw/ 에 추가")
    print("   2. pipeline/main_pipeline.py 실행")
    print("   3. 결과 시각화 확인")


if __name__ == "__main__":
    main()

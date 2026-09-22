"""
Soccer 3D Digital Twin - FastAPI Backend Server
실시간 트랙킹 데이터 처리 및 WebSocket 스트리밍
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 로컬 모듈 임포트 (선택적)
import sys
sys.path.append(str(Path(__file__).parent.parent))

# 파이프라인은 선택적으로 임포트 (초기화 지연)
Soccer3DPipeline = None  # 필요시 지연 로드
Detection = None

# 앱 초기화
app = FastAPI(
    title="Soccer 3D Digital Twin API",
    description="실시간 축구 경기 트랙킹 및 3D 시각화 백엔드",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 "*" 허용, 프로덕션에서는 특정 도메인 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
pipeline: Optional[Soccer3DPipeline] = None
connected_clients: List[WebSocket] = []


# 데이터 모델
class TrackingData(BaseModel):
    frame: int
    timestamp: float
    players: List[Dict]
    ball: Optional[Dict]
    homography_matrix: List[List[float]]


class ViewState(BaseModel):
    view_mode: str  # "3d_free", "top_down", "player_cam", "broadcast"
    camera_position: Optional[List[float]] = None
    camera_target: Optional[List[float]] = None
    selected_player_id: Optional[int] = None


class VideoUploadResponse(BaseModel):
    status: str
    message: str
    video_path: Optional[str] = None
    frame_count: int = 0
    duration: float = 0.0


# 라우트 정의
@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Soccer 3D Digital Twin API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload_video": "/upload_video",
            "process_frame": "/process_frame",
            "get_tracking_data": "/get_tracking_data/{frame}",
            "websocket": "/ws/stream"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_initialized": pipeline is not None
    }


@app.post("/upload_video", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """비디오 파일 업로드 및 저장"""
    try:
        # uploads 디렉토리 생성
        uploads_dir = Path(__file__).parent.parent / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        # 파일 저장
        video_path = uploads_dir / file.filename
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 비디오 정보 추출 (간단한 구현)
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        
        return VideoUploadResponse(
            status="success",
            message=f"비디오 '{file.filename}' 업로드 완료",
            video_path=str(video_path),
            frame_count=frame_count,
            duration=duration
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 오류: {str(e)}")


@app.get("/process_frame/{frame_number}")
async def process_frame(frame_number: int, video_path: Optional[str] = None):
    """특정 프레임 처리 및 트랙킹 데이터 반환"""
    global pipeline
    
    if pipeline is None and Soccer3DPipeline is not None:
        # 파이프라인 초기화
        config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
        pipeline = Soccer3DPipeline(config_path=str(config_path))
    
    try:
        # 실제 구현에서는 비디오에서 해당 프레임 추출
        # 여기서는 더미 데이터 반환
        dummy_data = generate_dummy_tracking_data(frame_number)
        return dummy_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프레임 처리 오류: {str(e)}")


@app.get("/get_tracking_data/{frame}")
async def get_tracking_data(frame: int):
    """특정 프레임의 트랙킹 데이터 조회"""
    # 더미 데이터 생성 (실제 구현에서는 DB 또는 캐시에서 조회)
    return generate_dummy_tracking_data(frame)


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket을 통한 실시간 트랙킹 데이터 스트리밍"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        frame = 0
        while True:
            # 클라이언트로부터 메시지 대기 (예: 재생/일시정지 제어)
            data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
            
            # 메시지 처리
            if data:
                message = json.loads(data)
                if message.get("type") == "start":
                    frame = message.get("frame", 0)
                elif message.get("type") == "pause":
                    await asyncio.sleep(0.1)
                    continue
            
            # 트랙킹 데이터 생성 및 전송
            tracking_data = generate_dummy_tracking_data(frame)
            await websocket.send_json(tracking_data)
            
            frame += 1
            await asyncio.sleep(0.033)  # 약 30 FPS
    
    except Exception as e:
        print(f"WebSocket 오류: {e}")
    finally:
        connected_clients.remove(websocket)
        await websocket.close()


@app.post("/set_view_state")
async def set_view_state(view_state: ViewState):
    """카메라 시점 상태 설정"""
    # 실제 구현에서는 3D 렌더러에 시점 정보 전달
    return {
        "status": "success",
        "view_mode": view_state.view_mode,
        "message": f"시점이 {view_state.view_mode} 모드로 변경되었습니다."
    }


# 유틸리티 함수
def generate_dummy_tracking_data(frame: int) -> Dict:
    """더미 트랙킹 데이터 생성 (테스트용)"""
    np.random.seed(frame)  # 재현성을 위한 시드 설정
    
    # 선수들 생성 (11 명)
    players = []
    for i in range(11):
        players.append({
            "id": i,
            "position_3d": [
                float(np.random.uniform(-52.5, 52.5)),  # X: 경기장 길이
                float(np.random.uniform(-34.0, 34.0)),  # Y: 경기장 너비
                0.0  # Z: 지면
            ],
            "velocity": float(np.random.uniform(0, 8)),  # m/s
            "team": "home" if i < 6 else "away",
            "confidence": float(np.random.uniform(0.8, 1.0))
        })
    
    # 공 생성
    ball = {
        "position_3d": [
            float(np.random.uniform(-52.5, 52.5)),
            float(np.random.uniform(-34.0, 34.0)),
            float(np.random.uniform(0, 0.5))  # 공 높이
        ],
        "velocity": float(np.random.uniform(0, 20)),
        "confidence": float(np.random.uniform(0.9, 1.0))
    }
    
    # 호모그래피 행렬 (더미)
    homography_matrix = [
        [0.1, 0.0, -60.0],
        [0.0, 0.13, -45.0],
        [0.0, 0.0, 1.0]
    ]
    
    return {
        "frame": frame,
        "timestamp": datetime.now().timestamp(),
        "players": players,
        "ball": ball,
        "homography_matrix": homography_matrix
    }


# 서버 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화 작업"""
    print("🚀 Soccer 3D Digital Twin 서버 시작...")
    # 필요시 파이프라인 미리 초기화
    # config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
    # global pipeline
    # pipeline = Soccer3DPipeline(config_path=str(config_path))


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리 작업"""
    print("👋 서버 종료...")
    # 리소스 정리


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

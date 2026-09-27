"""
Soccer 3D Digital Twin - FastAPI Backend Server
실시간 트랙킹 데이터 처리 및 WebSocket 스트리밍
"""

import asyncio
import json
import shutil
from urllib.parse import quote
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 로컬 모듈 임포트 (선택적)
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Windows 한글 콘솔(cp949)에서 이모지 등 출력 불가 문자가 있어도 서버가 죽지 않도록
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

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
pipeline = None
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
STREAM_INTERVAL = 1 / 30  # 약 30 FPS

# 업로드된 경기 영상 제공 (StaticFiles 는 Range 요청을 지원 → 브라우저 영상 탐색 가능)
UPLOADS_DIR.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=UPLOADS_DIR), name="media")


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
    video_url: Optional[str] = None
    frame_count: int = 0
    fps: float = 0.0
    duration: float = 0.0


def _video_info(path: Path) -> Dict:
    return {
        "name": path.name,
        "url": f"/media/{quote(path.name)}",
        "size": path.stat().st_size,
        "modified": path.stat().st_mtime,
    }


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
            "videos": "/videos",
            "media": "/media/{filename}",
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
def upload_video(file: UploadFile = File(...)):
    """비디오 파일 업로드 및 저장

    동기 함수(def)로 선언 → FastAPI 가 스레드풀에서 실행하므로
    파일 I/O·cv2 디코딩이 이벤트 루프(WebSocket 스트림)를 막지 않음.
    """
    # 경로 조작 방지: 디렉토리 성분을 제거하고 파일명만 사용 ("../../x" → "x")
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="파일명이 올바르지 않습니다.")

    UPLOADS_DIR.mkdir(exist_ok=True)
    video_path = UPLOADS_DIR / filename
    try:
        # 전체를 메모리에 올리지 않고 청크 단위로 복사
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(str(video_path))
        try:
            opened = cap.isOpened()
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
    except OSError as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"업로드 오류: {e}")

    if not opened or frame_count <= 0:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"'{filename}' 은(는) 읽을 수 있는 비디오 파일이 아닙니다.")

    return VideoUploadResponse(
        status="success",
        message=f"비디오 '{filename}' 업로드 완료",
        video_path=str(video_path),
        video_url=_video_info(video_path)["url"],
        frame_count=frame_count,
        fps=fps,
        duration=frame_count / fps if fps > 0 else 0.0
    )


@app.get("/videos")
def list_videos():
    """업로드된 경기 영상 목록 (최신순)"""
    files = [p for p in UPLOADS_DIR.iterdir() if p.is_file()]
    return [_video_info(p) for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]


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
    # resend: 일시정지 중 탐색했을 때 해당 프레임을 한 번 전송하기 위한 플래그
    state = {"frame": 0, "paused": False, "resend": False}

    async def receive_controls():
        """클라이언트 제어 메시지(start/pause/seek) 수신 — 송신 루프와 독립적으로 동작"""
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "start":
                    state["frame"] = int(message.get("frame", state["frame"]))
                    state["paused"] = False
                elif message.get("type") == "pause":
                    state["paused"] = True
                elif message.get("type") == "seek":
                    # 영상 탐색 동기화: 지정 프레임으로 이동하고 재생/정지 상태를 함께 설정
                    state["frame"] = int(message["frame"])
                    state["paused"] = bool(message.get("paused", state["paused"]))
                    state["resend"] = True
            except (ValueError, TypeError, AttributeError, KeyError):
                continue  # 잘못된 JSON/필드는 무시하고 연결 유지

    receiver = asyncio.create_task(receive_controls())
    loop = asyncio.get_running_loop()
    next_tick = loop.time()
    try:
        while not receiver.done():  # 수신 태스크가 끝났다 = 클라이언트 연결 종료
            if not state["paused"] or state["resend"]:
                state["resend"] = False
                await websocket.send_json(generate_dummy_tracking_data(state["frame"]))
                if not state["paused"]:
                    state["frame"] += 1
            # 고정 sleep 대신 목표 시각까지 대기 → 전송 시간·타이머 오차가 누적되지 않음
            next_tick = max(next_tick + STREAM_INTERVAL, loop.time() - STREAM_INTERVAL)
            await asyncio.sleep(max(0.0, next_tick - loop.time()))
    except WebSocketDisconnect:
        pass
    finally:
        receiver.cancel()


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
    # 프레임별 독립 난수 생성기: 재현성 유지 + 전역 np.random 상태 오염 없음(동시 요청 안전)
    rng = np.random.default_rng(frame)

    # 선수들 생성 (11 명)
    players = []
    for i in range(11):
        players.append({
            "id": i,
            "position_3d": [
                float(rng.uniform(-52.5, 52.5)),  # X: 경기장 길이
                float(rng.uniform(-34.0, 34.0)),  # Y: 경기장 너비
                0.0  # Z: 지면
            ],
            "velocity": float(rng.uniform(0, 8)),  # m/s
            "team": "home" if i < 6 else "away",
            "confidence": float(rng.uniform(0.8, 1.0))
        })

    # 공 생성
    ball = {
        "position_3d": [
            float(rng.uniform(-52.5, 52.5)),
            float(rng.uniform(-34.0, 34.0)),
            float(rng.uniform(0, 0.5))  # 공 높이
        ],
        "velocity": float(rng.uniform(0, 20)),
        "confidence": float(rng.uniform(0.9, 1.0))
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

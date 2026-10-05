"""
Soccer 3D Digital Twin - FastAPI Backend Server
실시간 트랙킹 데이터 처리 및 WebSocket 스트리밍
"""

import asyncio
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


# ---------------------------------------------------------------------------
# 영상 분석 (탐지·추적 → 결과 JSON, 경기장 보정)
# ---------------------------------------------------------------------------
ANALYSIS_DIR = UPLOADS_DIR / "analysis"
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_config.yaml"
analysis_jobs: Dict[str, Dict] = {}  # 영상 이름 → {state, progress, error}
# CPU 추론은 동시에 돌리면 서로 느려지기만 하므로 한 번에 하나씩 처리
# ponytail: 메모리 내 작업 상태 — 서버 재시작 시 진행 중 작업은 사라짐 (완료 결과는 파일로 남음)
analysis_executor = ThreadPoolExecutor(max_workers=1)


class CalibrationPoint(BaseModel):
    image: List[float]  # [u, v] 원본 해상도 픽셀
    pitch: List[float]  # [x, y] 경기장 좌표 (m)


class CalibrationKeyframe(BaseModel):
    frame: int
    points: List[CalibrationPoint] = []
    homography: Optional[List[float]] = None  # 자동 보정 키프레임: 이미지 → 경기장 3x3 (기준점 대신)
    source: Optional[str] = None              # 'auto' | None(수동)
    score: Optional[Dict[str, float]] = None  # 자동 보정 일치도


class CalibrationRequest(BaseModel):
    keyframes: List[CalibrationKeyframe]
    refine: Optional[bool] = None  # 경기장 라인 정렬 (None: 정밀 모드 라인 마스크가 있으면 사용)


def _video_file(name: str) -> Path:
    path = UPLOADS_DIR / Path(name).name  # 경로 조작 방지
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"'{name}' 영상이 없습니다.")
    return path


def _analysis_file(name: str) -> Path:
    return ANALYSIS_DIR / f"{Path(name).name}.json"


def _get_pipeline():
    """탐지 모델은 무거우므로 첫 분석 때 한 번만 로드 (분석 작업 스레드에서만 호출)"""
    global pipeline
    if pipeline is None:
        import os
        import torch
        # 분석이 CPU 를 전부 쓰면 같은 PC 의 브라우저(3D 화면)와 API 응답이 멈춘 듯 느려진다 → 코어 2 개는 남김
        threads = max(1, (os.cpu_count() or 2) - 2)
        torch.set_num_threads(threads)
        cv2.setNumThreads(threads)
        from pipeline.main_pipeline import Soccer3DPipeline as Pipeline
        p = Pipeline(config_path=str(CONFIG_PATH))
        p.initialize_detector()
        pipeline = p
    return pipeline


def _lines_file(name: str) -> Path:
    """정밀 모드에서 저장하는 프레임별 경기장 라인 마스크 (보정 시 라인 정렬에 사용)"""
    return ANALYSIS_DIR / f"{Path(name).name}.lines.npz"


class AnalysisCancelled(Exception):
    pass


def _run_analysis(name: str, mode: Optional[str] = None):
    from pipeline import video_analysis
    from pipeline.pitch import LineMaskStore
    from tracking.tracker import BallTracker

    job = analysis_jobs[name]
    if job.get("cancel"):
        return

    def progress(done, total):
        if job.get("cancel"):
            raise AnalysisCancelled()
        job.update(progress=done / max(total, 1))

    try:
        job["state"] = "running"
        p = _get_pipeline()
        profile = p.analysis_profile(mode) if hasattr(p, "analysis_profile") else {
            "mode": mode or "realtime", "stride": int(p.config.get("analysis", {}).get("stride", 1))}
        job["mode"] = profile["mode"]
        stride = int(profile.get("stride", 1))
        det = p.detector
        for attr, key in (("imgsz", "img_size"), ("ball_imgsz", "ball_imgsz"), ("ball_tile", "ball_tile")):
            if key in profile and hasattr(det, attr):
                setattr(det, attr, profile[key])  # 탐지 모델은 재사용하고 추론 설정만 모드별로 바꿈
        tracker = p.make_tracker()  # 영상마다 추적 ID 를 새로 시작
        tracker.ball_tracker = BallTracker(max_jump=60 * stride)  # 탐지 간격만큼 공 이동 허용 범위 확대
        lines = LineMaskStore((0, 0)) if profile.get("line_refine") else None
        result = video_analysis.analyze_video(
            _video_file(name), det, tracker, stride=stride, mode=profile["mode"],
            line_store=lines, postprocess=profile.get("postprocess"), progress=progress)
        _lines_file(name).unlink(missing_ok=True)
        if lines is not None and len(lines):
            ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
            lines.save(_lines_file(name))
            if profile.get("auto_calibrate"):
                job.update(stage="calibrating", progress=0.0)  # 사람 입력 없이 경기장 보정까지
                _try_auto_calibrate(result, lines, progress)
        video_analysis.save(result, _analysis_file(name))
        job.update(state="done", progress=1.0)
    except AnalysisCancelled:
        job.update(state="cancelled")
    except Exception as e:  # 작업 스레드 예외는 상태로 전달
        print(f"분석 실패 ({name}): {e}")
        job.update(state="error", error=str(e))


def _try_auto_calibrate(result: Dict, lines, progress=lambda d, t: None) -> Dict:
    """자동 보정 시도 → 결과를 result['auto_calibration'] 에 기록 (실패해도 분석 결과는 유지)"""
    from pipeline import video_analysis
    try:
        info = video_analysis.auto_calibrate(result, lines, progress=progress)
        result["auto_calibration"] = {"status": "ok", **info}
    except ValueError as e:
        result["auto_calibration"] = {"status": "failed", "reason": str(e)}
    return result["auto_calibration"]


def _run_auto_calibration(name: str):
    """분석이 끝난 영상의 자동 보정 작업. 라인 마스크가 없으면(빠른 미리보기 결과) 영상을 다시 읽어 만든다"""
    from pipeline import video_analysis
    from pipeline.pitch import LineMaskStore, video_line_masks

    job = analysis_jobs[name]
    if job.get("cancel"):
        return

    def progress(done, total):
        if job.get("cancel"):
            raise AnalysisCancelled()
        job.update(progress=done / max(total, 1))

    try:
        job["state"] = "running"
        result = video_analysis.load(_analysis_file(name))
        if _lines_file(name).is_file():
            lines = LineMaskStore.load(_lines_file(name))
        else:
            job.update(stage="lines")
            lines = video_line_masks(_video_file(name), result["frames"], max_frames=result["frame_count"],
                                     progress=progress)
            lines.save(_lines_file(name))
        job.update(stage="calibrating", progress=0.0)
        _try_auto_calibrate(result, lines, progress)
        video_analysis.save(result, _analysis_file(name))
        job.update(state="done", progress=1.0)
    except AnalysisCancelled:
        job.update(state="cancelled")
    except Exception as e:
        print(f"자동 보정 실패 ({name}): {e}")
        job.update(state="error", error=str(e))


def _analysis_status(name: str) -> Dict:
    job = analysis_jobs.get(name)
    if job and job["state"] in ("queued", "running", "error"):
        return {"name": name, **{k: v for k, v in job.items() if k != "cancel"}}
    path = _analysis_file(name)
    if path.is_file():
        from pipeline import video_analysis
        result = video_analysis.load(path)
        return {"name": name, "state": "done", "progress": 1.0,
                "calibrated": result.get("calibration") is not None,
                "mode": result.get("mode", "realtime"),
                "line_refine": _lines_file(name).is_file(),
                "has_stats": result.get("stats") is not None,
                "auto_calibration": result.get("auto_calibration")}
    if job and job["state"] == "cancelled":
        return {"name": name, "state": "cancelled", "progress": job.get("progress", 0.0)}
    return {"name": name, "state": "none", "progress": 0.0}


@app.post("/analysis/{name}")
def start_analysis(name: str, mode: Optional[str] = None):
    """영상 분석 시작 (백그라운드). mode: realtime | precise (생략 시 설정 기본값). 이미 진행 중이면 409"""
    _video_file(name)
    if mode is not None and mode not in ("realtime", "precise"):
        raise HTTPException(status_code=400, detail="mode 는 realtime 또는 precise 입니다.")
    if analysis_jobs.get(name, {}).get("state") in ("queued", "running"):
        raise HTTPException(status_code=409, detail="이미 분석 중입니다.")
    analysis_jobs[name] = {"state": "queued", "progress": 0.0, "mode": mode, "stage": "analyzing"}
    analysis_executor.submit(_run_analysis, name, mode)
    return _analysis_status(name)


@app.delete("/analysis/{name}")
def cancel_analysis(name: str):
    """진행 중(대기 포함)인 분석 취소 — 다음 프레임 처리 시점에 멈춤. 기존 결과 파일은 유지"""
    _video_file(name)
    job = analysis_jobs.get(name)
    if not job or job["state"] not in ("queued", "running"):
        raise HTTPException(status_code=409, detail="진행 중인 분석이 없습니다.")
    job["cancel"] = True
    if job["state"] == "queued":
        job["state"] = "cancelled"
    return {"name": name, "state": "cancelling" if job["state"] == "running" else "cancelled"}


@app.get("/analysis/{name}/status")
def analysis_status(name: str):
    _video_file(name)
    return _analysis_status(name)


@app.get("/analysis/{name}")
def get_analysis(name: str):
    """분석 결과 JSON (프레임별 박스·카메라 움직임·팀·경기장 좌표)"""
    path = _analysis_file(name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="분석 결과가 없습니다.")
    return FileResponse(path, media_type="application/json")


@app.post("/analysis/{name}/calibration")
def save_calibration(name: str, body: CalibrationRequest):
    """키프레임 보정점 저장 → 카메라 움직임으로 전 프레임에 전파해 경기장 좌표 계산"""
    from pipeline import video_analysis

    path = _analysis_file(name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="먼저 영상을 분석하세요.")
    if not body.keyframes:
        raise HTTPException(status_code=400, detail="보정 키프레임이 없습니다.")
    for kf in body.keyframes:
        if any(len(p.image) != 2 or len(p.pitch) != 2 for p in kf.points):
            raise HTTPException(status_code=400, detail="보정점 좌표는 [x, y] 형식이어야 합니다.")
        if kf.homography is not None and len(kf.homography) != 9:
            raise HTTPException(status_code=400, detail="homography 는 9 개 값이어야 합니다.")
    result = video_analysis.load(path)
    lines = None
    if body.refine is not False and _lines_file(name).is_file():
        from pipeline.pitch import LineMaskStore
        lines = LineMaskStore.load(_lines_file(name))  # 정밀 모드: 매 프레임 경기장 라인 정렬
    try:
        video_analysis.calibrate(result, [kf.model_dump(exclude_none=True) for kf in body.keyframes],
                                 line_masks=lines)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    video_analysis.save(result, path)
    covered = sum(w is not None for w in result["world"])
    return {"name": name, "calibrated_frames": covered, "frame_count": result["frame_count"],
            "refined_frames": result["calibration"].get("refined_frames"),
            "has_stats": result.get("stats") is not None}


@app.post("/analysis/{name}/calibration/auto")
def start_auto_calibration(name: str):
    """
    자동 경기장 보정 시작 (백그라운드) — 기준점 없이 경기장 라인·센터서클로 키프레임을 찾아 전 프레임 보정.
    진행 상황은 /status (stage: lines → calibrating), 결과는 status.auto_calibration
    """
    _video_file(name)
    if not _analysis_file(name).is_file():
        raise HTTPException(status_code=404, detail="먼저 영상을 분석하세요.")
    if analysis_jobs.get(name, {}).get("state") in ("queued", "running"):
        raise HTTPException(status_code=409, detail="이미 작업 중입니다.")
    analysis_jobs[name] = {"state": "queued", "progress": 0.0, "stage": "calibrating"}
    analysis_executor.submit(_run_auto_calibration, name)
    return _analysis_status(name)


@app.get("/analysis/{name}/stats")
def get_stats(name: str):
    """경기 지표 (선수별 거리·속력·스프린트, 점유율, 패스·턴오버, 팀 대형, 히트맵) — 정밀 모드 보정 후 생성"""
    from pipeline import video_analysis

    path = _analysis_file(name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="분석 결과가 없습니다.")
    stats = video_analysis.load(path).get("stats")
    if stats is None:
        raise HTTPException(status_code=404, detail="경기 지표가 없습니다. 경기장 보정을 먼저 저장하세요.")
    return stats


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

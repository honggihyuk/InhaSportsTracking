import { useState, useEffect } from 'react';

export const API_BASE_URL = 'http://localhost:8000';

export function useTrackingData(initialFrame = 0) {
  const [trackingData, setTrackingData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchTrackingData = async (frameNumber) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/get_tracking_data/${frameNumber}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setTrackingData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrackingData(initialFrame);
  }, []);

  return { trackingData, loading, error, refetch: fetchTrackingData };
}

export function useWebSocket() {
  const [ws, setWs] = useState(null);
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState(null);

  useEffect(() => {
    const websocket = new WebSocket(`ws://${API_BASE_URL.replace('http://', '')}/ws/stream`);
    
    websocket.onopen = () => {
      console.log('WebSocket 연결됨');
      setConnected(true);
      // 스트리밍 시작 요청
      websocket.send(JSON.stringify({ type: 'start', frame: 0 }));
    };

    websocket.onmessage = (event) => {
      const trackingData = JSON.parse(event.data);
      setData(trackingData);
    };

    websocket.onclose = () => {
      console.log('WebSocket 연결 종료');
      setConnected(false);
    };

    websocket.onerror = (error) => {
      console.error('WebSocket 오류:', error);
    };

    setWs(websocket);

    return () => {
      // 정리된 소켓의 이벤트가 새 소켓의 상태(connected/data)를 덮어쓰지 않도록 핸들러 해제
      websocket.onmessage = websocket.onerror = websocket.onclose = null;
      if (websocket.readyState === WebSocket.CONNECTING) {
        // 연결 중에 close() 하면 브라우저가 에러를 내므로 연결 직후 닫는다 (StrictMode 이중 마운트)
        websocket.onopen = () => websocket.close();
      } else {
        websocket.close();
      }
    };
  }, []);

  const sendCommand = (command) => {
    if (ws && connected) {
      ws.send(JSON.stringify(command));
    }
  };

  return { data, connected, sendCommand };
}

export async function uploadVideo(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload_video`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || '업로드 실패');
  }

  return await response.json();
}

export async function setViewState(viewState) {
  const response = await fetch(`${API_BASE_URL}/set_view_state`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(viewState),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || '시점 설정 실패');
  }

  return await response.json();
}

export async function listVideos() {
  const response = await fetch(`${API_BASE_URL}/videos`);
  if (!response.ok) {
    throw new Error(`영상 목록 조회 실패 (HTTP ${response.status})`);
  }
  const videos = await response.json();
  return videos.map((v) => ({ ...v, url: `${API_BASE_URL}${v.url}` }));
}

async function requestJson(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `요청 실패 (HTTP ${response.status})`);
  }
  return response.json();
}

const analysisPath = (name) => `/analysis/${encodeURIComponent(name)}`;

/** mode: 'precise'(고정밀, 기본) | 'realtime'(빠른 미리보기) */
const ANALYSIS_MODES = ['precise', 'realtime'];
export const startAnalysis = (name, mode) =>
  // 문자열 모드만 전달 (이벤트 객체 등이 넘어와도 서버 기본 모드로 시작)
  requestJson(`${analysisPath(name)}${ANALYSIS_MODES.includes(mode) ? `?mode=${mode}` : ''}`, { method: 'POST' });
export const cancelAnalysis = (name) => requestJson(analysisPath(name), { method: 'DELETE' });
export const getStats = (name) => requestJson(`${analysisPath(name)}/stats`);
/** 기준점 없이 경기장 라인·센터서클로 자동 보정 (백그라운드, /status 로 진행 확인) */
export const autoCalibrate = (name) =>
  requestJson(`${analysisPath(name)}/calibration/auto`, { method: 'POST' });
export const getAnalysisStatus = (name) => requestJson(`${analysisPath(name)}/status`);
export const getAnalysis = (name) => requestJson(analysisPath(name));
export const saveCalibration = (name, keyframes) =>
  requestJson(`${analysisPath(name)}/calibration`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ keyframes }),
  });

import { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000';

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

export function useWebSocket(frameRate = 30) {
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
      if (websocket) {
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

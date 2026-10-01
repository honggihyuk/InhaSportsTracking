// 영상 분석 결과(backend /analysis) 를 화면용 데이터로 바꾸는 순수 함수 모음
//
// 경기장 좌표: 센터 스폿 원점, x = 길이 방향(화면 오른쪽 +), y = 너비 방향(화면 먼 쪽 터치라인 +), 미터

export const BALL_ID = 0 // SoccerTracker.BALL_TRACK_ID
export const KIND_BALL = 1
const SPEED_WINDOW = 6 // 프레임, 속도 계산 간격 (30 FPS 기준 0.2 초)
const MAX_PLAYER_SPEED = 12 // m/s, 보정 오차로 튀는 값 표시 상한 (인간 최고 속력 ≈ 12.4 m/s)

const L = 52.5
const W = 34
const PB = 20.16 // 페널티 박스 반폭
const GA = 9.16 // 골 에어리어 반폭
const ARC = 7.31 // 페널티 아크와 박스 앞선의 교점 (스폿에서 9.15 m, 박스선까지 5.5 m)

// 보정용 경기장 기준점 (좌·우 대칭은 side 로 생성)
const sideLandmarks = (side, name) => {
  const s = side === 'left' ? -1 : 1
  return [
    { id: `${side}-corner-top`, label: `${name} 위 코너`, pitch: [s * L, W] },
    { id: `${side}-corner-bottom`, label: `${name} 아래 코너`, pitch: [s * L, -W] },
    { id: `${side}-pb-goal-top`, label: `${name} 페널티박스 · 골라인 위`, pitch: [s * L, PB] },
    { id: `${side}-pb-goal-bottom`, label: `${name} 페널티박스 · 골라인 아래`, pitch: [s * L, -PB] },
    { id: `${side}-pb-front-top`, label: `${name} 페널티박스 앞 위 모서리`, pitch: [s * (L - 16.5), PB] },
    { id: `${side}-pb-front-bottom`, label: `${name} 페널티박스 앞 아래 모서리`, pitch: [s * (L - 16.5), -PB] },
    { id: `${side}-arc-top`, label: `${name} 페널티 아크 · 박스선 위`, pitch: [s * (L - 16.5), ARC] },
    { id: `${side}-arc-bottom`, label: `${name} 페널티 아크 · 박스선 아래`, pitch: [s * (L - 16.5), -ARC] },
    { id: `${side}-ga-goal-top`, label: `${name} 골에어리어 · 골라인 위`, pitch: [s * L, GA] },
    { id: `${side}-ga-goal-bottom`, label: `${name} 골에어리어 · 골라인 아래`, pitch: [s * L, -GA] },
    { id: `${side}-ga-front-top`, label: `${name} 골에어리어 앞 위 모서리`, pitch: [s * (L - 5.5), GA] },
    { id: `${side}-ga-front-bottom`, label: `${name} 골에어리어 앞 아래 모서리`, pitch: [s * (L - 5.5), -GA] },
    { id: `${side}-spot`, label: `${name} 페널티 스폿`, pitch: [s * (L - 11), 0] },
  ]
}

export const LANDMARKS = [
  { id: 'center-spot', label: '센터 스폿', pitch: [0, 0] },
  { id: 'half-top', label: '하프라인 · 위쪽 터치라인', pitch: [0, W] },
  { id: 'half-bottom', label: '하프라인 · 아래쪽 터치라인', pitch: [0, -W] },
  { id: 'circle-top', label: '센터서클 · 하프라인 위', pitch: [0, 9.15] },
  { id: 'circle-bottom', label: '센터서클 · 하프라인 아래', pitch: [0, -9.15] },
  ...sideLandmarks('left', '왼쪽'),
  ...sideLandmarks('right', '오른쪽'),
]

/** 분석 결과의 프레임 f → 3D 트윈용 { players, ball } (보정 안 된 프레임이면 null) */
export function twinFrame(analysis, f) {
  const world = analysis?.world?.[f]
  if (!world) return null
  const prev = new Map((analysis.world[f - SPEED_WINDOW] ?? []).map(([id, x, y]) => [id, [x, y]]))
  const dt = SPEED_WINDOW / (analysis.fps || 30)
  const players = []
  let ball = null
  for (const [id, x, y, smoothed] of world) {
    // 정밀 모드는 칼만+RTS 평활 속력을 함께 저장 → 그대로 사용, 없으면 0.2 초 차분
    const p = prev.get(id)
    const speed = smoothed ?? (p ? Math.hypot(x - p[0], y - p[1]) / dt : 0)
    if (id === BALL_ID) {
      ball = { position_3d: [x, y, 0], velocity: speed }
    } else {
      players.push({
        id,
        team: analysis.teams?.[id] ?? 'other',
        position_3d: [x, y, 0],
        velocity: Math.min(speed, MAX_PLAYER_SPEED),
      })
    }
  }
  return { players, ball }
}

/** object-fit: contain 으로 표시된 영상의 화면 내 위치·배율 */
export function videoRect(containerW, containerH, videoW, videoH) {
  if (!videoW || !videoH) return null
  const scale = Math.min(containerW / videoW, containerH / videoH)
  return { x: (containerW - videoW * scale) / 2, y: (containerH - videoH * scale) / 2, scale }
}

/** 배경색 위 글자색 (밝은 배경 → 어두운 글자) */
export function inkOn(hex) {
  const n = parseInt(hex.slice(1), 16)
  const lum = (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255
  return lum > 0.6 ? '#0A0B0D' : '#F2F3F5'
}

/* ---------- 보정 확인용 경기장 라인 (pipeline/pitch.py 의 pitch_polylines 와 같은 모델) ---------- */
const arc = (cx, cy, r, a0, a1, n = 48) => Array.from({ length: n }, (_, i) => {
  const t = ((a0 + ((a1 - a0) * i) / (n - 1)) * Math.PI) / 180
  return [cx + r * Math.cos(t), cy + r * Math.sin(t)]
})

export const PITCH_POLYLINES = (() => {
  const lines = [
    [[-L, W], [L, W]], [[-L, -W], [L, -W]], [[-L, -W], [-L, W]], [[L, -W], [L, W]],
    [[0, -W], [0, W]], arc(0, 0, 9.15, 0, 360, 96),
  ]
  const half = (Math.acos((16.5 - 11) / 9.15) * 180) / Math.PI
  for (const s of [-1, 1]) {
    const gx = s * L, pb = s * (L - 16.5), ga = s * (L - 5.5)
    lines.push(
      [[gx, PB], [pb, PB], [pb, -PB], [gx, -PB]],
      [[gx, GA], [ga, GA], [ga, -GA], [gx, -GA]],
      arc(s * (L - 11), 0, 9.15, (s < 0 ? 0 : 180) - half, (s < 0 ? 0 : 180) + half, 32),
    )
  }
  // 직선은 원근 투영 후에도 직선이지만, 화면 밖으로 나가는 구간을 자연스럽게 자르도록 잘게 나눔
  return lines.map((pl) => pl.flatMap((p, i) => {
    if (i === 0) return [p]
    const q = pl[i - 1]
    const n = Math.max(1, Math.ceil(Math.hypot(p[0] - q[0], p[1] - q[1]) / 2))
    return Array.from({ length: n }, (_, k) => [q[0] + ((p[0] - q[0]) * (k + 1)) / n, q[1] + ((p[1] - q[1]) * (k + 1)) / n])
  }))
})()

export function invert3(m) {
  const [a, b, c, d, e, f, g, h, i] = m
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g
  const det = a * A + b * B + c * C
  if (Math.abs(det) < 1e-15) return null
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d]
    .map((v) => v / det)
}

/** 프레임 f 의 이미지→경기장 호모그래피로 경기장 라인을 원본 픽셀 좌표 폴리라인으로 (카메라 뒤쪽 점에서 끊음) */
export function projectedPitchLines(analysis, f) {
  const H = analysis?.homographies?.[f]
  const G = H && invert3(H)
  if (!G) return []
  const out = []
  for (const pl of PITCH_POLYLINES) {
    let cur = []
    for (const [x, y] of pl) {
      const w = G[6] * x + G[7] * y + G[8]
      if (w <= 1e-6) {
        if (cur.length > 1) out.push(cur)
        cur = []
        continue
      }
      cur.push([(G[0] * x + G[1] * y + G[2]) / w, (G[3] * x + G[4] * y + G[5]) / w])
    }
    if (cur.length > 1) out.push(cur)
  }
  return out
}

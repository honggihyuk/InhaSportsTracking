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
        role: analysis.roles?.[id] ?? null, // 'goalkeeper' | 'referee' | null
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
  return lum > 0.6 ? '#070908' : '#E6EAE7'
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

/* ---------- 경기 분석 화면 (AnalysisViews) 용 ---------- */
export const PITCH_L = L
export const PITCH_W = W
export const HEAT_CELL = 5 // m, analysis/stats.py 의 HEAT_CELL 과 같은 격자

const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
const linearToSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055)

function hexToOklab(hex) {
  const n = parseInt(hex.slice(1), 16)
  const [r, g, b] = [n >> 16, (n >> 8) & 255, n & 255].map((v) => srgbToLinear(v / 255))
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s]
}

function oklabToHex([L_, a, b]) {
  const l = (L_ + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L_ - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L_ - 0.0894841775 * a - 1.291485548 * b) ** 3
  const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s]
  return '#' + rgb.map((c) => Math.round(Math.min(1, Math.max(0, linearToSrgb(c))) * 255).toString(16).padStart(2, '0')).join('')
}

/**
 * 팀 색(유니폼에서 뽑은 색일 수 있음)을 어두운 잔디 위에서 보이도록 OKLab 밝기만 [0.56, 0.78] 로 옮긴다 (색상·채도 유지).
 * 아주 어두운 유니폼(예: 짙은 갈색)도 잔디(#13261A) 위 3:1 대비를 넘게 된다.
 */
export function vizTeamColor(hex) {
  if (!/^#[0-9a-f]{6}$/i.test(hex ?? '')) return hex
  const [l, a, b] = hexToOklab(hex)
  const lifted = Math.min(0.78, Math.max(0.56, l))
  return lifted === l ? hex : oklabToHex([lifted, a, b])
}

/** 두 색의 OKLab 거리 ×100 (팀 색이 너무 비슷한지 판단) */
export function colorDistance(h1, h2) {
  const p = hexToOklab(h1), q = hexToOklab(h2)
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) * 100
}

/**
 * world[from..to] 의 팀별 위치 밀도 — stats.py heatmaps() 와 같은 격자(행 = y 아래→위, 열 = x 왼쪽→오른쪽)
 * Returns: { home: { grid, total }, away: {...} } (grid 는 표본 수, 정규화 전)
 */
export function heatCounts(analysis, from = 0, to = Infinity) {
  const nx = Math.ceil((2 * L) / HEAT_CELL), ny = Math.ceil((2 * W) / HEAT_CELL)
  const out = {}
  for (const team of ['home', 'away']) out[team] = { grid: Array.from({ length: ny }, () => new Array(nx).fill(0)), total: 0 }
  const world = analysis?.world ?? []
  const last = Math.min(to, world.length - 1)
  for (let t = Math.max(0, from); t <= last; t++) {
    for (const [id, x, y] of world[t] ?? []) {
      const g = out[analysis.teams?.[id]]
      if (!g) continue
      const i = Math.min(ny - 1, Math.max(0, Math.floor((y + W) / HEAT_CELL)))
      const j = Math.min(nx - 1, Math.max(0, Math.floor((x + L) / HEAT_CELL)))
      g.grid[i][j] += 1
      g.total += 1
    }
  }
  return out
}

/** 격자를 공격 방향 기준 3 × 3 구역(수비·중원·공격 × 왼쪽·가운데·오른쪽) 점유 비율로 — 히트맵의 표 보기 */
export function zoneShares(grid, defends = -1) {
  const ny = grid.length, nx = grid[0]?.length ?? 0
  const zones = Array.from({ length: 3 }, () => [0, 0, 0])
  let total = 0
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      const v = grid[i][j]
      if (!v) continue
      const cx = -L + (j + 0.5) * HEAT_CELL, cy = -W + (i + 0.5) * HEAT_CELL
      const ax = defends > 0 ? -cx : cx // 공격 방향을 +x 로
      const ay = defends > 0 ? -cy : cy // 공격 방향 기준 왼쪽이 +y
      const third = ax < -L / 3 ? 0 : ax < L / 3 ? 1 : 2
      const lane = ay > W / 3 ? 0 : ay > -W / 3 ? 1 : 2
      zones[third][lane] += v
      total += v
    }
  }
  return zones.map((row) => row.map((v) => (total ? v / total : 0)))
}

/**
 * 팀 패스 네트워크 — analysis/stats.py pass_network() 와 같은 규칙.
 * 저장된 결과에 stats.pass_network 가 있고 전체 구간이면 그대로 쓰고, 아니면(이전 결과 · 구간 필터) 이벤트에서 다시 만든다.
 */
export function passNetwork(analysis, stats, team, upto = Infinity) {
  const saved = stats?.pass_network?.[team]
  const passes = (stats?.events ?? []).filter((e) => e.type === 'pass' && e.team === team && e.end_frame <= upto)
  if (saved && passes.length === saved.passes) return saved
  const world = analysis?.world ?? []
  const touches = new Map(), counts = new Map(), edges = new Map()
  const posAt = (t, id) => (world[t] ?? []).find((e) => e[0] === id)
  for (const e of passes) {
    for (const [id, t, key] of [[e.from, e.frame, 'passes'], [e.to, e.end_frame, 'received']]) {
      const c = counts.get(id) ?? { passes: 0, received: 0 }
      c[key] += 1
      counts.set(id, c)
      const p = posAt(t, id)
      if (p) touches.set(id, [...(touches.get(id) ?? []), [p[1], p[2]]])
    }
    const a = Math.min(e.from, e.to), b = Math.max(e.from, e.to)
    const k = `${a}-${b}`
    const edge = edges.get(k) ?? { a, b, count: 0, a_to_b: 0, b_to_a: 0 }
    edge.count += 1
    edge[e.from === a ? 'a_to_b' : 'b_to_a'] += 1
    edges.set(k, edge)
  }
  const nodes = []
  for (const [id, c] of [...counts].sort((p, q) => p[0] - q[0])) {
    const info = stats?.players?.[id] ?? {}
    const pts = touches.get(id)
    const position = pts
      ? [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length]
      : info.mean_position
    if (!position) continue
    nodes.push({ id, jersey_number: info.jersey_number ?? null, role: info.role ?? 'player', position, ...c })
  }
  const kept = new Set(nodes.map((n) => n.id))
  return {
    nodes,
    edges: [...edges.values()].filter((e) => kept.has(e.a) && kept.has(e.b)),
    passes: passes.length,
  }
}

/** 볼록 껍질 (Andrew monotone chain), 점 [[x, y], ...] → 반시계 순서 꼭짓점 */
export function convexHull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  if (p.length < 3) return p
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  const half = (pts) => {
    const h = []
    for (const q of pts) {
      while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], q) <= 0) h.pop()
      h.push(q)
    }
    h.pop()
    return h
  }
  return [...half(p), ...half([...p].reverse())]
}

/**
 * 프레임 f 의 팀 대형 — stats.py team_shape() 와 같은 정의(골키퍼 제외 필드 선수):
 * 무게중심, 폭(y 범위), 길이(x 범위), 수비 라인(지키는 골문 쪽 최후방 x), 그리고 화면용 볼록 껍질
 */
export function shapeAt(analysis, f, team, defends = -1) {
  const pts = (analysis?.world?.[f] ?? [])
    .filter(([id]) => analysis.teams?.[id] === team && analysis.roles?.[id] !== 'goalkeeper')
    .map(([, x, y]) => [x, y])
  if (pts.length < 2) return null
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
  return {
    points: pts,
    hull: convexHull(pts),
    centroid: [xs.reduce((a, b) => a + b, 0) / pts.length, ys.reduce((a, b) => a + b, 0) / pts.length],
    width: Math.max(...ys) - Math.min(...ys),
    length: Math.max(...xs) - Math.min(...xs),
    defensive_line: defends < 0 ? Math.min(...xs) : Math.max(...xs),
  }
}

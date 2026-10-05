import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Line } from '@react-three/drei'
import * as THREE from 'three'
import {
  useWebSocket, uploadVideo, listVideos, startAnalysis, cancelAnalysis, getAnalysisStatus, getAnalysis,
  saveCalibration, autoCalibrate,
} from './components/api'
import { VideoOverlay } from './components/VideoOverlay'
import { CalibrationPanel } from './components/CalibrationPanel'
import { twinFrame, LANDMARKS, BALL_ID } from './analysis'
import './index.css'

const FPS = 30
const FIELD_LENGTH = 105
const FIELD_WIDTH = 68
const DEFAULT_TIMELINE = 100 // s, 영상이 없을 때 타임라인 길이
const POSSESSION_RADIUS = 3 // m, 공과 이 거리 이내의 가장 가까운 선수를 소유자로 간주
const DRIFT_FRAMES = 15 // 영상과 트래킹 스트림이 이만큼 어긋나면 재동기화

const COLORS = {
  home: '#F2555A',
  away: '#4C8DFF',
  line: '#E9EEF2',
  turfA: '#1C3B28',
  turfB: '#19351F',
  background: '#0A0B0D',
}

// 백엔드 경기장 좌표 [x(길이), y(너비), z(높이)] → three.js 월드 좌표 [x, 높이, z]
// (FootballField 가 X 축으로 -90° 회전되어 있어 경기장 y 는 월드 -z 방향)
const toWorld = ([x, y, z = 0]) => [x, z, -y]

const formatTime = (s) => {
  if (!Number.isFinite(s)) s = 0
  const m = Math.floor(s / 60)
  return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(1).padStart(4, '0')}`
}
const formatSize = (bytes) =>
  bytes > 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${(bytes / 1e6).toFixed(1)} MB`

/* ---------- 아이콘 (인라인 SVG, stroke 기반) ---------- */
const ICONS = {
  play: <path d="M7 4.5v15l12-7.5z" fill="currentColor" stroke="none" />,
  pause: <><rect x="6" y="4.5" width="4" height="15" rx="1" fill="currentColor" stroke="none" /><rect x="14" y="4.5" width="4" height="15" rx="1" fill="currentColor" stroke="none" /></>,
  restart: <><path d="M3 12a9 9 0 1 0 3-6.7" /><path d="M3 4v5h5" /></>,
  upload: <><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></>,
  film: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4" /></>,
  cube: <><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z" /><path d="m4 7.5 8 4.5 8-4.5M12 12v9" /></>,
  split: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M12 4v16" /></>,
  alert: <><circle cx="12" cy="12" r="9" /><path d="M12 7.5v5.5M12 16.5h.01" /></>,
  scan: <><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /><rect x="8" y="8" width="8" height="8" rx="1" /></>,
  target: <><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="2" /><path d="M12 2v4M12 18v4M2 12h4M18 12h4" /></>,
}
function Icon({ name, size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICONS[name]}
    </svg>
  )
}

function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="tablist" aria-label={label}>
      {options.map(([key, text, icon]) => (
        <button key={key} role="tab" aria-selected={value === key}
          className={value === key ? 'active' : ''} onClick={() => onChange(key)}>
          {icon && <Icon name={icon} size={15} />}
          <span>{text}</span>
        </button>
      ))}
    </div>
  )
}

/* ---------- 3D 씬 ---------- */
const L = FIELD_LENGTH / 2
const W = FIELD_WIDTH / 2
const rect = (x1, y1, x2, y2) => [[x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0], [x1, y1, 0]]
const FIELD_LINES = [
  rect(-L, -W, L, W),                   // 터치라인 / 골라인
  [[0, -W, 0], [0, W, 0]],              // 중앙선
  rect(-L, -20.16, -L + 16.5, 20.16),   // 페널티 박스 (골라인에서 16.5m, 폭 40.32m)
  rect(L - 16.5, -20.16, L, 20.16),
  rect(-L, -9.16, -L + 5.5, 9.16),      // 골 에어리어 (5.5m, 폭 18.32m)
  rect(L - 5.5, -9.16, L, 9.16),
]
const STRIPES = 14 // 잔디 깎은 줄무늬

function FootballField() {
  const stripeWidth = FIELD_LENGTH / STRIPES
  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {/* 경기장 바깥 여백 */}
      <mesh position={[0, 0, -0.01]}>
        <planeGeometry args={[FIELD_LENGTH + 12, FIELD_WIDTH + 10]} />
        <meshStandardMaterial color="#132519" roughness={1} />
      </mesh>
      {Array.from({ length: STRIPES }, (_, i) => (
        <mesh key={i} position={[-L + stripeWidth * (i + 0.5), 0, 0]}>
          <planeGeometry args={[stripeWidth, FIELD_WIDTH]} />
          <meshStandardMaterial color={i % 2 ? COLORS.turfA : COLORS.turfB} roughness={0.95} />
        </mesh>
      ))}
      {/* 라인은 잔디 위로 살짝 띄워 z-fighting 방지 */}
      <group position={[0, 0, 0.02]}>
        {FIELD_LINES.map((points, i) => (
          <Line key={i} points={points} color={COLORS.line} lineWidth={1.6} transparent opacity={0.9} />
        ))}
        {/* 센터 서클: 그룹이 이미 바닥에 눕혀져 있으므로 추가 회전 없음 */}
        <mesh>
          <ringGeometry args={[9.08, 9.22, 96]} />
          <meshBasicMaterial color={COLORS.line} transparent opacity={0.9} side={THREE.DoubleSide} />
        </mesh>
        <mesh>
          <circleGeometry args={[0.25, 24]} />
          <meshBasicMaterial color={COLORS.line} />
        </mesh>
      </group>
    </group>
  )
}

// 선수 번호 라벨: 브라우저 canvas 로 그린 텍스처 (drei <Text> 는 글꼴을 외부 CDN 에서 받아
// 오프라인·CDN 차단 환경에서 3D 화면과 UI 갱신이 멈춤 → 외부 요청이 없는 방식으로 교체)
const labelCache = new Map()
function labelTexture(text) {
  let tex = labelCache.get(text)
  if (tex) return tex
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 64
  const draw = () => {
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.font = '700 40px "Pretendard Variable", Pretendard, system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.lineWidth = 6
    ctx.strokeStyle = '#000000'
    ctx.strokeText(text, 64, 34)
    ctx.fillStyle = '#F2F3F5'
    ctx.fillText(text, 64, 34)
  }
  draw()
  tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  // 서체가 아직 로드 전이면 로드 후 다시 그림
  document.fonts?.ready.then(() => { draw(); tex.needsUpdate = true })
  labelCache.set(text, tex)
  return tex
}

function PlayerMarker({ player, color }) {
  return (
    <group position={toWorld(player.position_3d)}>
      {/* 바닥 그림자 링 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.03, 0]}>
        <ringGeometry args={[0.75, 1.05, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.35} />
      </mesh>
      <mesh position={[0, 0.9, 0]}>
        <capsuleGeometry args={[0.45, 0.9, 8, 16]} />
        <meshStandardMaterial color={color} roughness={0.35} metalness={0.1} />
      </mesh>
      <sprite position={[0, 2.35, 0]} scale={[1.7, 0.85, 1]}>
        <spriteMaterial map={labelTexture(String(player.id))} transparent depthWrite={false} />
      </sprite>
    </group>
  )
}

function BallMarker({ ball }) {
  const [x, y, z] = toWorld(ball.position_3d)
  return (
    <group position={[x, y + 0.35, z]}>
      <mesh>
        <sphereGeometry args={[0.35, 24, 24]} />
        <meshStandardMaterial color="#FFFFFF" emissive="#D7FF3C" emissiveIntensity={0.35} />
      </mesh>
      <pointLight color="#D7FF3C" intensity={2} distance={6} decay={2} />
    </group>
  )
}

const CAMERA_PRESETS = {
  '3d': [0, 55, 85],
  topdown: [0, 115, 0.01], // 정확히 수직이면 OrbitControls 가 불안정하여 살짝 기울임
}

// 뷰 모드에 따라 카메라 이동. 선수 시점은 매 프레임 선수 위치에서 공을 바라봄
function CameraRig({ cameraMode, player, ball }) {
  const { camera, controls, size, invalidate } = useThree()
  const aspect = size.width / size.height

  useEffect(() => {
    const preset = CAMERA_PRESETS[cameraMode]
    if (!preset) return
    // 세로로 긴 화면(분할 모드 등)에서도 경기장 전체가 보이도록 거리 보정
    const fit = Math.max(1.3, 1.4 / aspect)
    camera.position.set(...preset.map((v) => v * fit))
    controls?.target.set(0, 0, 0)
    controls?.update()
    invalidate() // frameloop="demand" 이므로 직접 다시 그리기 요청
  }, [cameraMode, camera, controls, aspect, invalidate])

  useFrame(() => {
    if (cameraMode !== 'player' || !player) return
    const [px, , pz] = toWorld(player.position_3d)
    camera.position.set(px, 2.5, pz)
    if (ball) {
      const [bx, by, bz] = toWorld(ball.position_3d)
      camera.lookAt(bx, by + 0.35, bz)
    }
  })

  return null
}

function Scene({ players, ball, cameraMode, teamColors }) {
  const followed = players[0] // 선수 시점: 첫 번째 선수
  return (
    <>
      <color attach="background" args={[COLORS.background]} />
      <hemisphereLight args={['#dfe8ff', '#0b140e', 0.9]} />
      <directionalLight position={[40, 80, 30]} intensity={1.4} />
      <FootballField />
      {players
        .filter((p) => !(cameraMode === 'player' && p === followed)) // 시점 선수 자신은 숨김
        .map((p) => <PlayerMarker key={p.id} player={p} color={teamColors[p.team] ?? teamColors.other} />)}
      {ball && <BallMarker ball={ball} />}
      <OrbitControls makeDefault enabled={cameraMode !== 'player'} enableDamping
        maxPolarAngle={Math.PI / 2.15} minDistance={20} maxDistance={260} />
      <CameraRig cameraMode={cameraMode} player={followed} ball={ball} />
    </>
  )
}

/* ---------- 통계 ---------- */
function possessionTeam(players, ball) {
  if (!ball) return null
  const [bx, by] = ball.position_3d
  let best = null
  let bestDist = POSSESSION_RADIUS
  for (const p of players) {
    const d = Math.hypot(p.position_3d[0] - bx, p.position_3d[1] - by)
    if (d <= bestDist) [best, bestDist] = [p, d]
  }
  return best?.team ?? null
}

function teamStats(players, team) {
  const list = players.filter((p) => p.team === team)
  const avg = list.length ? list.reduce((s, p) => s + p.velocity, 0) / list.length : 0
  const max = list.length ? Math.max(...list.map((p) => p.velocity)) : 0
  return { count: list.length, avg, max }
}

function CompareRow({ label, home, away, format = (v) => v }) {
  const total = home + away
  const homeShare = total > 0 ? (home / total) * 100 : 50
  return (
    <div className="compare-row">
      <div className="compare-values">
        <span className="num">{format(home)}</span>
        <span className="compare-label">{label}</span>
        <span className="num">{format(away)}</span>
      </div>
      <div className="compare-bar">
        <span className="home" style={{ width: `${homeShare}%` }} />
        <span className="away" style={{ width: `${100 - homeShare}%` }} />
      </div>
    </div>
  )
}

/* ---------- 영상 분석 카드 ---------- */
const MODES = [
  ['precise', '고정밀', '매 프레임 탐지 · 공 타일 탐지 · 트랙 잇기 · 경기장 라인 정렬 보정 · 좌표 평활화 · 경기 지표. CPU 에서는 프레임당 수 초가 걸립니다.'],
  ['realtime', '빠른 미리보기', '3 프레임마다 탐지하고 사이를 보간합니다. 후처리·라인 정렬·경기 지표는 없습니다.'],
]
const STATE_LABEL = { none: '분석 전', queued: '대기 중', done: '완료', error: '오류', cancelled: '취소됨' }
const STAGE_LABEL = { analyzing: '분석 중', lines: '라인 검출 중', calibrating: '자동 보정 중' }

function AnalysisCard({ status, summary, mode, onMode, showBoxes, onToggleBoxes, showLines, onToggleLines,
  onStart, onCancel, onCalibrate, onAutoCalibrate }) {
  const state = status?.state ?? 'none'
  const pct = Math.round((status?.progress ?? 0) * 100)
  const doneMode = status?.mode ?? summary?.mode
  return (
    <section className="card analysis-card">
      <div className="card-head">
        <div className="eyebrow">영상 분석</div>
        <span className={`tag tag-${state}`}>
          {state === 'running' ? `${STAGE_LABEL[status?.stage] ?? '분석 중'} ${pct}%` : STATE_LABEL[state]}
          {state === 'done' && doneMode && ` · ${doneMode === 'precise' ? '고정밀' : '미리보기'}`}
        </span>
      </div>

      {(state === 'none' || state === 'error' || state === 'cancelled') && (
        <>
          <Segmented label="분석 모드" value={mode} onChange={onMode}
            options={MODES.map(([key, text]) => [key, text])} />
          <p className="card-text">{state === 'error' ? status.error : MODES.find(([k]) => k === mode)[2]}</p>
          {/* onClick 에 onStart 를 그대로 넘기면 클릭 이벤트가 분석 모드 인자로 전달됨 (?mode=[object Object] → 400) */}
          <button className="btn btn-primary btn-block" onClick={() => onStart()}>
            <Icon name="scan" size={16} />{state === 'none' ? '분석 시작' : '다시 분석'}
          </button>
        </>
      )}

      {(state === 'queued' || state === 'running') && (
        <>
          <div className="progress"><span style={{ width: `${pct}%` }} /></div>
          <p className="card-text muted">
            {status?.stage === 'calibrating'
              ? '경기장 라인과 센터서클로 기준점 없이 보정하고 있습니다. '
              : status?.mode === 'precise' ? '고정밀 분석 중입니다. 끝나면 경기장을 자동으로 보정합니다. ' : ''}
            다른 화면을 봐도 계속 진행됩니다.
          </p>
          <button className="btn btn-ghost btn-block" onClick={() => onCancel()}>취소</button>
        </>
      )}

      {state === 'done' && summary && (
        <>
          <div className="kv"><span>추적된 선수</span><strong className="num">{summary.players}<small> 트랙</small></strong></div>
          <div className="kv"><span>팀 분류</span><strong className="num">
            {summary.home} · {summary.away}<small> (기타 {summary.other})</small></strong></div>
          <div className="kv"><span>공 검출</span><strong className="num">{summary.ballPct}<small> % 프레임</small></strong></div>
          {summary.post && (
            <div className="kv"><span>후처리</span><strong className="num">
              잇기 {summary.post.stitched}<small> · 보간 {summary.post.filled_boxes + summary.post.filled_ball}</small></strong></div>
          )}
          <div className="kv"><span>경기장 보정</span><strong className={summary.calibratedPct ? '' : 'warn'}>
            {summary.calibratedPct ? <span className="num">{summary.calibratedPct}<small> % 프레임</small></span> : '필요'}</strong></div>
          {summary.calibratedPct > 0 && (
            <div className="kv"><span>키프레임</span><strong className="num">
              자동 {summary.autoKeyframes}<small> · 수동 {summary.manualKeyframes}</small></strong></div>
          )}
          {summary.refinedPct != null && (
            <div className="kv"><span>라인 정렬</span><strong className="num">{summary.refinedPct}<small> % 프레임</small></strong></div>
          )}
          {status?.auto_calibration?.status === 'failed' && !summary.calibratedPct && (
            <p className="card-text warn-text">{status.auto_calibration.reason}</p>
          )}
          <label className="switch">
            <input type="checkbox" checked={showBoxes} onChange={onToggleBoxes} />
            <span>영상에 탐지 박스 표시</span>
          </label>
          {summary.calibratedPct > 0 && (
            <label className="switch">
              <input type="checkbox" checked={showLines} onChange={onToggleLines} />
              <span>보정된 경기장 라인 표시</span>
            </label>
          )}
          {summary.calibratedPct < 100 && (
            <button className={`btn btn-block ${summary.calibratedPct ? 'btn-ghost' : 'btn-primary'}`} onClick={() => onAutoCalibrate()}>
              <Icon name="scan" size={15} />{summary.calibratedPct ? '자동 보정 다시 실행' : '자동 보정'}
            </button>
          )}
          <button className="btn btn-ghost btn-block" onClick={() => onCalibrate()}>
            <Icon name="target" size={15} />{summary.calibratedPct ? '현재 프레임 보정 추가' : '기준점 직접 지정'}
          </button>
          {doneMode !== 'precise' && (
            <button className="btn btn-ghost btn-block" onClick={() => onStart('precise')}>
              <Icon name="scan" size={15} />고정밀 모드로 재분석
            </button>
          )}
        </>
      )}
    </section>
  )
}

/* ---------- 경기 지표 카드 (고정밀 모드 + 보정 후) ---------- */
const EVENT_LABEL = { pass: '패스', turnover: '턴오버', sprint: '스프린트' }

function MatchStatsCard({ stats, fps, teamColors, onSeek, currentFrame }) {
  const [tab, setTab] = useState('players')
  const players = Object.entries(stats.players)
    .filter(([, p]) => p.team !== 'other' && p.seconds >= 1)
    .sort((a, b) => b[1].distance - a[1].distance)
  const { home, away } = stats.teams
  const t = (f) => formatTime(f / fps)
  return (
    <section className="card stats-card" style={{ '--home': teamColors.home, '--away': teamColors.away }}>
      <div className="card-head">
        <div className="eyebrow">경기 지표</div>
        <span className="tag tag-done">고정밀</span>
      </div>
      <CompareRow label="점유율 %" home={home.possession * 100} away={away.possession * 100} format={(v) => v.toFixed(0)} />
      <CompareRow label="이동 거리 m" home={home.distance} away={away.distance} format={(v) => v.toFixed(0)} />
      <CompareRow label="스프린트" home={home.sprints} away={away.sprints} />
      <CompareRow label="패스" home={home.passes} away={away.passes} />
      <Segmented label="지표 보기" value={tab} onChange={setTab} options={[['players', '선수'], ['events', '이벤트']]} />
      {tab === 'players' ? (
        <table className="stats-table num">
          <thead><tr><th>ID</th><th>거리 m</th><th>최고 m/s</th><th>스프린트</th></tr></thead>
          <tbody>
            {players.slice(0, 12).map(([id, p]) => (
              <tr key={id}>
                <td><span className="team-dot" style={{ background: teamColors[p.team] }} />#{id}
                  {p.role === 'goalkeeper' && <small> GK</small>}</td>
                <td>{p.distance.toFixed(0)}</td><td>{p.max_speed.toFixed(1)}</td><td>{p.sprints.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <ul className="event-list">
          {stats.events.length === 0 && <li className="muted small">감지된 이벤트가 없습니다.</li>}
          {stats.events.slice(0, 60).map((e, i) => (
            <li key={i}>
              <button className={currentFrame >= e.frame && currentFrame <= (e.end_frame ?? e.frame) ? 'active' : ''}
                onClick={() => onSeek(e.frame / fps)}>
                <span className="num muted">{t(e.frame)}</span>
                <span className="team-dot" style={{ background: teamColors[e.team] ?? teamColors.other }} />
                <span>{EVENT_LABEL[e.type]}</span>
                <span className="num muted">
                  {e.type === 'sprint' ? `#${e.player} ${e.max_speed.toFixed(1)} m/s` : `#${e.from} → #${e.to}`}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

function summarize(data) {
  const players = new Set()
  let ballFrames = 0
  for (const objs of data.frames) {
    if (objs.some((o) => o[0] === BALL_ID)) ballFrames++
    for (const o of objs) if (o[0] !== BALL_ID) players.add(o[0])
  }
  const teams = Object.values(data.teams ?? {})
  const count = (t) => teams.filter((x) => x === t).length
  const n = Math.max(data.frames.length, 1)
  return {
    players: players.size,
    home: count('home'),
    away: count('away'),
    other: count('other'),
    ballPct: Math.round((ballFrames / n) * 100),
    calibratedPct: data.world ? Math.round((data.world.filter(Boolean).length / n) * 100) : 0,
    refinedPct: data.calibration?.refined_frames != null
      ? Math.round((data.calibration.refined_frames / n) * 100) : null,
    mode: data.mode ?? 'realtime',
    post: data.postprocess ?? null,
    autoKeyframes: (data.calibration?.keyframes ?? []).filter((k) => k.source === 'auto').length,
    manualKeyframes: (data.calibration?.keyframes ?? []).filter((k) => k.source !== 'auto').length,
  }
}

const DEFAULT_TEAM_COLORS = { home: COLORS.home, away: COLORS.away, other: '#9AA3AD' }

/* ---------- 앱 ---------- */
function App() {
  const { data, connected, sendCommand } = useWebSocket()
  const [layout, setLayout] = useState('split')
  const [cameraMode, setCameraMode] = useState('3d')
  const [streamPlaying, setStreamPlaying] = useState(true) // 영상이 없을 때 재생 상태
  const [videos, setVideos] = useState([])
  const [source, setSource] = useState(null) // 선택된 경기 영상 {name, url, size}
  const [video, setVideo] = useState({ ready: false, playing: false, duration: 0, error: null })
  const [videoFrame, setVideoFrame] = useState(0) // 화면에 표시 중인 영상 프레임 번호
  const [analysis, setAnalysis] = useState({ status: null, data: null })
  const [showBoxes, setShowBoxes] = useState(true)
  const [showLines, setShowLines] = useState(false)
  const [runMode, setRunMode] = useState('precise') // 분석 시작 시 선택한 모드
  const [calib, setCalib] = useState(null) // 보정 모드 {frame, points, activeId, saving, error}
  const [upload, setUpload] = useState(null)
  const videoRef = useRef(null)
  const frameRef = useRef(0)

  const streamFrame = data?.frame ?? 0
  useEffect(() => { frameRef.current = streamFrame }, [streamFrame])

  // 영상이 재생 가능하면 영상이 기준 시계, 아니면 트래킹 스트림이 기준
  const videoMaster = Boolean(source && video.ready && !video.error)
  const fps = analysis.data?.fps || FPS
  const playing = videoMaster ? video.playing : streamPlaying
  const duration = videoMaster ? video.duration : DEFAULT_TIMELINE
  const currentFrame = videoMaster ? videoFrame : streamFrame
  const currentTime = currentFrame / (videoMaster ? fps : FPS)

  // 3D 데이터: 분석된 영상이면 분석 결과(경기장 보정 필요), 아니면 더미 스트림
  const analysisMode = Boolean(videoMaster && analysis.data)
  const calibrated = Boolean(analysis.data?.world)
  const twin = analysisMode ? twinFrame(analysis.data, videoFrame) : null
  const players = analysisMode ? (twin?.players ?? []) : (data?.players ?? [])
  const ball = analysisMode ? (twin?.ball ?? null) : (data?.ball ?? null)
  const teamColors = analysisMode && analysis.data.team_colors?.home
    ? { ...DEFAULT_TEAM_COLORS, ...analysis.data.team_colors }
    : DEFAULT_TEAM_COLORS
  const summary = useMemo(() => (analysis.data ? summarize(analysis.data) : null), [analysis.data])

  const home = teamStats(players, 'home')
  const away = teamStats(players, 'away')
  const owner = possessionTeam(players, ball)

  const refreshVideos = useCallback(async () => {
    try {
      const list = await listVideos()
      setVideos(list)
      return list
    } catch {
      return []
    }
  }, [])

  useEffect(() => {
    listVideos().then(setVideos).catch(() => {})
  }, [])

  // 선택한 영상의 분석 상태·결과 불러오기, 분석 중이면 완료될 때까지 폴링
  const sourceName = source?.name
  const analysisState = analysis.status?.state
  useEffect(() => {
    if (!sourceName) return
    let cancelled = false
    const load = async () => {
      try {
        const status = await getAnalysisStatus(sourceName)
        const result = status.state === 'done' ? await getAnalysis(sourceName) : null
        if (!cancelled) setAnalysis({ status, data: result })
      } catch {
        if (!cancelled) setAnalysis({ status: null, data: null })
      }
    }
    load()
    return () => { cancelled = true }
  }, [sourceName])

  useEffect(() => {
    if (!sourceName || (analysisState !== 'queued' && analysisState !== 'running')) return
    const timer = setInterval(async () => {
      try {
        const status = await getAnalysisStatus(sourceName)
        const result = status.state === 'done' ? await getAnalysis(sourceName) : null
        setAnalysis((a) => ({ status, data: result ?? a.data }))
      } catch { /* 다음 주기에 재시도 */ }
    }, 1500)
    return () => clearInterval(timer)
  }, [sourceName, analysisState])

  // 표시 중인 영상 프레임을 프레임 단위로 추적 (박스 오버레이·3D 가 영상과 정확히 맞도록)
  useEffect(() => {
    const v = videoRef.current
    if (!v || !video.ready) return
    let handle
    const onFrame = (_, meta) => {
      setVideoFrame(Math.round(meta.mediaTime * fps))
      handle = v.requestVideoFrameCallback(onFrame)
    }
    const onSeeked = () => setVideoFrame(Math.round(v.currentTime * fps))
    const onTime = () => { if (!v.requestVideoFrameCallback) onSeeked() } // 미지원 브라우저 대체
    if (v.requestVideoFrameCallback) handle = v.requestVideoFrameCallback(onFrame)
    v.addEventListener('seeked', onSeeked)
    v.addEventListener('timeupdate', onTime)
    onSeeked()
    return () => {
      if (handle !== undefined) v.cancelVideoFrameCallback(handle)
      v.removeEventListener('seeked', onSeeked)
      v.removeEventListener('timeupdate', onTime)
    }
  }, [video.ready, sourceName, fps])

  // 트래킹 스트림을 영상 시각에 맞춤 (분석 결과가 없는 영상에서 더미 3D 동기화용)
  const syncStream = useCallback((paused) => {
    const v = videoRef.current
    if (v) sendCommand({ type: 'seek', frame: Math.round(v.currentTime * FPS), paused })
  }, [sendCommand])

  const selectSource = (item) => {
    setSource(item)
    setVideo({ ready: false, playing: false, duration: 0, error: null })
    setVideoFrame(0)
    setAnalysis({ status: null, data: null })
    setCalib(null)
    // 새 영상은 처음·정지 상태에서 시작 (재생 불가 코덱이면 이 상태로 스트림이 기준이 됨)
    sendCommand({ type: 'seek', frame: 0, paused: true })
    setStreamPlaying(false)
    if (layout === 'twin') setLayout('split')
  }

  const markReady = (v) => {
    // updater 는 나중에 실행되므로 영상 값은 미리 꺼내 둔다 (그 시점엔 이벤트 currentTarget 이 null)
    const { duration } = v
    setVideo((s) => (s.ready ? s : { ...s, ready: true, duration }))
  }

  const videoHandlers = {
    onLoadedMetadata: (e) => { markReady(e.currentTarget); syncStream(true) },
    // 일부 브라우저·캐시된 영상은 loadedmetadata 를 놓칠 수 있어, 첫 프레임 로드 시에도 준비 상태로 표시
    onLoadedData: (e) => markReady(e.currentTarget),
    onPlay: () => { setVideo((s) => ({ ...s, playing: true })); syncStream(false) },
    onPause: () => { setVideo((s) => ({ ...s, playing: false })); syncStream(true) },
    onSeeked: (e) => syncStream(e.currentTarget.paused),
    onTimeUpdate: (e) => {
      // 스트림과 영상이 벌어지면 재동기화 (네트워크 지연·탭 비활성 등)
      const v = e.currentTarget
      if (!v.paused && Math.abs(frameRef.current - v.currentTime * FPS) > DRIFT_FRAMES) syncStream(false)
    },
    onError: (e) => {
      const code = e.currentTarget.error?.code
      setVideo((s) => ({
        ...s,
        playing: false,
        error: code === 4
          ? '이 브라우저에서 재생할 수 없는 코덱입니다. H.264(MP4) · VP9/AV1(WebM) 형식으로 변환해 주세요.'
          : '영상을 불러오지 못했습니다.',
      }))
    },
  }

  const togglePlay = () => {
    if (videoMaster) {
      const v = videoRef.current
      if (v.paused) v.play().catch(() => {}) // 재생 실패는 onError 에서 안내
      else v.pause()
      return
    }
    sendCommand({ type: 'seek', frame: streamFrame, paused: streamPlaying })
    setStreamPlaying(!streamPlaying)
  }

  const seekTo = (seconds) => {
    if (videoMaster) {
      videoRef.current.currentTime = seconds // onSeeked 에서 스트림 동기화
      return
    }
    sendCommand({ type: 'seek', frame: Math.round(seconds * FPS), paused: !streamPlaying })
  }

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    e.target.value = '' // 같은 파일 재선택 허용
    if (!file) return
    setUpload({ status: 'uploading', message: `${file.name} 업로드 중` })
    try {
      const res = await uploadVideo(file)
      setUpload({ status: 'done', message: `${res.frame_count.toLocaleString()} 프레임 · ${formatTime(res.duration)}` })
      const list = await refreshVideos()
      const uploaded = list.find((v) => v.name === file.name)
      if (uploaded) selectSource(uploaded)
    } catch (err) {
      setUpload({ status: 'error', message: err.message })
    }
  }

  const handleStartAnalysis = async (mode = runMode) => {
    try {
      const status = await startAnalysis(source.name, mode)
      setAnalysis((a) => ({ ...a, status }))
    } catch (err) {
      setAnalysis((a) => ({ ...a, status: { state: 'error', error: err.message } }))
    }
  }

  const handleCancelAnalysis = async () => {
    try {
      await cancelAnalysis(source.name)
      setAnalysis((a) => ({ ...a, status: { ...a.status, state: 'cancelled' } }))
    } catch { /* 이미 끝난 작업 — 다음 폴링에서 상태 갱신 */ }
  }

  const handleAutoCalibrate = async () => {
    try {
      const status = await autoCalibrate(source.name)
      setAnalysis((a) => ({ ...a, status })) // 진행 중 → 폴링이 끝나면 결과를 다시 불러옴
    } catch (err) {
      setAnalysis((a) => ({ ...a, status: { ...a.status, auto_calibration: { status: 'failed', reason: err.message } } }))
    }
  }

  /* ---- 경기장 보정 ---- */
  const startCalibration = () => {
    videoRef.current?.pause()
    if (layout === 'twin') setLayout('split')
    setCalib({ frame: videoFrame, points: [], activeId: null, saving: false, error: null })
  }

  const pickPoint = (image) => setCalib((c) => {
    if (!c.activeId) return { ...c, error: '먼저 도면에서 기준점을 선택하세요.' }
    const landmark = LANDMARKS.find((l) => l.id === c.activeId)
    const points = [...c.points.filter((p) => p.landmark.id !== landmark.id), { landmark, image }]
    return { ...c, points, activeId: null, error: null }
  })

  const saveCalib = async () => {
    const keyframe = {
      frame: calib.frame,
      points: calib.points.map((p) => ({ image: p.image, pitch: p.landmark.pitch })),
    }
    setCalib((c) => ({ ...c, saving: true, error: null }))
    try {
      // 다른 프레임의 기존 보정은 유지 (가장 가까운 키프레임 기준으로 전파됨).
      // 페이지를 연 뒤 다른 창에서 추가된 보정을 덮어쓰지 않도록 저장 직전 최신 목록을 다시 받는다.
      const latest = await getAnalysis(source.name)
      const others = (latest.calibration?.keyframes ?? []).filter((k) => k.frame !== calib.frame)
      await saveCalibration(source.name, [...others, keyframe])
      const [status, result] = await Promise.all([getAnalysisStatus(source.name), getAnalysis(source.name)])
      setAnalysis({ status, data: result })
      setCalib(null)
    } catch (err) {
      setCalib((c) => ({ ...c, saving: false, error: err.message }))
    }
  }

  const calibrating = Boolean(calib)
  const progress = duration > 0 ? Math.min(currentTime / duration, 1) * 100 : 0
  const transportDisabled = calibrating || (!connected && !videoMaster)
  const statusLabel = !connected && !analysisMode ? 'OFFLINE'
    : playing ? (analysisMode ? 'PLAYING' : 'LIVE') : 'PAUSED'

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-overline">SOCCER ANALYTICS</div>
            <div className="brand-name">Digital Twin</div>
          </div>
        </div>

        <Segmented label="화면 구성" value={layout} onChange={setLayout} options={[
          ['split', '분할', 'split'],
          ['video', '경기 영상', 'film'],
          ['twin', '3D 트윈', 'cube'],
        ]} />

        <div className={`status-pill ${statusLabel === 'OFFLINE' ? 'off' : playing ? 'live' : 'idle'}`}>
          <span className="dot" />
          {statusLabel}
        </div>
      </header>

      <main className="workspace">
        <section className={`stage layout-${layout}`}>
          {/* 경기 영상: 3D 전용 화면에서도 기준 시계 역할을 하도록 숨기기만 함 */}
          <div className="pane pane-video">
            <div className="pane-label"><Icon name="film" size={14} />경기 영상</div>
            {source ? (
              <>
                <video key={source.url} ref={videoRef} src={source.url} preload="metadata"
                  playsInline onClick={calibrating ? undefined : togglePlay} {...videoHandlers} />
                {analysisMode && (
                  <VideoOverlay analysis={analysis.data} frameIndex={videoFrame} teamColors={teamColors}
                    showBoxes={showBoxes} showLines={showLines}
                    calibration={calib && { points: calib.points, activeId: calib.activeId, onPick: pickPoint }} />
                )}
                {!video.ready && !video.error && <div className="pane-center muted">영상 불러오는 중</div>}
                {video.error && (
                  <div className="pane-center">
                    <Icon name="alert" size={28} />
                    <p className="pane-title">{source.name}</p>
                    <p className="muted">{video.error}</p>
                  </div>
                )}
              </>
            ) : (
              <div className="pane-center">
                <Icon name="film" size={32} />
                <p className="pane-title">경기 영상을 선택하세요</p>
                <p className="muted">라이브러리에서 선택하거나 새 영상을 업로드한 뒤<br />분석하면 탐지 박스와 3D 트윈이 영상에 맞춰 재생됩니다.</p>
                <label className="btn btn-primary">
                  <Icon name="upload" size={16} />영상 업로드
                  <input type="file" accept="video/*" hidden onChange={handleUpload} />
                </label>
              </div>
            )}
          </div>

          <div className="pane pane-twin">
            <div className="pane-label">
              <Icon name="cube" size={14} />3D 디지털 트윈
              <span className={`source-chip ${analysisMode && calibrated ? 'real' : ''}`}>
                {analysisMode ? (calibrated ? '분석 데이터' : '보정 필요') : '더미 데이터'}
              </span>
            </div>
            <div className="pane-tools">
              <Segmented label="카메라" value={cameraMode} onChange={setCameraMode} options={[
                ['3d', '자유 시점'],
                ['topdown', '탑다운'],
                ['player', '선수 시점'],
              ]} />
            </div>
            {/* demand: 데이터·카메라가 바뀔 때만 다시 그림 (정지 화면에서 CPU/GPU 를 쓰지 않음) */}
            <Canvas frameloop="demand" dpr={[1, 2]} camera={{ position: CAMERA_PRESETS['3d'], fov: 45 }}>
              <Scene players={players} ball={ball} cameraMode={cameraMode} teamColors={teamColors} />
            </Canvas>
            {analysisMode && !calibrated && !calibrating && (
              <div className="pane-center overlay">
                <Icon name="target" size={28} />
                <p className="pane-title">경기장 보정 후 3D 트윈에 연동됩니다</p>
                <p className="muted">자동 보정은 화면의 경기장 라인·센터서클로 기준점 없이 보정합니다.<br />
                  잘 안 되면 영상에서 기준점 4 곳 이상을 직접 지정하세요.</p>
                <div className="btn-row">
                  <button className="btn btn-primary" onClick={handleAutoCalibrate}>
                    <Icon name="scan" size={16} />자동 보정
                  </button>
                  <button className="btn btn-ghost" onClick={startCalibration}>
                    <Icon name="target" size={16} />직접 지정
                  </button>
                </div>
              </div>
            )}
            {analysisMode && calibrated && !twin && (
              <div className="pane-note">이 구간은 카메라 전환으로 보정이 끊겼습니다. 이 구간에서 보정을 추가하세요.</div>
            )}
            {!connected && !analysisMode && (
              <div className="pane-center overlay">
                <Icon name="alert" size={28} />
                <p className="pane-title">트래킹 서버에 연결할 수 없습니다</p>
                <p className="muted">백엔드(localhost:8000) 실행 후 새로고침하세요.</p>
              </div>
            )}
          </div>
        </section>

        <aside className="rail">
          <section className="card clock-card">
            <div className="eyebrow">경기 시간</div>
            <div className="clock num">{formatTime(currentTime)}</div>
            <div className="meta num">
              FRAME {currentFrame.toLocaleString()} · {Math.round(videoMaster ? fps : FPS)} FPS
            </div>
          </section>

          {calibrating ? (
            <CalibrationPanel frame={calib.frame} points={calib.points} activeId={calib.activeId}
              saving={calib.saving} error={calib.error}
              onSelect={(id) => setCalib((c) => ({ ...c, activeId: id, error: null }))}
              onRemove={(id) => setCalib((c) => ({ ...c, points: c.points.filter((p) => p.landmark.id !== id) }))}
              onSave={saveCalib} onCancel={() => setCalib(null)} />
          ) : source && !video.error && (
            <AnalysisCard status={analysis.status} summary={summary} mode={runMode} onMode={setRunMode}
              showBoxes={showBoxes} onToggleBoxes={() => setShowBoxes((s) => !s)}
              showLines={showLines} onToggleLines={() => setShowLines((s) => !s)}
              onStart={handleStartAnalysis} onCancel={handleCancelAnalysis} onCalibrate={startCalibration}
              onAutoCalibrate={handleAutoCalibrate} />
          )}

          {analysisMode && analysis.data?.stats && !calibrating && (
            <MatchStatsCard stats={analysis.data.stats} fps={fps} teamColors={teamColors}
              onSeek={seekTo} currentFrame={videoFrame} />
          )}

          <section className="card" style={{ '--home': teamColors.home, '--away': teamColors.away }}>
            <div className="teams">
              <span className="team home">HOME</span>
              <span className="possession">
                <span className="eyebrow">볼 소유</span>
                <strong className={owner ?? ''}>{owner === 'home' ? '홈' : owner === 'away' ? '원정' : '경합'}</strong>
              </span>
              <span className="team away">AWAY</span>
            </div>
            <CompareRow label="선수" home={home.count} away={away.count} />
            <CompareRow label="평균 속도 m/s" home={home.avg} away={away.avg} format={(v) => v.toFixed(1)} />
            <CompareRow label="최고 속도 m/s" home={home.max} away={away.max} format={(v) => v.toFixed(1)} />
          </section>

          <section className="card">
            <div className="eyebrow">볼</div>
            <div className="kv">
              <span>속도</span><strong className="num">{ball ? ball.velocity.toFixed(1) : '–'}<small> m/s</small></strong>
            </div>
            <div className="kv">
              <span>위치 (x, y)</span>
              <strong className="num">{ball ? `${ball.position_3d[0].toFixed(1)}, ${ball.position_3d[1].toFixed(1)}` : '–'}<small> m</small></strong>
            </div>
          </section>

          <section className="card library">
            <div className="library-head">
              <div className="eyebrow">경기 영상 라이브러리</div>
              <label className="btn btn-ghost" title="영상 업로드">
                <Icon name="upload" size={15} />업로드
                <input type="file" accept="video/*" hidden onChange={handleUpload} />
              </label>
            </div>
            {upload && <p className={`upload-status ${upload.status}`}>{upload.message}</p>}
            {videos.length === 0 ? (
              <p className="muted small">업로드된 영상이 없습니다.</p>
            ) : (
              <ul>
                {videos.map((v) => (
                  <li key={v.name}>
                    <button className={source?.name === v.name ? 'active' : ''} onClick={() => selectSource(v)}>
                      <Icon name="film" size={16} />
                      <span className="video-name">{v.name}</span>
                      <span className="video-meta num">{formatSize(v.size)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </aside>
      </main>

      <footer className="transport">
        <button className="play-btn" onClick={togglePlay} disabled={transportDisabled}
          aria-label={playing ? '일시정지' : '재생'}>
          <Icon name={playing ? 'pause' : 'play'} size={20} />
        </button>
        <button className="icon-btn" onClick={() => seekTo(0)} disabled={transportDisabled} aria-label="처음으로">
          <Icon name="restart" size={18} />
        </button>
        <span className="time num">{formatTime(currentTime)}</span>
        <input type="range" className="scrubber" min="0" max={duration || DEFAULT_TIMELINE} step="0.01"
          value={Math.min(currentTime, duration || DEFAULT_TIMELINE)}
          style={{ '--progress': `${progress}%` }}
          onChange={(e) => seekTo(parseFloat(e.target.value))}
          disabled={transportDisabled} aria-label="타임라인" />
        <span className="time num muted">{formatTime(duration)}</span>
        <span className="source-name">{videoMaster ? source.name : '트래킹 스트림'}</span>
      </footer>
    </div>
  )
}

export default App

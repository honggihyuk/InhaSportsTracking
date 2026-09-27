import { useState, useEffect, useRef, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Line } from '@react-three/drei'
import * as THREE from 'three'
import { useWebSocket, uploadVideo, listVideos } from './components/api'
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

function PlayerMarker({ player }) {
  const color = COLORS[player.team] ?? '#9aa3ad'
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
      <Text position={[0, 2.35, 0]} fontSize={0.85} color="#F2F3F5" anchorX="center" anchorY="middle"
        outlineWidth={0.04} outlineColor="#000000">
        {player.id}
      </Text>
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
  const { camera, controls, size } = useThree()
  const aspect = size.width / size.height

  useEffect(() => {
    const preset = CAMERA_PRESETS[cameraMode]
    if (!preset) return
    // 세로로 긴 화면(분할 모드 등)에서도 경기장 전체가 보이도록 거리 보정
    const fit = Math.max(1.3, 1.4 / aspect)
    camera.position.set(...preset.map((v) => v * fit))
    controls?.target.set(0, 0, 0)
    controls?.update()
  }, [cameraMode, camera, controls, aspect])

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

function Scene({ players, ball, cameraMode }) {
  const followed = players[0] // 선수 시점: 첫 번째 선수
  return (
    <>
      <color attach="background" args={[COLORS.background]} />
      <hemisphereLight args={['#dfe8ff', '#0b140e', 0.9]} />
      <directionalLight position={[40, 80, 30]} intensity={1.4} />
      <FootballField />
      {players
        .filter((p) => !(cameraMode === 'player' && p === followed)) // 시점 선수 자신은 숨김
        .map((p) => <PlayerMarker key={p.id} player={p} />)}
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

/* ---------- 앱 ---------- */
function App() {
  const { data, connected, sendCommand } = useWebSocket()
  const [layout, setLayout] = useState('split')
  const [cameraMode, setCameraMode] = useState('3d')
  const [streamPlaying, setStreamPlaying] = useState(true) // 영상이 없을 때 재생 상태
  const [videos, setVideos] = useState([])
  const [source, setSource] = useState(null) // 선택된 경기 영상 {name, url, size}
  const [video, setVideo] = useState({ ready: false, playing: false, duration: 0, error: null })
  const [upload, setUpload] = useState(null)
  const videoRef = useRef(null)
  const frameRef = useRef(0)

  const players = data?.players ?? []
  const ball = data?.ball ?? null
  const frame = data?.frame ?? 0
  useEffect(() => { frameRef.current = frame }, [frame])

  // 영상이 재생 가능하면 영상이 기준 시계, 아니면 트래킹 스트림이 기준
  const videoMaster = Boolean(source && video.ready && !video.error)
  const playing = videoMaster ? video.playing : streamPlaying
  const duration = videoMaster ? video.duration : DEFAULT_TIMELINE
  const currentTime = frame / FPS

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

  // 트래킹 스트림을 영상 시각에 맞춤
  const syncStream = useCallback((paused) => {
    const v = videoRef.current
    if (v) sendCommand({ type: 'seek', frame: Math.round(v.currentTime * FPS), paused })
  }, [sendCommand])

  const selectSource = (item) => {
    setSource(item)
    setVideo({ ready: false, playing: false, duration: 0, error: null })
    // 새 영상은 처음·정지 상태에서 시작 (재생 불가 코덱이면 이 상태로 스트림이 기준이 됨)
    sendCommand({ type: 'seek', frame: 0, paused: true })
    setStreamPlaying(false)
    if (layout === 'twin') setLayout('split')
  }

  const videoHandlers = {
    onLoadedMetadata: (e) => {
      // updater 는 나중에 실행되므로 이벤트 값은 미리 꺼내 둔다 (그 시점엔 currentTarget 이 null)
      const { duration } = e.currentTarget
      setVideo((s) => ({ ...s, ready: true, duration }))
      syncStream(true)
    },
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
    sendCommand({ type: 'seek', frame, paused: streamPlaying })
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

  const progress = duration > 0 ? Math.min(currentTime / duration, 1) * 100 : 0

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

        <div className={`status-pill ${connected ? (playing ? 'live' : 'idle') : 'off'}`}>
          <span className="dot" />
          {connected ? (playing ? 'LIVE' : 'PAUSED') : 'OFFLINE'}
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
                  playsInline onClick={togglePlay} {...videoHandlers} />
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
                <p className="muted">라이브러리에서 선택하거나 새 영상을 업로드하면<br />3D 트윈과 같은 타임라인으로 재생됩니다.</p>
                <label className="btn btn-primary">
                  <Icon name="upload" size={16} />영상 업로드
                  <input type="file" accept="video/*" hidden onChange={handleUpload} />
                </label>
              </div>
            )}
          </div>

          <div className="pane pane-twin">
            <div className="pane-label"><Icon name="cube" size={14} />3D 디지털 트윈</div>
            <div className="pane-tools">
              <Segmented label="카메라" value={cameraMode} onChange={setCameraMode} options={[
                ['3d', '자유 시점'],
                ['topdown', '탑다운'],
                ['player', '선수 시점'],
              ]} />
            </div>
            <Canvas dpr={[1, 2]} camera={{ position: CAMERA_PRESETS['3d'], fov: 45 }}>
              <Scene players={players} ball={ball} cameraMode={cameraMode} />
            </Canvas>
            {!connected && (
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
              FRAME {frame.toLocaleString()} · {FPS} FPS
            </div>
          </section>

          <section className="card">
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
        <button className="play-btn" onClick={togglePlay} disabled={!connected && !videoMaster}
          aria-label={playing ? '일시정지' : '재생'}>
          <Icon name={playing ? 'pause' : 'play'} size={20} />
        </button>
        <button className="icon-btn" onClick={() => seekTo(0)} disabled={!connected && !videoMaster} aria-label="처음으로">
          <Icon name="restart" size={18} />
        </button>
        <span className="time num">{formatTime(currentTime)}</span>
        <input type="range" className="scrubber" min="0" max={duration || DEFAULT_TIMELINE} step="0.1"
          value={Math.min(currentTime, duration || DEFAULT_TIMELINE)}
          style={{ '--progress': `${progress}%` }}
          onChange={(e) => seekTo(parseFloat(e.target.value))}
          disabled={!connected && !videoMaster} aria-label="타임라인" />
        <span className="time num muted">{formatTime(duration)}</span>
        <span className="source-name">{videoMaster ? source.name : '트래킹 스트림'}</span>
      </footer>
    </div>
  )
}

export default App

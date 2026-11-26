import { useEffect, useMemo, useRef, useState } from 'react'
import {
  PITCH_POLYLINES, PITCH_L as L, PITCH_W as W, HEAT_CELL, BALL_ID,
  vizTeamColor, colorDistance, inkOn, heatCounts, zoneShares, passNetwork, shapeAt,
} from '../analysis'

/**
 * 경기 분석 화면 — 팀 히트맵 · 패스 네트워크 · 팀 대형(경기장 + 시계열)
 *
 * 데이터: analysis/stats.py compute() 결과(stats.heatmaps · pass_network · shape · events)와 분석 결과의 world 좌표.
 * 현재 시점과 연동: 경기장 위에 현재 프레임의 선수 위치를 표시하고, '현재 시점까지' 범위를 고르면 히트맵·패스 네트워크가
 * 영상 재생에 맞춰 누적된다. 대형 시계열의 세로선은 현재 시각, 클릭하면 그 시각으로 이동.
 *
 * 색: 팀 색(teamColors)을 쓰되 유니폼 색이 어두운 잔디 위에서 안 보이면 밝기만 옮김(vizTeamColor).
 * 두 팀 색이 너무 비슷하면(OKLab ΔE < 15) 기본 홈/원정 색으로 바꾼다.
 */

const DEFAULT = { home: '#E5534B', away: '#4A9BF0' }
const TURF = '#13261A'
const TEAM_LABEL = { home: '홈', away: '원정' }
const TRAIL_S = 10 // 대형 화면의 무게중심 궤적 길이 (초)
const PAD = 3 // 경기장 바깥 여백 m

const VIEWS = [['heatmap', '히트맵'], ['passes', '패스 네트워크'], ['shape', '팀 대형']]
const RANGES = [['all', '전체 경기'], ['upto', '현재 시점까지']]
const METRICS = [
  ['width', '폭', '선수들이 너비 방향으로 퍼진 거리(y 범위)'],
  ['length', '길이', '최전방~최후방 거리(x 범위)'],
  ['defensive_line', '수비 라인', '골키퍼를 뺀 최후방 선수의 x 좌표'],
  ['centroid_x', '무게중심 x', '필드 선수 평균 x 좌표 (+ = 오른쪽 골문 쪽)'],
]

const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '–')
const formatTime = (s) => {
  const m = Math.floor(s / 60)
  return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(1).padStart(4, '0')}`
}
const playerName = (id, jersey) => (jersey != null ? `#${id} · ${jersey}번` : `#${id}`)

function Tabs({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="tablist" aria-label={label}>
      {options.map(([key, text]) => (
        <button key={key} role="tab" aria-selected={value === key}
          className={value === key ? 'active' : ''} onClick={() => onChange(key)}>
          <span>{text}</span>
        </button>
      ))}
    </div>
  )
}

/* ---------- 경기장 (탑다운, SVG y = -경기장 y) ---------- */
function Pitch({ children, label, onLeave }) {
  return (
    <svg className="av-pitch" viewBox={`${-L - PAD} ${-W - PAD} ${2 * (L + PAD)} ${2 * (W + PAD)}`}
      role="img" aria-label={label} onPointerLeave={onLeave}>
      <rect x={-L - PAD} y={-W - PAD} width={2 * (L + PAD)} height={2 * (W + PAD)} fill={TURF} />
      <g className="av-pitch-lines">
        {PITCH_POLYLINES.map((pl, i) => (
          <polyline key={i} points={pl.map(([x, y]) => `${x},${-y}`).join(' ')} vectorEffect="non-scaling-stroke" />
        ))}
      </g>
      {children}
    </svg>
  )
}

/** 현재 프레임 선수 위치 (시점 연동 표시) */
function CurrentPlayers({ analysis, frame, teams, colors }) {
  const entries = analysis?.world?.[frame] ?? []
  return (
    <g className="av-now" aria-hidden="true">
      {entries.map(([id, x, y]) => {
        if (id === BALL_ID) return <circle key="ball" cx={x} cy={-y} r={0.7} className="av-now-ball" vectorEffect="non-scaling-stroke" />
        const team = analysis.teams?.[id]
        if (!teams.includes(team)) return null
        return <circle key={id} cx={x} cy={-y} r={0.85} fill={colors[team]} className="av-now-dot" vectorEffect="non-scaling-stroke" />
      })}
    </g>
  )
}

/* ---------- 히트맵 ---------- */
function Heatmap({ grid, color, onHover }) {
  const max = Math.max(0, ...grid.flat())
  if (!max) return null
  const g = 0.25 // 셀 사이 간격 m (표면 간격)
  return (
    <g>
      {grid.map((row, i) => row.map((v, j) => {
        if (!v) return null
        const x = -L + j * HEAT_CELL, y = -W + i * HEAT_CELL
        const w = Math.min(HEAT_CELL, L - x), h = Math.min(HEAT_CELL, W - y)
        const rel = v / max
        return (
          <rect key={`${i}-${j}`} x={x + g / 2} y={-(y + h) + g / 2} width={w - g} height={h - g} rx={0.5}
            fill={color} fillOpacity={0.1 + 0.78 * rel} className="av-cell"
            onPointerMove={(e) => onHover(e, [`${(rel * 100).toFixed(0)}%`, `최대 칸 대비 밀도`,
              `x ${x.toFixed(0)}~${(x + w).toFixed(0)} m · y ${y.toFixed(0)}~${(y + h).toFixed(0)} m`])} />
        )
      }))}
    </g>
  )
}

/* ---------- 패스 네트워크 ---------- */
function PassNetwork({ net, color, onHover, onSeekPair }) {
  const byId = new Map(net.nodes.map((n) => [n.id, n]))
  const maxEdge = Math.max(1, ...net.edges.map((e) => e.count))
  const ink = inkOn(color)
  return (
    <g>
      {net.edges.map((e) => {
        const a = byId.get(e.a), b = byId.get(e.b)
        const pts = { x1: a.position[0], y1: -a.position[1], x2: b.position[0], y2: -b.position[1] }
        const label = `${playerName(e.a, a.jersey_number)} ↔ ${playerName(e.b, b.jersey_number)}`
        const tip = [`${e.count}회`, label, `${e.a}→${e.b} ${e.a_to_b} · ${e.b_to_a} ${e.b}→${e.a}`, '클릭: 첫 패스로 이동']
        return (
          <g key={`${e.a}-${e.b}`} className="av-edge" tabIndex={0} role="button" aria-label={`${label} 패스 ${e.count}회`}
            onPointerMove={(ev) => onHover(ev, tip)} onFocus={(ev) => onHover(ev, tip)}
            onClick={() => onSeekPair(e.a, e.b)} onKeyDown={(ev) => ev.key === 'Enter' && onSeekPair(e.a, e.b)}>
            <line {...pts} className="av-edge-hit" vectorEffect="non-scaling-stroke" />
            <line {...pts} stroke={color} strokeWidth={1.5 + (5 * e.count) / maxEdge} strokeOpacity={0.75}
              strokeLinecap="round" vectorEffect="non-scaling-stroke" />
          </g>
        )
      })}
      {net.nodes.map((n) => {
        const r = 1.7 + 0.45 * Math.sqrt(n.passes + n.received)
        const tip = [`${n.passes + n.received}회 관여`, playerName(n.id, n.jersey_number) + (n.role === 'goalkeeper' ? ' GK' : ''),
          `패스 ${n.passes} · 받음 ${n.received}`, `평균 위치 ${fmt(n.position[0])}, ${fmt(n.position[1])} m`]
        return (
          <g key={n.id} className="av-node" tabIndex={0} aria-label={tip.slice(1).join(', ')}
            onPointerMove={(ev) => onHover(ev, tip)} onFocus={(ev) => onHover(ev, tip)}>
            <circle cx={n.position[0]} cy={-n.position[1]} r={r + 1.6} fill="transparent" />
            <circle cx={n.position[0]} cy={-n.position[1]} r={r} fill={color} stroke={TURF} strokeWidth={2}
              vectorEffect="non-scaling-stroke" />
            <text x={n.position[0]} y={-n.position[1]} fill={ink} className="av-node-label">
              {n.jersey_number ?? n.id}
            </text>
          </g>
        )
      })}
    </g>
  )
}

/* ---------- 팀 대형 (현재 프레임) ---------- */
function ShapeLayer({ shape, trail, color, team }) {
  if (!shape) return null
  const [cx, cy] = shape.centroid
  return (
    <g>
      {shape.hull.length >= 3 && (
        <polygon points={shape.hull.map(([x, y]) => `${x},${-y}`).join(' ')} fill={color} fillOpacity={0.12}
          stroke={color} strokeWidth={2} strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      )}
      {trail.length > 1 && (
        <polyline points={trail.map(([x, y]) => `${x},${-y}`).join(' ')} fill="none" stroke={color} strokeOpacity={0.55}
          strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      )}
      <line x1={shape.defensive_line} x2={shape.defensive_line} y1={-W} y2={W} stroke={color} strokeWidth={2}
        vectorEffect="non-scaling-stroke" />
      <text x={shape.defensive_line} y={-W - 0.9} className="av-pitch-note">{TEAM_LABEL[team]} 수비 라인</text>
      <circle cx={cx} cy={-cy} r={1.4} fill={color} stroke={TURF} strokeWidth={2} vectorEffect="non-scaling-stroke" />
    </g>
  )
}

/* ---------- 대형 시계열 (한 지표, 두 팀, 한 축) ---------- */
function ShapeChart({ series, metric, currentTime, duration, colors, onSeek }) {
  const wrapRef = useRef(null)
  const [width, setWidth] = useState(600)
  const [hoverT, setHoverT] = useState(null)
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(240, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const H = 150, m = { l: 34, r: 56, t: 10, b: 22 }
  const values = ['home', 'away'].flatMap((t) => series[t].map((p) => p.v)).filter(Number.isFinite)
  if (!values.length) return <p className="av-empty">대형 시계열이 없습니다 (경기장 보정 구간이 부족).</p>
  let lo = Math.min(...values), hi = Math.max(...values)
  const span = hi - lo || 10
  const step = [1, 2, 5, 10, 20, 25, 50].find((s) => span / s <= 4) ?? 50
  lo = Math.floor(lo / step) * step
  hi = Math.ceil(hi / step) * step
  if (hi === lo) hi = lo + step
  const ticks = []
  for (let v = lo; v <= hi + 1e-9; v += step) ticks.push(v)
  const T = Math.max(duration || 0, series.home.at(-1)?.t ?? 0, series.away.at(-1)?.t ?? 0) || 1
  const sx = (t) => m.l + (t / T) * (width - m.l - m.r)
  const sy = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b)
  const path = (pts) => pts.map((p, i) => `${i === 0 || p.gap ? 'M' : 'L'}${sx(p.t).toFixed(1)},${sy(p.v).toFixed(1)}`).join('')
  const nearest = (pts, t) => pts.reduce((best, p) => (!best || Math.abs(p.t - t) < Math.abs(best.t - t) ? p : best), null)
  const tAt = (e) => {
    const r = e.currentTarget.getBoundingClientRect()
    return Math.min(T, Math.max(0, ((e.clientX - r.left - m.l) / (width - m.l - m.r)) * T))
  }
  const readT = hoverT ?? currentTime
  const read = { home: nearest(series.home, readT), away: nearest(series.away, readT) }
  const unit = METRICS.find((x) => x[0] === metric)

  // 끝값 직접 라벨 (겹치면 범례·툴팁에 맡김)
  const ends = ['home', 'away'].map((t) => ({ t, p: series[t].at(-1) })).filter((e) => e.p)
  const endsCollide = ends.length === 2 && Math.abs(sy(ends[0].p.v) - sy(ends[1].p.v)) < 12

  return (
    <div className="av-chart" ref={wrapRef}>
      <svg width={width} height={H} role="img" aria-label={`팀 ${unit[1]} 시계열 (m)`}
        onPointerMove={(e) => setHoverT(tAt(e))} onPointerLeave={() => setHoverT(null)}
        onClick={(e) => onSeek(tAt(e))}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={m.l} x2={width - m.r} y1={sy(v)} y2={sy(v)} className="av-grid" />
            <text x={m.l - 6} y={sy(v)} className="av-tick" textAnchor="end" dominantBaseline="middle">{v}</text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <text key={f} x={sx(f * T)} y={H - 6} className="av-tick" textAnchor="middle">{formatTime(f * T)}</text>
        ))}
        {['home', 'away'].map((t) => series[t].length > 0 && (
          <path key={t} d={path(series[t])} fill="none" stroke={colors[t]} strokeWidth={2}
            strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {!endsCollide && ends.map(({ t, p }) => (
          <text key={t} x={sx(p.t) + 6} y={sy(p.v)} className="av-end-label" dominantBaseline="middle">
            {TEAM_LABEL[t]} {fmt(p.v)}
          </text>
        ))}
        <line x1={sx(currentTime)} x2={sx(currentTime)} y1={m.t} y2={H - m.b} className="av-now-line" />
        {hoverT != null && <line x1={sx(hoverT)} x2={sx(hoverT)} y1={m.t} y2={H - m.b} className="av-hover-line" />}
        {['home', 'away'].map((t) => read[t] && (
          <circle key={t} cx={sx(read[t].t)} cy={sy(read[t].v)} r={4} fill={colors[t]} stroke="var(--surface-2)" strokeWidth={2} />
        ))}
      </svg>
      <div className="av-readout num" aria-live="polite">
        <span className="muted">{hoverT != null ? formatTime(hoverT) : `현재 ${formatTime(currentTime)}`}</span>
        {['home', 'away'].map((t) => (
          <span key={t}><i style={{ background: colors[t] }} /><b>{read[t] ? fmt(read[t].v) : '–'}</b> m {TEAM_LABEL[t]}</span>
        ))}
        <span className="muted">{hoverT != null ? '클릭: 이 시각으로 이동' : unit[2]}</span>
      </div>
    </div>
  )
}

/* ---------- 본체 ---------- */
export function AnalysisViews({ analysis, stats, currentFrame, fps, duration, teamColors, onSeek }) {
  const [view, setView] = useState('heatmap')
  const [team, setTeam] = useState('home')
  const [range, setRange] = useState('all')
  const [metric, setMetric] = useState('width')
  const [showTable, setShowTable] = useState(false)
  const [tip, setTip] = useState(null)
  const stageRef = useRef(null)

  const colors = useMemo(() => {
    const c = { home: vizTeamColor(teamColors.home), away: vizTeamColor(teamColors.away) }
    return colorDistance(c.home, c.away) < 15 ? DEFAULT : c
  }, [teamColors.home, teamColors.away])

  // '현재 시점까지' 누적은 0.5 초 단위로만 다시 계산 (재생 중 매 프레임 계산하지 않음)
  const step = Math.max(1, Math.round(fps / 2))
  const upto = range === 'upto' ? Math.floor(currentFrame / step) * step + step - 1 : Infinity
  const heat = useMemo(() => {
    if (range === 'all' && stats?.heatmaps?.home) return { home: { grid: stats.heatmaps.home }, away: { grid: stats.heatmaps.away } }
    return heatCounts(analysis, 0, upto)
  }, [analysis, stats, range, upto])
  const net = useMemo(() => passNetwork(analysis, stats, team, upto), [analysis, stats, team, upto])
  const defends = (t) => stats?.teams?.[t]?.defends ?? (t === 'home' ? -1 : 1)

  const shapes = {
    home: shapeAt(analysis, currentFrame, 'home', defends('home')),
    away: shapeAt(analysis, currentFrame, 'away', defends('away')),
  }
  const series = useMemo(() => {
    const out = { home: [], away: [] }
    const every = Math.max(1, Math.round(fps / 2))
    for (const row of stats?.shape ?? []) {
      for (const t of ['home', 'away']) {
        const s = row[t]
        if (!s) continue
        const v = metric === 'centroid_x' ? s.centroid[0] : s[metric]
        const prev = out[t].at(-1)
        out[t].push({ t: row.frame / fps, frame: row.frame, v, cx: s.centroid, gap: prev ? row.frame - prev.frame > 2 * every : false })
      }
    }
    return out
  }, [stats, metric, fps])
  const trails = useMemo(() => {
    const from = currentFrame - TRAIL_S * fps
    const pick = (t) => (stats?.shape ?? []).filter((r) => r[t] && r.frame >= from && r.frame <= currentFrame).map((r) => r[t].centroid)
    return { home: pick('home'), away: pick('away') }
  }, [stats, currentFrame, fps])

  const hover = (e, lines) => {
    const box = stageRef.current?.getBoundingClientRect()
    if (!box) return
    const src = e.clientX != null ? e : null
    const r = src ? null : e.currentTarget.getBoundingClientRect()
    setTip({ x: (src ? e.clientX : r.left + r.width / 2) - box.left, y: (src ? e.clientY : r.top) - box.top, lines })
  }
  const seekPair = (a, b) => {
    const e = (stats?.events ?? []).find((x) => x.type === 'pass' && ((x.from === a && x.to === b) || (x.from === b && x.to === a)))
    if (e) onSeek(e.frame / fps)
  }

  const uptoLabel = range === 'upto' ? `0:00 ~ ${formatTime(currentFrame / fps)}` : '전체 경기'
  const teamTabs = view !== 'shape' && (
    <div className="segmented av-team" role="tablist" aria-label="팀 선택">
      {['home', 'away'].map((t) => (
        <button key={t} role="tab" aria-selected={team === t} className={team === t ? 'active' : ''} onClick={() => setTeam(t)}>
          <i style={{ background: colors[t] }} /><span>{TEAM_LABEL[t]}</span>
        </button>
      ))}
    </div>
  )

  let caption, legend, table
  if (view === 'heatmap') {
    const zones = zoneShares(heat[team].grid, defends(team))
    caption = `${TEAM_LABEL[team]} 위치 밀도 · ${HEAT_CELL} m 격자 · ${uptoLabel}`
    legend = (
      <div className="av-legend">
        <div className="av-ramp" style={{ '--c': colors[team] }} aria-hidden="true" />
        <div className="av-ramp-labels"><span>적음</span><span>많음</span></div>
        <p className="muted">칸 색이 진할수록 그 칸에 {TEAM_LABEL[team]} 선수가 오래 머문 것입니다. 가장 많은 칸 = 100%.</p>
      </div>
    )
    table = (
      <table className="stats-table num av-table">
        <caption>구역별 점유 비율 (공격 방향 기준)</caption>
        <thead><tr><th>구역</th><th>왼쪽</th><th>가운데</th><th>오른쪽</th></tr></thead>
        <tbody>
          {['수비 1/3', '중원', '공격 1/3'].map((name, i) => (
            <tr key={name}><td>{name}</td>{zones[i].map((v, j) => <td key={j}>{(v * 100).toFixed(0)}%</td>)}</tr>
          ))}
        </tbody>
      </table>
    )
  } else if (view === 'passes') {
    caption = `${TEAM_LABEL[team]} 패스 ${net.passes}회 · 선수 ${net.nodes.length}명 · ${uptoLabel}`
    legend = (
      <div className="av-legend">
        <p className="muted">원 = 패스를 주고받은 선수(패스 순간의 평균 위치), 숫자 = 등번호(없으면 트랙 ID).
          선 굵기 = 두 선수 사이 패스 횟수, 원 크기 = 관여 횟수. 선을 클릭하면 그 패스로 이동합니다.</p>
      </div>
    )
    const byId = new Map(net.nodes.map((n) => [n.id, n]))
    table = (
      <table className="stats-table num av-table">
        <caption>선수 쌍별 패스</caption>
        <thead><tr><th>선수 쌍</th><th>횟수</th><th>방향별</th></tr></thead>
        <tbody>
          {[...net.edges].sort((a, b) => b.count - a.count).slice(0, 10).map((e) => (
            <tr key={`${e.a}-${e.b}`}>
              <td>{playerName(e.a, byId.get(e.a)?.jersey_number)} ↔ {playerName(e.b, byId.get(e.b)?.jersey_number)}</td>
              <td>{e.count}</td><td>{e.a_to_b} / {e.b_to_a}</td>
            </tr>
          ))}
          {net.edges.length === 0 && <tr><td colSpan={3} className="muted">패스 없음</td></tr>}
        </tbody>
      </table>
    )
  } else {
    caption = `현재 ${formatTime(currentFrame / fps)} 대형 · 색 영역 = 필드 선수 범위, 선 = 최근 ${TRAIL_S}초 무게중심 이동`
    legend = (
      <div className="av-legend">
        <p className="muted">골키퍼를 뺀 필드 선수 기준. 세로선은 각 팀의 수비 라인(최후방 선수), 큰 점은 무게중심입니다.</p>
      </div>
    )
    table = (
      <table className="stats-table num av-table">
        <caption>현재 프레임 대형 (m)</caption>
        <thead><tr><th>지표</th><th>{TEAM_LABEL.home}</th><th>{TEAM_LABEL.away}</th></tr></thead>
        <tbody>
          {[['폭', (s) => s.width], ['길이', (s) => s.length], ['수비 라인 x', (s) => s.defensive_line],
            ['무게중심 x', (s) => s.centroid[0]], ['필드 선수', (s) => s.points.length]].map(([name, f]) => (
            <tr key={name}><td>{name}</td>
              {['home', 'away'].map((t) => <td key={t}>{shapes[t] ? (name === '필드 선수' ? f(shapes[t]) : fmt(f(shapes[t]))) : '–'}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  const emptyPitch = view === 'heatmap' ? !heat[team].grid.flat().some((v) => v > 0)
    : view === 'passes' ? net.nodes.length === 0
      : !shapes.home && !shapes.away

  return (
    <section className="card analysis-views">
      <div className="card-head">
        <div className="eyebrow">경기 분석</div>
        <Tabs label="분석 화면" value={view} onChange={(v) => { setView(v); setTip(null) }} options={VIEWS} />
      </div>

      <div className="av-filters">
        {teamTabs}
        {view !== 'shape'
          ? <Tabs label="구간" value={range} onChange={setRange} options={RANGES} />
          : <Tabs label="시계열 지표" value={metric} onChange={setMetric} options={METRICS.map(([k, t]) => [k, t])} />}
        <span className="toolbar-spacer" />
        <label className="switch">
          <input type="checkbox" checked={showTable} onChange={() => setShowTable((s) => !s)} />
          <span>표 보기</span>
        </label>
      </div>

      <div className="av-body">
        <div className="av-stage" ref={stageRef}>
          <Pitch label={caption} onLeave={() => setTip(null)}>
            {view === 'heatmap' && <Heatmap grid={heat[team].grid} color={colors[team]} onHover={hover} />}
            {view === 'passes' && <PassNetwork net={net} color={colors[team]} onHover={hover} onSeekPair={seekPair} />}
            {view === 'shape' && ['home', 'away'].map((t) => (
              <ShapeLayer key={t} team={t} shape={shapes[t]} trail={trails[t]} color={colors[t]} />
            ))}
            <CurrentPlayers analysis={analysis} frame={currentFrame} colors={colors}
              teams={view === 'shape' ? ['home', 'away'] : [team]} />
          </Pitch>
          {emptyPitch && (
            <div className="av-empty-overlay">
              {view === 'passes' ? `이 구간에 ${TEAM_LABEL[team]} 팀 패스가 없습니다.`
                : view === 'shape' ? '이 프레임은 경기장 보정이 끊겨 대형을 계산할 수 없습니다.'
                  : '이 구간에 위치 데이터가 없습니다.'}
            </div>
          )}
          {tip && (
            <div className="av-tip" style={{ left: tip.x, top: tip.y }} role="tooltip">
              <b className="num">{tip.lines[0]}</b>
              {tip.lines.slice(1).map((l) => <span key={l}>{l}</span>)}
            </div>
          )}
        </div>
        <div className="av-side">
          <p className="av-caption">{caption}</p>
          {legend}
          <p className="av-now-note muted"><i className="av-now-key" />작은 점: 현재 프레임 선수 위치 (노란 점 = 공)</p>
          {showTable && table}
        </div>
      </div>

      {view === 'shape' && (
        <ShapeChart series={series} metric={metric} currentTime={currentFrame / fps}
          duration={duration || (analysis?.world?.length ?? 0) / fps} colors={colors} onSeek={onSeek} />
      )}
    </section>
  )
}

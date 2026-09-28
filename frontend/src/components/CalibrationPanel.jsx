import { LANDMARKS } from '../analysis'

const L = 52.5
const W = 34

// 탑다운 경기장 (SVG 는 y 가 아래로 증가하므로 경기장 y 를 뒤집어 그림)
function PitchLines() {
  const box = (x1, x2, hw) => <rect x={Math.min(x1, x2)} y={-hw} width={Math.abs(x2 - x1)} height={hw * 2} />
  return (
    <g className="pitch-lines">
      <rect x={-L} y={-W} width={L * 2} height={W * 2} />
      <line x1={0} y1={-W} x2={0} y2={W} />
      <circle cx={0} cy={0} r={9.15} />
      {box(-L, -L + 16.5, 20.16)}
      {box(L, L - 16.5, 20.16)}
      {box(-L, -L + 5.5, 9.16)}
      {box(L, L - 5.5, 9.16)}
      <path d={`M ${-L + 16.5} -7.31 A 9.15 9.15 0 0 1 ${-L + 16.5} 7.31`} />
      <path d={`M ${L - 16.5} -7.31 A 9.15 9.15 0 0 0 ${L - 16.5} 7.31`} />
    </g>
  )
}

/**
 * 경기장 보정 패널
 * 1) 도면에서 기준점을 선택 → 2) 영상에서 같은 지점을 클릭 (VideoOverlay 가 처리)
 * 4 점 이상이면 저장. 현재 프레임이 키프레임이 되고, 카메라 움직임으로 전 프레임에 전파된다.
 */
export function CalibrationPanel({ frame, points, activeId, onSelect, onRemove, onSave, onCancel, saving, error }) {
  const placed = new Map(points.map((p) => [p.landmark.id, p]))
  const active = LANDMARKS.find((l) => l.id === activeId)
  return (
    <section className="card calibration">
      <div className="card-head">
        <div className="eyebrow">경기장 보정 · 프레임 {frame.toLocaleString()}</div>
        <span className="num small muted">{points.length}/4+</span>
      </div>
      <svg viewBox={`${-L - 3} ${-W - 3} ${L * 2 + 6} ${W * 2 + 6}`} className="pitch-map" role="img"
        aria-label="경기장 기준점 선택 도면">
        <PitchLines />
        {LANDMARKS.map((l) => (
          <circle key={l.id} cx={l.pitch[0]} cy={-l.pitch[1]} r={l.id === activeId ? 2.4 : 1.6}
            className={`landmark ${placed.has(l.id) ? 'placed' : ''} ${l.id === activeId ? 'active' : ''}`}
            onClick={() => onSelect(l.id)}>
            <title>{l.label}</title>
          </circle>
        ))}
      </svg>
      <p className="calibration-hint">
        {active
          ? <><strong>{active.label}</strong> — 영상에서 이 지점을 클릭하세요.</>
          : '도면에서 영상에 보이는 기준점을 선택하세요. 서로 한 직선 위에 있지 않은 4 점 이상이 필요합니다.'}
      </p>
      {points.length > 0 && (
        <ul className="calibration-points">
          {points.map(({ landmark, image }) => (
            <li key={landmark.id}>
              <span>{landmark.label}</span>
              <span className="num muted">{Math.round(image[0])}, {Math.round(image[1])}</span>
              <button className="link-btn" onClick={() => onRemove(landmark.id)} aria-label={`${landmark.label} 삭제`}>삭제</button>
            </li>
          ))}
        </ul>
      )}
      {error && <p className="upload-status error">{error}</p>}
      <div className="card-actions">
        <button className="btn btn-ghost" onClick={onCancel}>취소</button>
        <button className="btn btn-primary" onClick={onSave} disabled={points.length < 4 || saving}>
          {saving ? '계산 중' : '보정 저장'}
        </button>
      </div>
    </section>
  )
}

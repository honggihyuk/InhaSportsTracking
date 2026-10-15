import { useState } from 'react'
import { askMatch } from './api'

const EXAMPLES = ['가장 많이 뛴 선수는?', '스프린트 장면을 보여줘', '점유율은 어땠어?']
const TOOL_LABEL = {
  match_overview: '경기 개요', player_stats: '선수 지표', get_possession: '점유', find_events: '이벤트',
  team_shape: '팀 대형', query_tracks: '궤적', get_frame: '장면 보기', crop_player: '선수 확대',
}

/** 모델 답변의 **굵게** 만 살려서 표시 (그 밖의 마크다운은 글자 그대로) */
const Answer = ({ text }) => (
  <p className="ask-a">
    {text.split(/\*\*(.+?)\*\*/g).map((part, i) => (i % 2 ? <strong key={i}>{part}</strong> : part))}
  </p>
)

const formatTime = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`

/* ---------- 경기 질의 카드 (VLM 에이전트) — 답변의 근거 프레임을 누르면 그 장면으로 이동 ---------- */
export function AskPanel({ name, fps, onSeek }) {
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState([]) // {question, answer, frames, steps, error}
  const [busy, setBusy] = useState(false)

  const ask = async (q) => {
    const text = (q ?? question).trim()
    if (!text || busy) return
    setBusy(true)
    setQuestion('')
    const history = turns.filter((t) => t.answer).flatMap((t) => [
      { role: 'user', content: t.question }, { role: 'assistant', content: t.answer }])
    try {
      const res = await askMatch(name, text, history)
      setTurns((ts) => [...ts, { question: text, ...res }])
    } catch (e) {
      setTurns((ts) => [...ts, { question: text, error: e.message }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card ask-card">
      <div className="card-head">
        <div className="eyebrow">경기 질의</div>
        <span className="tag">VLM</span>
      </div>
      {turns.length === 0 && (
        <div className="ask-examples">
          {EXAMPLES.map((q) => (
            <button key={q} className="btn btn-ghost" onClick={() => ask(q)} disabled={busy}>{q}</button>
          ))}
        </div>
      )}
      <ul className="ask-log">
        {turns.map((t, i) => (
          <li key={i}>
            <p className="ask-q">{t.question}</p>
            {t.error ? <p className="card-text warn-text">{t.error}</p> : <Answer text={t.answer} />}
            {t.frames?.length > 0 && (
              <div className="ask-frames">
                {t.frames.map((f) => (
                  <button key={f} className="btn btn-ghost num" onClick={() => onSeek(f / fps)}>
                    {formatTime(f / fps)}
                  </button>
                ))}
              </div>
            )}
            {t.steps?.length > 0 && (
              <p className="ask-steps muted">근거: {[...new Set(t.steps.map((s) => TOOL_LABEL[s.tool] ?? s.tool))].join(' · ')}</p>
            )}
          </li>
        ))}
        {busy && <li className="muted small">분석 데이터를 살펴보는 중… (첫 질문은 모델을 불러오느라 더 걸립니다)</li>}
      </ul>
      <form className="ask-form" onSubmit={(e) => { e.preventDefault(); ask() }}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="경기에 대해 물어보세요"
          disabled={busy} aria-label="질문" />
        <button className="btn btn-primary" type="submit" disabled={busy || !question.trim()}>질문</button>
      </form>
    </section>
  )
}

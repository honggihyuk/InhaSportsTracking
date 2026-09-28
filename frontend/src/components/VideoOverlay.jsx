import { useEffect, useRef, useState } from 'react'
import { KIND_BALL, videoRect, inkOn } from '../analysis'

const ACCENT = '#D7FF3C'
const OTHER = '#9AA3AD'

/**
 * 경기 영상 위 탐지 박스 오버레이 (canvas)
 * - 분석 결과의 frameIndex 프레임 박스를 팀 색으로 그림 (선수 #ID, 공 BALL)
 * - 보정 모드: 클릭한 지점을 원본 해상도 픽셀 좌표로 onPick 에 전달, 찍은 점 표시
 */
export function VideoOverlay({ analysis, frameIndex, teamColors, showBoxes, calibration }) {
  const canvasRef = useRef(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  useEffect(() => {
    const el = canvasRef.current?.parentElement
    if (!el) return
    setSize({ w: el.clientWidth, h: el.clientHeight }) // 첫 측정은 즉시 (ResizeObserver 는 다음 렌더링 단계에서 호출)
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize({ w: width, h: height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !size.w) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = size.w * dpr
    canvas.height = size.h * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size.w, size.h)
    const rect = videoRect(size.w, size.h, analysis?.width, analysis?.height)
    if (!rect) return
    const X = (u) => rect.x + u * rect.scale
    const Y = (v) => rect.y + v * rect.scale
    ctx.font = '600 11px "Pretendard Variable", system-ui, sans-serif'
    ctx.textBaseline = 'middle'

    const chip = (text, x, y, bg) => {
      const w = ctx.measureText(text).width + 10
      ctx.fillStyle = bg
      ctx.beginPath()
      ctx.roundRect(x, y - 16, w, 15, 4)
      ctx.fill()
      ctx.fillStyle = inkOn(bg)
      ctx.fillText(text, x + 5, y - 8.5)
    }

    if (showBoxes && !calibration) {
      for (const [id, kind, x1, y1, x2, y2] of analysis.frames[frameIndex] ?? []) {
        if (kind === KIND_BALL) {
          const cx = X((x1 + x2) / 2), cy = Y((y1 + y2) / 2)
          const r = Math.max((x2 - x1) * rect.scale, (y2 - y1) * rect.scale) / 2 + 5
          ctx.strokeStyle = ACCENT
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.arc(cx, cy, r, 0, Math.PI * 2)
          ctx.stroke()
          chip('BALL', cx + r - 4, cy - r + 6, ACCENT)
        } else {
          const color = teamColors[analysis.teams?.[id]] ?? OTHER
          ctx.strokeStyle = color
          ctx.lineWidth = 1.75
          ctx.strokeRect(X(x1), Y(y1), (x2 - x1) * rect.scale, (y2 - y1) * rect.scale)
          chip(`#${id}`, X(x1), Y(y1), color)
        }
      }
    }

    if (calibration) {
      for (const { landmark, image } of calibration.points) {
        const x = X(image[0]), y = Y(image[1])
        const active = landmark.id === calibration.activeId
        ctx.fillStyle = ACCENT
        ctx.strokeStyle = '#0A0B0D'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(x, y, active ? 6 : 4.5, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
        chip(landmark.label, x + 8, y + 4, 'rgba(10, 11, 13, 0.8)')
      }
    }
  }, [analysis, frameIndex, teamColors, showBoxes, calibration, size])

  const handleClick = (e) => {
    const rect = videoRect(size.w, size.h, analysis?.width, analysis?.height)
    if (!calibration || !rect) return
    const box = canvasRef.current.getBoundingClientRect()
    const u = (e.clientX - box.left - rect.x) / rect.scale
    const v = (e.clientY - box.top - rect.y) / rect.scale
    if (u < 0 || v < 0 || u > analysis.width || v > analysis.height) return
    calibration.onPick([Math.round(u * 10) / 10, Math.round(v * 10) / 10])
  }

  return (
    <canvas ref={canvasRef} className={`video-overlay ${calibration ? 'picking' : ''}`}
      onClick={handleClick} aria-hidden={!calibration} />
  )
}

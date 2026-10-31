import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// 서체를 번들에 포함 (외부 CDN 없이 동작)
// 라틴: Archivo(본문) · Big Shoulders Display(제목·수치), 한글: Pretendard
// 라벨·시간용 Spline Sans Mono 는 공백을 뺀 범위로 index.css 에서 직접 선언 ("CS Mono")
import '@fontsource-variable/archivo'
import '@fontsource-variable/big-shoulders-display'
import 'pretendard/dist/web/variable/pretendardvariable-dynamic-subset.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

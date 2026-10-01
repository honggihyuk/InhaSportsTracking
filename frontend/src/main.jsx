import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
// 한글 UI 서체를 번들에 포함 (외부 CDN 없이 동작)
import 'pretendard/dist/web/variable/pretendardvariable-dynamic-subset.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

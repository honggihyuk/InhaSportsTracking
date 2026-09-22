import { useState, useEffect, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Text, Line } from '@react-three/drei'
import * as THREE from 'three'
import './index.css'

// 축구장 컴포넌트
function FootballField() {
  const fieldLength = 105
  const fieldWidth = 68
  
  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {/* 잔디 필드 */}
      <mesh position={[0, 0, 0]}>
        <planeGeometry args={[fieldLength, fieldWidth]} />
        <meshStandardMaterial color="#2d5a27" side={THREE.DoubleSide} />
      </mesh>
      
      {/* 라인 그리기 */}
      <Line
        points={[
          [-fieldLength/2, -fieldWidth/2, 0],
          [fieldLength/2, -fieldWidth/2, 0],
          [fieldLength/2, fieldWidth/2, 0],
          [-fieldLength/2, fieldWidth/2, 0],
          [-fieldLength/2, -fieldWidth/2, 0]
        ]}
        color="white"
        lineWidth={2}
      />
      
      {/* 중앙선 */}
      <Line
        points={[
          [0, -fieldWidth/2, 0],
          [0, fieldWidth/2, 0]
        ]}
        color="white"
        lineWidth={2}
      />
      
      {/* 중앙 서클 */}
      <mesh position={[0, 0, 0.01]} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[9.15, 9.35, 64]} />
        <meshBasicMaterial color="white" side={THREE.DoubleSide} />
      </mesh>
      
      {/* 페널티 박스 (간략화) */}
      <Line
        points={[
          [-fieldLength/2 + 16.5, -20.16, 0],
          [-fieldLength/2 + 16.5, 20.16, 0],
          [-fieldLength/2 + 57.5, 20.16, 0],
          [-fieldLength/2 + 57.5, -20.16, 0],
          [-fieldLength/2 + 16.5, -20.16, 0]
        ]}
        color="white"
        lineWidth={2}
      />
      
      <Line
        points={[
          [fieldLength/2 - 16.5, -20.16, 0],
          [fieldLength/2 - 16.5, 20.16, 0],
          [fieldLength/2 - 57.5, 20.16, 0],
          [fieldLength/2 - 57.5, -20.16, 0],
          [fieldLength/2 - 16.5, -20.16, 0]
        ]}
        color="white"
        lineWidth={2}
      />
    </group>
  )
}

// 선수 마커 컴포넌트
function PlayerMarker({ position, team, id }) {
  const color = team === 'home' ? '#ff4444' : '#4444ff'
  
  return (
    <group position={[position[0], 0, position[1]]}>
      <mesh position={[0, 1, 0]}>
        <sphereGeometry args={[0.8, 16, 16]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <Text
        position={[0, 2, 0]}
        fontSize={0.5}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {id}
      </Text>
    </group>
  )
}

// 공 마커 컴포넌트
function BallMarker({ position }) {
  return (
    <mesh position={[position[0], 0.5, position[1]]}>
      <sphereGeometry args={[0.4, 16, 16]} />
      <meshStandardMaterial color="white" />
    </mesh>
  )
}

// 메인 3D 씬 컴포넌트
function Scene({ players, ball }) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      
      <FootballField />
      
      {/* 선수들 렌더링 */}
      {players.map((player) => (
        <PlayerMarker
          key={player.id}
          id={player.id}
          team={player.team}
          position={[player.x, player.y]}
        />
      ))}
      
      {/* 공 렌더링 */}
      {ball && <BallMarker position={[ball.x, ball.z]} />}
      
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxPolarAngle={Math.PI / 2.1}
      />
    </>
  )
}

// 더미 데이터 생성 함수
const generateDummyData = () => {
  const players = []
  // 홈 팀 (11 명)
  for (let i = 1; i <= 11; i++) {
    players.push({
      id: i,
      team: 'home',
      x: (Math.random() - 0.5) * 80,
      y: (Math.random() - 0.5) * 50
    })
  }
  // 원정 팀 (11 명)
  for (let i = 12; i <= 22; i++) {
    players.push({
      id: i,
      team: 'away',
      x: (Math.random() - 0.5) * 80,
      y: (Math.random() - 0.5) * 50
    })
  }
  
  const ball = {
    x: (Math.random() - 0.5) * 100,
    z: (Math.random() - 0.5) * 60
  }
  
  return { players, ball }
}

function App() {
  const [viewMode, setViewMode] = useState('3d')
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [data, setData] = useState(generateDummyData())
  const animationRef = useRef()
  
  // 애니메이션 루프
  useEffect(() => {
    if (isPlaying) {
      const animate = () => {
        setCurrentTime(prev => prev + 0.1)
        setData(prevData => ({
          players: prevData.players.map(p => ({
            ...p,
            x: p.x + (Math.random() - 0.5) * 0.5,
            y: p.y + (Math.random() - 0.5) * 0.5
          })),
          ball: {
            x: prevData.ball.x + (Math.random() - 0.5) * 1,
            z: prevData.ball.z + (Math.random() - 0.5) * 1
          }
        }))
        animationRef.current = requestAnimationFrame(animate)
      }
      animationRef.current = requestAnimationFrame(animate)
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying])
  
  const handlePlayPause = () => setIsPlaying(!isPlaying)
  
  const handleReset = () => {
    setIsPlaying(false)
    setCurrentTime(0)
    setData(generateDummyData())
  }
  
  return (
    <div className="app-container">
      <header className="header">
        <h1>⚽ Soccer 3D Digital Twin</h1>
        <div className="header-controls">
          <button onClick={handlePlayPause} className="control-btn">
            {isPlaying ? '⏸️ 일시정지' : '▶️ 재생'}
          </button>
          <button onClick={handleReset} className="control-btn">
            🔄 리셋
          </button>
        </div>
      </header>
      
      <div className="main-content">
        <div className="viewer-container">
          <Canvas camera={{ position: [0, 50, 100], fov: 60 }}>
            <Scene players={data.players} ball={data.ball} />
          </Canvas>
          
          <div className="view-controls">
            <button 
              className={`view-btn ${viewMode === '3d' ? 'active' : ''}`}
              onClick={() => setViewMode('3d')}
            >
              🎥 3D 뷰
            </button>
            <button 
              className={`view-btn ${viewMode === 'topdown' ? 'active' : ''}`}
              onClick={() => setViewMode('topdown')}
            >
              📊 탑다운
            </button>
            <button 
              className={`view-btn ${viewMode === 'player' ? 'active' : ''}`}
              onClick={() => setViewMode('player')}
            >
              👤 선수 시점
            </button>
          </div>
        </div>
        
        <aside className="stats-panel">
          <h2>📈 실시간 통계</h2>
          <div className="stat-item">
            <span>경기 시간:</span>
            <strong>{currentTime.toFixed(1)}초</strong>
          </div>
          <div className="stat-item">
            <span>홈 팀 선수:</span>
            <strong>11 명</strong>
          </div>
          <div className="stat-item">
            <span>원정 팀 선수:</span>
            <strong>11 명</strong>
          </div>
          <div className="stat-item">
            <span>공 소유:</span>
            <strong>홈팀</strong>
          </div>
          
          <div className="upload-section">
            <h3>📁 비디오 업로드</h3>
            <input type="file" accept="video/*" className="file-input" />
            <p className="upload-hint">축구 경기 영상을 업로드하세요</p>
          </div>
        </aside>
      </div>
      
      <footer className="timeline-footer">
        <input 
          type="range" 
          min="0" 
          max="100" 
          value={(currentTime % 100)} 
          onChange={(e) => setCurrentTime(parseFloat(e.target.value))}
          className="timeline-slider"
        />
        <div className="timeline-info">
          <span>{currentTime.toFixed(1)}s / 100.0s</span>
        </div>
      </footer>
    </div>
  )
}

export default App

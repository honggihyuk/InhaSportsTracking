export function ViewControls({ viewMode, setViewMode, isPlaying, setIsPlaying, currentFrame, setCurrentFrame }) {
  const viewModes = [
    { id: '3d_free', label: '🎥 3D Free View', icon: '🎥' },
    { id: 'top_down', label: '📍 Top-Down', icon: '📍' },
    { id: 'broadcast', label: '📺 Broadcast', icon: '📺' },
    { id: 'player_cam', label: '👤 Player Cam', icon: '👤' }
  ];

  return (
    <div className="view-controls">
      <div className="view-mode-buttons">
        {viewModes.map((mode) => (
          <button
            key={mode.id}
            className={`view-mode-btn ${viewMode === mode.id ? 'active' : ''}`}
            onClick={() => setViewMode(mode.id)}
          >
            {mode.label}
          </button>
        ))}
      </div>

      <div className="playback-controls">
        <button 
          className="control-btn"
          onClick={() => setIsPlaying(!isPlaying)}
        >
          {isPlaying ? '⏸️ 일시정지' : '▶️ 재생'}
        </button>
        
        <input
          type="range"
          min="0"
          max="1000"
          value={currentFrame}
          onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
          className="timeline-slider"
        />
        
        <span className="frame-counter">Frame: {currentFrame}</span>
      </div>
    </div>
  );
}

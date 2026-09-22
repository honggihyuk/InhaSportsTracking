export function StatsPanel({ trackingData }) {
  if (!trackingData) {
    return (
      <div className="stats-panel">
        <h3>실시간 통계</h3>
        <p>데이터 로딩 중...</p>
      </div>
    );
  }

  const { players, ball, frame } = trackingData;

  // 팀별 통계 계산
  const homePlayers = players.filter(p => p.team === 'home');
  const awayPlayers = players.filter(p => p.team === 'away');

  const homeAvgVelocity = homePlayers.reduce((sum, p) => sum + p.velocity, 0) / homePlayers.length;
  const awayAvgVelocity = awayPlayers.reduce((sum, p) => sum + p.velocity, 0) / awayPlayers.length;

  const homeMaxVelocity = Math.max(...homePlayers.map(p => p.velocity));
  const awayMaxVelocity = Math.max(...awayPlayers.map(p => p.velocity));

  return (
    <div className="stats-panel">
      <h3>📊 실시간 통계</h3>
      
      <div className="stat-row">
        <span>프레임:</span>
        <span className="stat-value">{frame}</span>
      </div>

      <div className="stat-section">
        <h4>⚽ 공 정보</h4>
        <div className="stat-row">
          <span>속도:</span>
          <span className="stat-value">{ball?.velocity?.toFixed(1)} m/s</span>
        </div>
        <div className="stat-row">
          <span>위치:</span>
          <span className="stat-value">
            ({ball?.position_3d[0]?.toFixed(1)}, {ball?.position_3d[1]?.toFixed(1)})
          </span>
        </div>
      </div>

      <div className="stat-section">
        <h4>🔴 홈팀 (Home)</h4>
        <div className="stat-row">
          <span>선수 수:</span>
          <span className="stat-value">{homePlayers.length}명</span>
        </div>
        <div className="stat-row">
          <span>평균 속도:</span>
          <span className="stat-value">{homeAvgVelocity?.toFixed(1)} m/s</span>
        </div>
        <div className="stat-row">
          <span>최고 속도:</span>
          <span className="stat-value">{homeMaxVelocity?.toFixed(1)} m/s</span>
        </div>
      </div>

      <div className="stat-section">
        <h4>🔵 어웨이팀 (Away)</h4>
        <div className="stat-row">
          <span>선수 수:</span>
          <span className="stat-value">{awayPlayers.length}명</span>
        </div>
        <div className="stat-row">
          <span>평균 속도:</span>
          <span className="stat-value">{awayAvgVelocity?.toFixed(1)} m/s</span>
        </div>
        <div className="stat-row">
          <span>최고 속도:</span>
          <span className="stat-value">{awayMaxVelocity?.toFixed(1)} m/s</span>
        </div>
      </div>
    </div>
  );
}

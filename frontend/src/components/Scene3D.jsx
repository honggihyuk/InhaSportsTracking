import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei';
import { SoccerField } from './SoccerField';
import { PlayerMarker } from './PlayerMarker';
import { BallMarker } from './BallMarker';

export function Scene3D({ trackingData, viewMode = '3d_free' }) {
  // 카메라 초기 위치 설정
  const getCameraPosition = () => {
    switch (viewMode) {
      case 'top_down':
        return [0, 100, 0];
      case 'broadcast':
        return [0, 30, 80];
      case 'player_cam':
        return trackingData?.players?.[0]?.position_3d || [0, 5, 10];
      default: // 3d_free
        return [50, 30, 50];
    }
  };

  const getCameraTarget = () => {
    if (viewMode === 'player_cam' && trackingData?.players?.length > 0) {
      const playerPos = trackingData.players[0].position_3d;
      return playerPos;
    }
    return [0, 0, 0];
  };

  return (
    <Canvas style={{ width: '100%', height: '100%' }}>
      <PerspectiveCamera 
        makeDefault 
        position={getCameraPosition()} 
        fov={50}
      />
      <OrbitControls 
        target={getCameraTarget()}
        enableDamping
        dampingFactor={0.05}
      />
      
      {/* 환경 조명 */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 20, 10]} intensity={1} castShadow />
      <Environment preset="outdoor" />

      {/* 축구장 */}
      <SoccerField />

      {/* 선수들 렌더링 */}
      {trackingData?.players?.map((player) => (
        <PlayerMarker
          key={player.id}
          position={player.position_3d}
          team={player.team}
          id={player.id}
          velocity={player.velocity}
        />
      ))}

      {/* 공 렌더링 */}
      {trackingData?.ball && (
        <BallMarker
          position={trackingData.ball.position_3d}
          velocity={trackingData.ball.velocity}
        />
      )}
    </Canvas>
  );
}

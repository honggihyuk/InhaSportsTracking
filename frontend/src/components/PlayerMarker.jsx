import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export function PlayerMarker({ position, team, id, velocity }) {
  const markerRef = useRef();
  
  // 팀별 색상 설정
  const color = team === 'home' ? '#ff4444' : '#4444ff';
  
  // 애니메이션: 펄싱 효과
  useFrame((state) => {
    if (markerRef.current) {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
      markerRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <group ref={markerRef} position={position}>
      {/* 선수 마커 (원기둥) */}
      <mesh position={[0, 1, 0]} castShadow>
        <cylinderGeometry args={[0.5, 0.5, 2, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
      </mesh>
      
      {/* 번호 텍스트 (간단한 구체로 대체) */}
      <mesh position={[0, 2.5, 0]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
      
      {/* 속도 표시 (화살표) */}
      {velocity > 2 && (
        <mesh position={[0, 0.5, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.2, 0.5, 8]} />
          <meshStandardMaterial color="#ffff00" />
        </mesh>
      )}
    </group>
  );
}

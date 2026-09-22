import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export function BallMarker({ position, velocity }) {
  const ballRef = useRef();
  
  // 공 회전 애니메이션
  useFrame((state, delta) => {
    if (ballRef.current) {
      // 공의 속도에 따라 회전
      const rotationSpeed = velocity * 0.1;
      ballRef.current.rotation.x += rotationSpeed * delta;
      ballRef.current.rotation.z += rotationSpeed * delta;
    }
  });

  return (
    <group ref={ballRef} position={position}>
      {/* 축구공 (구체) */}
      <mesh castShadow>
        <sphereGeometry args={[0.22, 32, 32]} />
        <meshStandardMaterial 
          color="#ffffff" 
          emissive="#eeeeee"
          emissiveIntensity={0.2}
        />
      </mesh>
      
      {/* 공 패턴 (검은색 pentagon) */}
      {/* 간단한 표현을 위해 여러 개의 작은 구체 사용 */}
      <mesh position={[0.1, 0.1, 0.15]}>
        <dodecahedronGeometry args={[0.08, 0]} />
        <meshStandardMaterial color="#000000" />
      </mesh>
      <mesh position={[-0.1, -0.1, 0.15]}>
        <dodecahedronGeometry args={[0.08, 0]} />
        <meshStandardMaterial color="#000000" />
      </mesh>
      <mesh position={[0.1, -0.1, -0.15]}>
        <dodecahedronGeometry args={[0.08, 0]} />
        <meshStandardMaterial color="#000000" />
      </mesh>
      <mesh position={[-0.1, 0.1, -0.15]}>
        <dodecahedronGeometry args={[0.08, 0]} />
        <meshStandardMaterial color="#000000" />
      </mesh>
    </group>
  );
}

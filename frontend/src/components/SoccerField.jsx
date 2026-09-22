import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export function SoccerField() {
  const fieldRef = useRef();

  // 축구장 규격 (미터 기준)
  const fieldLength = 105; // 미터
  const fieldWidth = 68;   // 미터

  return (
    <group ref={fieldRef}>
      {/* 잔디 필드 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <planeGeometry args={[fieldLength, fieldWidth]} />
        <meshStandardMaterial color="#2d5a27" side={THREE.DoubleSide} />
      </mesh>

      {/* 라인 (흰색) */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(fieldLength, 0.2, fieldWidth)]} />
        <lineBasicMaterial color="#ffffff" linewidth={2} />
      </lineSegments>

      {/* 센터 서클 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
        <ringGeometry args={[9.15, 9.35, 64]} />
        <meshBasicMaterial color="#ffffff" side={THREE.DoubleSide} />
      </mesh>

      {/* 센터 스팟 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
        <circleGeometry args={[0.3, 32]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>

      {/* 골대 (왼쪽) */}
      <group position={[-fieldLength / 2, 0, 0]}>
        <mesh position={[0, 1.22, -3.66]}>
          <boxGeometry args={[0.12, 2.44, 0.12]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>
        <mesh position={[0, 2.44, 0]}>
          <boxGeometry args={[7.32, 0.12, 0.12]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>
        <mesh position={[0, 1.22, 3.66]}>
          <boxGeometry args={[0.12, 2.44, 0.12]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>
      </group>

      {/* 골대 (오른쪽) */}
      <group position={[fieldLength / 2, 0, 0]}>
        <mesh position={[0, 1.22, -3.66]}>
          <boxGeometry args={[0.12, 2.44, 0.12]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>
        <mesh position={[0, 2.44, 0]}>
          <boxGeometry args={[7.32, 0.12, 0.12]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>
        <mesh position={[0, 1.22, 3.66]}>
          <boxGeometry args={[0.12, 2.44, 0.12]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>
      </group>

      {/* 페널티 박스 (왼쪽) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-fieldLength / 2 + 20.5, 0.1, 0]}>
        <rectangleGeometry args={[40.32, 16.5]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.3} side={THREE.DoubleSide} />
      </mesh>

      {/* 페널티 박스 (오른쪽) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[fieldLength / 2 - 20.5, 0.1, 0]}>
        <rectangleGeometry args={[40.32, 16.5]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.3} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

// RectangleGeometry 헬퍼
class RectangleGeometry extends THREE.PlaneGeometry {
  constructor(width, height) {
    super(width, height);
  }
}

THREE.RectangleGeometry = RectangleGeometry;

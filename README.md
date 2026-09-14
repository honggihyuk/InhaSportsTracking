# Soccer 3D Digital Twin - Sports Analysis & Visualization Platform

## 프로젝트 개요
축구 영상 (2D) + 움직임 트랙킹 (Tracking/Homography) + 3D Gaussian Splatting (3DGS) & MLP 변형 기술을 통합하여 스포츠 분석 시각화 솔루션을 구축합니다.

## 핵심 아키텍처

### 기존 기술 Base
- **Roboflow Sports**: YOLO/ByteTrack/BoT-SORT 등으로 선수 및 공을 탐지/추적하고, 호모그래피 (Homography) 변환을 통해 2D 경기장 (Top-down view) 으로 위치를 매핑
- **GeoTrafficView-3D**: 2D CCTV 픽셀 좌표를 캘리브레이션/변환식을 통해 3D 지도 좌표계로 옮기고, 차선 및 객체 포즈를 바인딩하여 디지털 트윈 환경에 렌더링

### 확장 구상 (3D Gaussian Splatting + Dynamic MLP)
- 표준 공간 (Canonical Space) 의 3D 가우시안 (Gaussian) 을 정의
- 시간/트랙킹 데이터 (위치, 뼈대 키포인트 포즈 등) 를 입력받는 MLP(Deformation Network) 를 통해 표준 가우시안의 위치 (μ), 크기 (S), 회전 (R), 불투명도 (o) 를 변형 (Deform)
- 3D 공간 상에 자유로운 시점으로 축구 경기를 재현 (Volumetric Video/Free-viewpoint Rendering)

## 개발 파이프라인

### 1 단계: 데이터 수집 및 멀티뷰/트랙킹 파이프라인 구축
1. **영상 수집 & 카메라 캘리브레이션**
   - 경기장 다중 시점 (Multi-view) 카메라 영상 또는 방송용 카메라 영상 확보
   - 경기장 규격 라인 (Point-line correspondences) 기반 호모그래피 및 카메라 Extrinsics/Intrinsics 매매개변수 추정

2. **선수 및 공 2D/3D 트랙킹**
   - Roboflow Sports 파이프라인 적용: YOLOv8/v10 + ByteTrack 으로 선수 ID 유지
   - 2D Bounding Box 기반으로 3D World Coordinate(경기장 중심 원점 좌표계) 위치 좌표 추정
   - (고도화 시) RTMPose / OpenPose 등을 도입하여 선수의 3D Skeleton Keypoints(2D/3D 관절 위치) 추출

### 2 단계: 표준 공간 (Canonical Space) 가우시안 및 Deformation MLP 설계
1. **Canonical 3D Gaussian 정의**
   - 표준 자세 (예: T-pose 등) 를 가진 선수 및 경기장의 기준 3DGS 모델 구축

2. **Deformation Network (MLP) 구현**
   - 입력: 표준 가우시안 위치 x, 시간 t, 또는 프레임별 선수의 3D Pose/Translation Vector
   - 출력: 변형된 위치 Offset Δx, 회전 변형 Δr, 크기 변형 Δs
   - Deformable 3DGS (예: Deformable-GS, Dynamic3DGS 기술 참조) 구조 탑재

### 3 단계: 3D 공간 렌더링 & 시점 자유화 (Free-Viewpoint Engine)
1. **Real-time Gaussian Rasterization**
   - CUDA 기반 3DGS 실시간 렌더러 (Diff-Gaussian-Rasterization) 연동

2. **인터랙티브 시점제어 (Web/Desktop App UI)**
   - GeoTrafficView-3D 에서 구축하셨던 웹 기반 3D 렌더링 엔진 (deck.gl, Three.js, WebGL 등) 노하우를 활용
   - 자유로운 시점 전환 (전술 시점, 특정 선수 시점, 심판 시점 등) 기능 구현

### 4 단계: 정보 시각화 및 데이터 적재 (Sports Digital Twin UI)
1. **시각 정보 레이어 결합**
   - 선수별 이동 속도, 뛴 거리, 패스 줄기, 히트맵, 오프사이드 라인 등의 통계/분석 데이터를 3D 공간 상에 오버레이 레이어로 바인딩

2. **공간 DB 적재 및 질의 (LLM 연동)**
   - GeoTrafficView-3D 의 구조처럼 프레임/시간대별 데이터 (위치, 속도, 이벤트) 를 PostGIS/SQLite 공간 DB 에 적재
   - LLM 기반 데이터 질의 기능 구현 (예: "2 반 프레임에서 손흥민의 이동 거리는?", "가장 패스 성공률이 높은 영역은?")

## 기술적 가능성 및 타당성

| 평가 항목 | 가능성 | 상세 분석 및 해결 방안 |
| --- | --- | --- |
| **선수 트랙킹 및 좌표 매핑** | **매우 높음 (High)** | Roboflow Sports 및 GeoTrafficView-3D 프로젝트를 통해 호모그래피, 좌표 변환, ID 추적 노하우가 이미 입증됨 |
| **3DGS 기반 Dynamic 표현** | **보통~높음 (Medium-High)** | 최근 Deformable-GS, 4D Gaussian Splatting 연구가 활발하여 PyTorch/CUDA 오픈소스를 직관적으로 이식 가능. 단일 방송 카메라 화면만 사용할 경우 가려짐 (Occlusion) 영역에 대한 가우시안 복원이 한계가 있으므로, **Multi-view 영상** 확보 또는 **Template-guided 3D Mesh (예: SMPL) 연동 가우시안**을 활용하는 것이 시각적 품질을 높이는 열쇠 |
| **실시간성 (Real-time Rendering)** | **높음 (High)** | 3DGS 는 기존 NeRF 대비 렌더링 속도가 매우 빠르며 (100+ FPS 가능), MLP 가우시안 변형 역시 경량화된 MLP 를 쓰면 웹/엔드포인트 디바이스에서 수월하게 구동 가능 |
| **사용자 경험 (UX) 확장성** | **매우 높음 (High)** | 단순 2D 중계 영상을 넘어, 오프사이드 3D 판정 시점, 전술 감독 시점 (Bird's-eye view), 특정 선수 1 인칭 시점 등 압도적인 몰입감과 정보 전달력을 제공할 수 있음 |

## 핵심 제언

1. **SMPL (Human Body Model) + 3DGS 결합 고려**
   - 단순 MLP 로 t(시간) 에 따른 변형만 학습하면 동작이 복잡한 축구 동작에서 형상이 뭉개질 수 있음
   - 2D 영상에서 **SMPL 인체 3D 파라미터**를 먼저 뽑고, SMPL 메쉬 표면에 가우시안을 바인딩한 뒤 MLP 로 세부 변형을 주는 **Human-centric 3DGS (예: GauHuman, Animatable Gaussians)** 방식을 추천

2. **GeoTrafficView-3D 의 모듈 재활용**
   - 기존 프로젝트의 **[CCTV/영상 수집 → 객체 탐지 → 3D 공간 변환 → 공간 DB 적재 → LLM 질의]** 파이프라인 구조는 축구 경기 데이터 분석 플랫폼으로 100% 1:1 대응 가능

## 프로젝트 구조

```
/workspace
├── data/                    # 데이터 저장소
│   ├── raw/                 # 원본 영상 데이터
│   ├── processed/           # 처리된 데이터 (트랙킹 결과, 호모그래피 행렬 등)
│   └── models/              # 사전 학습된 모델 (YOLO, SMPL 등)
├── tracking/                # 트랙킹 파이프라인
│   ├── detector.py          # 객체 탐지 (YOLO)
│   ├── tracker.py           # 객체 추적 (ByteTrack/BoT-SORT)
│   ├── homography.py        # 호모그래피 변환
│   └── coordinate_mapper.py # 2D→3D 좌표 매핑
├── gs_model/                # 3D Gaussian Splatting 모델
│   ├── canonical_gs.py      # 표준 3D 가우시안 정의
│   ├── deformation_mlp.py   # 변형 MLP 네트워크
│   └── renderer.py          # 3DGS 렌더러
├── pipeline/                # 전체 파이프라인 통합
│   ├── data_loader.py       # 데이터 로딩 및 전처리
│   ├── processor.py         # 메인 처리 파이프라인
│   └── exporter.py          # 결과 내보내기
├── visualizer/              # 시각화 인터페이스
│   ├── src/                 # 프론트엔드 소스 (Three.js/deck.gl)
│   └── public/              # 정적 자산
├── configs/                 # 설정 파일
│   ├── camera_calibration.yaml
│   ├── field_dimensions.yaml
│   └── model_config.yaml
├── tests/                   # 테스트 코드
├── requirements.txt         # Python 의존성
├── package.json             # Node.js 의존성 (프론트엔드)
└── README.md                # 프로젝트 문서
```

## 설치 및 실행

### Python 백엔드
```bash
pip install -r requirements.txt
python pipeline/processor.py --config configs/model_config.yaml
```

### Web 프론트엔드
```bash
cd visualizer
npm install
npm run dev
```

## 라이선스
MIT License
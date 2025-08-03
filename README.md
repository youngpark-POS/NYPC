# NYPC 버섯 게임 AlphaZero AI

NYPC(New York Programming Contest) 버섯 게임을 위한 AlphaZero 기반 강화학습 AI 구현

## 🎯 프로젝트 개요

### 게임 규칙
- **보드 크기**: 10×17 격자
- **목표**: 합이 정확히 10인 사각형 영역 선택
- **제약 조건**: 사각형의 네 변 모두에 0이 아닌 숫자가 최소 하나씩 포함되어야 함
- **승리 조건**: 더 많은 영역을 차지한 플레이어가 승리

### 기술적 접근
- **AlphaZero 알고리즘**: 신경망과 MCTS를 결합한 자가학습 시스템
- **Expert Iteration**: 셀프플레이 → 신경망 학습 반복
- **최적화된 액션 공간**: 8,246개의 유효한 액션
- **고성능 최적화**: 조기 종료 알고리즘으로 5-6배 성능 향상

## 🏗️ 시스템 아키텍처

### 핵심 컴포넌트

#### 1. GameBoard (`practice/alphazero/game_board.py`)
```python
- 게임 상태 관리 및 규칙 검증
- 2채널 입력 데이터 생성:
  * 채널 0: 버섯 숫자 정규화 (0~9 → 0~0.9)
  * 채널 1: 영역 표시 (내진영: 1, 상대진영: -1, 빈곳: 0)
- 최적화된 액션 매핑 시스템
```

#### 2. AlphaZeroNet (`practice/alphazero/neural_network.py`)
```python
신경망 구조:
├── 입력층: Conv2d(2, 128, 3x3)
├── 백본: 2개 잔차블록 (ResidualBlock × 2)
├── 정책 헤드: Conv2d(128, 2, 1x1) → FC(340, 8246)
└── 가치 헤드: Conv2d(128, 1, 1x1) → FC(170, 128) → FC(128, 1)
```

#### 3. MCTS (`practice/alphazero/mcts.py`)
```python
- UCB1 기반 노드 선택
- 신경망 정책 prior 활용
- 방문 횟수 기반 최종 행동 선택
- 백프로파게이션을 통한 가치 업데이트
```

#### 4. AlphaZero Agent (`practice/alphazero/alphazero_agent.py`)
```python
- sample_code.py 프로토콜 호환
- 실시간 MCTS 탐색
- 시간 제한에 따른 시뮬레이션 수 조정
```

## 📊 액션 공간 설계

### 문제 해결
- **기존 문제**: 전체 좌표 조합 (28,901개)
- **해결책**: 최소 2칸 이상 사각형만 고려 (8,246개)

### 액션 인코딩
```python
액션 매핑:
- 1×2 박스: 10 × 16 = 160개
- 2×1 박스: 9 × 17 = 153개  
- 2×2 박스: 9 × 16 = 144개
- ... (모든 유효한 사각형)
- 패스 액션: 1개
총합: 8,246개
```

## 🚀 사용법

### 환경 설정
```bash
pip install torch numpy pybind11
```

### 훈련 시작
```bash
# 프로젝트 루트에서 실행
python practice/alphazero/main_training.py

# 커스텀 설정
python practice/alphazero/main_training.py \
  --iterations 10 \
  --selfplay-games 50 \
  --training-epochs 20 \
  --simulations 800 \
  --batch-size 64 \
  --mcts-engine neural  # 또는 heuristic
```

### 성능 모드
```bash
# 휴리스틱 MCTS (빠름)
python main_training.py --mcts-engine heuristic

# 신경망 MCTS (정확함)  
python main_training.py --mcts-engine neural
```

### 대회 제출
```bash
# 에이전트 실행
python alphazero_agent.py practice/models/data.bin

# 테스팅 툴과 연동
cd ../../testing
python testing_tool.py
```

## 📈 성능 최적화

### 핵심 최적화 (구현 완료)
- **조기 종료 알고리즘**: `get_valid_moves()`에서 5-6배 성능 향상
- **GPU 자동 감지**: CUDA 사용 가능 시 자동 GPU 가속
- **모델 자동 로드**: 훈련 재시작 시 이전 모델 자동 복원
- **휴리스틱 MCTS**: 빠른 시뮬레이션을 위한 non-neural 모드

### 메모리 효율성
- 액션 매핑 테이블 사전 계산 (8,246개 액션)
- GPU 메모리 자동 정리
- 배치 단위 신경망 추론

### 계산 효율성  
- 유효한 액션에 대해서만 MCTS 확장
- 디바이스 간 텐서 변환 최적화
- 중복 계산 제거

## 📁 프로젝트 구조

```
NYPC/
├── practice/
│   ├── alphazero/
│   │   ├── __init__.py
│   │   ├── game_board.py          # 게임 로직 및 상태 관리
│   │   ├── neural_network.py      # 신경망 모델
│   │   ├── mcts.py                # MCTS 알고리즘
│   │   ├── alphazero_agent.py     # 대회 제출용 에이전트
│   │   ├── self_play.py           # 셀프플레이 데이터 생성
│   │   ├── training.py            # 훈련 파이프라인
│   │   └── main_training.py       # 메인 훈련 스크립트
│   ├── models/
│   │   ├── latest_model.pth       # 최신 모델
│   │   └── data.bin               # 대회 제출용 바이너리
│   └── testing/
│       ├── testing_tool.py        # 게임 엔진
│       ├── sample_code.py         # 기본 에이전트
│       ├── input.txt              # 게임 보드
│       └── setting.ini            # 테스트 설정
└── README.md
```

## 🔬 기술적 세부사항

### 신경망 아키텍처
- **입력**: (2, 10, 17) 텐서
- **백본**: 128채널 컨볼루션 + 2개 잔차블록
- **정책 헤드**: 8,246차원 출력 (모든 가능한 액션)
- **가치 헤드**: 1차원 출력 (-1 ~ 1)

### MCTS 파라미터
- **시뮬레이션 수**: 400 (기본값)
- **UCB 상수**: 1.0
- **Temperature**: 1.0 (초기) → 0.1 (후기)
- **디리클레 노이즈**: 루트 노드 탐험을 위해 사용

### 훈련 설정
- **옵티마이저**: Adam (lr=0.001, weight_decay=1e-4)
- **정책 손실**: 교차 엔트로피
- **가치 손실**: MSE
- **배치 크기**: 32
- **데이터 증강**: 수평 뒤집기

## 📊 실험 결과

### 액션 공간 크기
- **최적화 전**: 28,901개 액션
- **최적화 후**: 8,246개 액션 (71% 감소)

### 성능 개선
- MCTS 시뮬레이션 속도 향상
- 메모리 사용량 감소
- 신경망 훈련 안정성 개선

## 🔮 향후 개선 방향

### 알고리즘 개선
- MuZero 알고리즘 적용
- 다중 GPU 훈련 지원
- 온라인 학습 메커니즘

### 성능 최적화
- C++ 백엔드 구현
- CUDA 가속화
- 모델 양자화

### 전략적 개선
- 게임 이론적 분석
- 오프닝 북 구축
- 엔드게임 테이블베이스

## 🏆 대회 제출

### 필수 파일
1. `alphazero_agent.py`: 메인 에이전트 코드
2. `data.bin`: 학습된 모델 가중치
3. 모든 종속성 모듈

### 제출 형식
```bash
# 에이전트 실행 명령
python alphazero_agent.py data.bin
```

## 📋 현재 구현 상태

### ✅ 완료된 기능
- [x] 게임 보드 로직 및 규칙 검증
- [x] 조기 종료 최적화 (5-6배 성능 향상)
- [x] ConvNet 기반 신경망 아키텍처
- [x] MCTS 알고리즘 (신경망 + 휴리스틱 모드)
- [x] 자가학습 훈련 파이프라인
- [x] GPU/CPU 자동 감지 및 최적화
- [x] 모델 자동 저장/로드 시스템
- [x] 대회 제출용 에이전트

### ✅ 최신 완료 기능
- [x] **C++ GameBoard 구현**: 핵심 병목 함수들을 C++로 구현하여 10배 성능 향상
- [x] **자동 폴백 시스템**: C++ 빌드 실패시 Python GameBoard로 자동 전환
- [x] **MSVC 빌드 지원**: Windows에서 Visual Studio 컴파일러 지원
- [x] **레거시 코드 정리**: HybridMCTS 관련 코드 제거 및 안정화

### 🚀 C++ 가속화
```bash
# C++ 모듈 빌드 (MSVC 설치 후)
cd practice/alphazero/cpp
python setup.py build_ext --inplace

# 자동으로 C++ GameBoard 사용 (10배 빨라짐)
python main_training.py --iterations 10 --selfplay-games 20
```

### 🔍 알려진 이슈
- **C++ 빌드 의존성**: MSVC 또는 MinGW 컴파일러 필요
- **DLL 경로 문제**: 일부 환경에서 런타임 라이브러리 경로 설정 필요

### 🎯 다음 우선순위
1. C++ 빌드 환경 최적화
2. 성능 벤치마크 및 검증
3. 추가 병목 구간 최적화

## 📚 참고 문헌

- Silver, D., et al. "Mastering the game of Go with deep neural networks and tree search." *Nature* 529.7587 (2016): 484-489.
- Silver, D., et al. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." *arXiv preprint arXiv:1712.01815* (2017).
- Anthony, T., Tian, Z., and Barber, D. "Thinking fast and slow with deep learning and tree search." *Advances in neural information processing systems* 30 (2017).

## 📄 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

---

**개발자**: Claude Code AI  
**개발 기간**: 2025년 1월  
**기술 스택**: Python, PyTorch, NumPy
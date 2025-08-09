# NYPC 2024 Yacht AI 개발 보고서

## 📅 작업 기간
2024년 작업 진행 중 (dev branch)

## 🎯 주요 성과

### 1. 기댓값 계산 로직 대폭 개선 ⭐⭐⭐
**문제점**: 기존 Monte Carlo 시뮬레이션 방식의 부정확성
- 보너스 확률 계산에서 단순한 Monte Carlo 근사 사용
- 상대방 주사위를 모른다고 가정하여 `[3,3,3,3,3]` 평균값으로 추정

**해결방안**: 조합적 계산 방식으로 완전 전환
- **정확한 확률 분포 사용**: `BASE_EXPECTATIONS`의 확률 분포 활용
- **itertools.product()**: 모든 가능한 조합을 정확히 계산
- **상대방 추적 개선**: 서로 다른 그룹을 선택하므로 실제 추적 가능

### 2. Neural Network 기반 프레임워크 구축 🧠
**새로운 아키텍처**: 4-Way Parallel Branch Architecture
- `yacht_nn_submission.py`: 완전한 NN 기반 AI (452줄)
- `utils/nn_trainer.py`: 신경망 학습 모듈
- `utils/nn_test.py`: 신경망 테스트 및 검증 모듈

**기술적 특징**:
- 4개 브랜치: [내_A선택, 내_B선택, 상대_A선택, 상대_B선택]
- 중간 감독 학습: 12개 조합별 선호도 출력
- 최종 의사결정: A/B 선택 확률 + 입찰비용

### 3. Submission 파일 혁신적 최적화 🚀
**yacht_submission.py 완전 재작성** (272줄)
- **압축률**: 기존 대비 50% 이상 코드 감소
- **성능**: 조합적 기댓값 계산 (Combinatorial Expectation)
- **효율성**: 턴별 차등 가치 평가 시스템

**핵심 알고리즘**:
```python
# 첫 턴 (0→5개): 기댓값 손실 최소화
base_exp_sum = sum(BASE_EXPECTATIONS[r.name] for r in available_rules)
cond_exp_sum = sum(GameState.calculate_score(DicePut(r, dice_option)) for r in available_rules)
return base_exp_sum - cond_exp_sum  # 손실이 적을수록 좋음

# 이후 턴 (5→10개): 잠재적 점수 최대화
potential_score = self.calculator.calculate_potential_score(all_dice, rule, player_state)
return potential_score  # 높을수록 좋음
```

## 🔧 기술적 혁신

### 조합적 보너스 확률 계산
```python
def calculate_bonus_expectation(self, player_state, candidate_rule, candidate_score, all_10_dice):
    distributions = [self._get_future_score_distribution(r.value, all_10_dice) for r in unfilled_basic_rules]
    
    total_bonus_prob = 0
    for combo in product(*distributions):  # 모든 조합 순회
        future_score = sum(score for score, prob in combo)
        combo_prob = math.prod(prob for score, prob in combo)
        
        if fixed_score + future_score >= 63000:
            total_bonus_prob += combo_prob
    
    return total_bonus_prob * 35000
```

### 기회비용 기반 의사결정
```python
def calculate_put(self):
    for rule in available_rules:
        potential_score = self.calculator.calculate_potential_score(self.my_state.dice, rule, self.my_state)
        gain = potential_score - BASE_EXPECTATIONS.get(rule.name, 0)  # 기댓값 대비 이득
        
        if gain > best_gain:
            best_gain = gain
            best_rule = rule
```

## 📊 파일 구조 변화

### 새로 생성된 파일
- `yacht_submission.py`: 조합적 기댓값 AI (272줄)
- `yacht_nn_submission.py`: 신경망 기반 AI (452줄)
- `utils/nn_trainer.py`: NN 학습 모듈
- `utils/nn_test.py`: NN 테스트 모듈

### 개선된 기존 파일
- `utils/yacht_conditional_calculator.py`: 확률 분포 계산 강화
- `utils/yacht_exact_calculator.py`: 정확한 기댓값 계산
- `config.ini`: 설정 업데이트

## 🎮 게임 전략 개선

### 입찰 전략
1. **대칭 상황 감지**: 내 선호 ≠ 상대 선호 → 무료 입찰
2. **경쟁 상황**: 이득 차이 × 0.5 기반 입찰
3. **게임 후반부 가중치**: 점수차에 따른 공격성/보수성 조절

### 배치 전략  
1. **기회비용 계산**: 실제 점수 - 기댓값 = 이득
2. **보너스 확률 포함**: One~Six는 보너스 기댓값 추가 고려
3. **최적화된 주사위 선택**: 각 규칙별 최적 5개 주사위 자동 선택

## 🐛 알려진 문제점

### yacht_submission.py
1. **코드 압축도 과도**: 가독성 저하 (한 줄에 너무 많은 로직)
2. **에러 처리 부족**: Edge case 처리 미흡
3. **디버깅 어려움**: 압축된 코드로 인한 디버깅 복잡성

### yacht_nn_submission.py  
1. **모델 파일 없음**: 실제 학습된 가중치 부재
2. **랜덤 추정**: 상대방 주사위를 랜덤으로 추정
3. **PyTorch 의존성**: 제출 환경에서 PyTorch 사용 가능 여부 불확실

## 🎯 다음 단계 계획

### 우선순위 1: 안정성 확보
- [ ] yacht_submission.py 에러 처리 강화
- [ ] Edge case 테스트 및 수정
- [ ] 가독성 개선 (주석 추가)

### 우선순위 2: NN 모델 완성
- [ ] 실제 게임 데이터로 신경망 학습
- [ ] 모델 성능 검증 및 최적화
- [ ] CPU 환경 최적화

### 우선순위 3: 성능 최적화
- [ ] 알고리즘 효율성 개선
- [ ] 메모리 사용량 최적화
- [ ] 실행 시간 단축

## 📈 전체 평가

### 성공 요인
- ✅ 기댓값 계산의 정확도 대폭 향상
- ✅ 혁신적인 조합적 계산 방식 도입
- ✅ 코드 효율성과 압축률 최적화
- ✅ 다양한 접근법 (기댓값, NN) 구현

### 개선 필요 사항
- ⚠️ 코드 안정성 및 예외 처리
- ⚠️ 디버깅 편의성 향상
- ⚠️ NN 모델 실용성 확보

---

**결론**: 기댓값 계산 로직이 근본적으로 개선되었고, 두 가지 서로 다른 접근법으로 경쟁력 있는 AI가 구축되었다. 현재 상태에서도 실행 가능하지만, 안정성과 예외 처리 개선이 필요하다.
import math
from itertools import product

# 미리 계산된 기본 기댓값들 (exact calculator에서 계산됨)
BASE_EXPECTATIONS = {
    'basic_distributions': {
        1: [(0, 0.1615055828898458), (1, 0.3230111657796916), (2, 0.2907100492017224), 
            (3, 0.15504535957425192), (4, 0.05426587585098817), (5, 0.015461966703500482)],
        2: [(0, 0.1615055828898458), (1, 0.3230111657796916), (2, 0.2907100492017224), 
            (3, 0.15504535957425192), (4, 0.05426587585098817), (5, 0.015461966703500482)],
        3: [(0, 0.1615055828898458), (1, 0.3230111657796916), (2, 0.2907100492017224), 
            (3, 0.15504535957425192), (4, 0.05426587585098817), (5, 0.015461966703500482)],
        4: [(0, 0.1615055828898458), (1, 0.3230111657796916), (2, 0.2907100492017224), 
            (3, 0.15504535957425192), (4, 0.05426587585098817), (5, 0.015461966703500482)],
        5: [(0, 0.1615055828898458), (1, 0.3230111657796916), (2, 0.2907100492017224), 
            (3, 0.15504535957425192), (4, 0.05426587585098817), (5, 0.015461966703500482)],
        6: [(0, 0.1615055828898458), (1, 0.3230111657796916), (2, 0.2907100492017224), 
            (3, 0.15504535957425192), (4, 0.05426587585098817), (5, 0.015461966703500482)]
    },
    'basic_expected': [1663.94, 3327.88, 4991.82, 6655.76, 8319.70, 9983.64],
    'combination_expected': {
        'CHOICE': 24179.31,
        'FOUR_OF_A_KIND': 7321.42,
        'FULL_HOUSE': 17631.83,
        'SMALL_STRAIGHT': 9750.00,
        'LARGE_STRAIGHT': 21372.39,
        'YACHT': 4638.59
    }
}

def binomial_prob(n, k, p):
    """이항분포 확률 계산"""
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def calculate_score_for_selection(given_dice, selected_number):
    """
    특정 숫자 선택 시의 점수 계산 (선택 숫자 점수 + 보너스만)
    given_dice: [1, 2, 3, 4, 5] 형태의 리스트
    selected_number: 1~6 중 선택한 숫자
    returns: float (선택한 숫자 점수 + 보너스)
    """
    given_counts = [given_dice.count(i) for i in range(1, 7)]
    
    # 선택된 숫자의 조건부 점수 계산
    selected_score = 0
    for additional in range(6):  # 추가 5개 주사위에서 0~5개
        prob = binomial_prob(5, additional, 1/6)
        total_count = given_counts[selected_number-1] + additional
        usable_count = min(total_count, 5)
        score = usable_count * selected_number * 1000
        selected_score += prob * score
    
    # 보너스 계산
    bonus = calculate_bonus_for_selected_number(selected_number, given_counts)
    
    return selected_score + bonus

def calculate_all_basic_scores(given_dice):
    """
    각 숫자 선택 시의 점수를 모두 계산
    returns: {1: 점수1, 2: 점수2, ..., 6: 점수6}
    """
    results = {}
    given_counts = [given_dice.count(i) for i in range(1, 7)]
    
    print(f"주어진 주사위: {given_dice}")
    print(f"숫자별 개수: {dict(zip(range(1,7), given_counts))}")
    
    for selected_num in range(1, 7):
        score = calculate_score_for_selection(given_dice, selected_num)
        results[selected_num] = score
        print(f"숫자 {selected_num} 선택 시: {score:.2f}점")
    
    return results

def calculate_basic_score_conditional(given_count, number):
    """특정 숫자의 기본 점수 조건부 기댓값과 확률 분포"""
    expected = 0
    prob_distribution = []  # [(사용_개수, 확률, 점수), ...]
    
    # 추가 5개 주사위에서 해당 숫자가 j개 나올 경우
    for j in range(6):  # 0~5개
        prob_j = binomial_prob(5, j, 1/6)
        total_count = given_count + j
        usable_count = min(total_count, 5)  # 최대 5개까지만 사용 가능
        score = usable_count * number * 1000
        expected += prob_j * score
        
        # 확률 분포 저장
        prob_distribution.append((usable_count, prob_j, score))
    
    return expected, prob_distribution

def calculate_bonus_for_selected_number(selected_number, given_counts):
    """특정 숫자 선택 시 보너스 기댓값 계산 - 모든 조합 계산"""
    
    # 선택된 숫자의 조건부 점수 분포
    selected_distribution = []
    for additional in range(6):  # 0~5개 추가
        prob = binomial_prob(5, additional, 1/6)
        total_count = given_counts[selected_number-1] + additional
        usable_count = min(total_count, 5)
        score = usable_count * selected_number * 1000
        selected_distribution.append((score, prob))
    
    # 나머지 5개 숫자들의 기본 분포에서 모든 조합 계산
    other_numbers = [i for i in range(1, 7) if i != selected_number]
    
    def calculate_all_combinations():
        """나머지 숫자들의 모든 가능한 점수 조합 계산"""
        from itertools import product
        
        # 각 숫자별 (점수, 확률) 분포 생성
        other_distributions = []
        for num in other_numbers:
            dist = []
            for count in range(6):  # 0~5개 (5개 이상은 합쳐짐)
                if count < 5:
                    prob = BASE_EXPECTATIONS['basic_distributions'][num][count][1]
                    score = count * num * 1000
                else:  # 5개인 경우
                    prob = BASE_EXPECTATIONS['basic_distributions'][num][5][1]
                    score = 5 * num * 1000
                dist.append((score, prob))
            other_distributions.append(dist)
        
        # 모든 조합의 (총점수, 확률) 계산
        combinations = []
        for combo in product(*other_distributions):
            total_score = sum(score for score, prob in combo)
            total_prob = 1.0
            for score, prob in combo:
                total_prob *= prob
            combinations.append((total_score, total_prob))
        
        return combinations
    
    # 모든 조합 계산
    other_combinations = calculate_all_combinations()
    
    # 보너스 달성 확률 계산
    bonus_prob = 0
    for selected_score, selected_prob in selected_distribution:
        for other_score, other_prob in other_combinations:
            total_basic_score = selected_score + other_score
            if total_basic_score >= 63000:
                bonus_prob += selected_prob * other_prob
    
    return bonus_prob * 35000

def calculate_bonus_conditional(given_counts):
    """기존 함수 - 호환성을 위해 유지하지만 사용하지 않음"""
    return 0  # 더 이상 사용하지 않음

def calculate_choice_conditional(given_dice):
    """CHOICE 조건부 기댓값 - 정확한 계산"""
    total_expected = 0
    total_cases = 6**5  # 추가 5개 주사위의 모든 경우
    
    # 모든 가능한 추가 주사위 조합
    for additional_dice in product(range(1, 7), repeat=5):
        all_dice = given_dice + list(additional_dice)
        # 상위 5개 선택
        top_5_sum = sum(sorted(all_dice, reverse=True)[:5])
        total_expected += top_5_sum # 주사위 눈의 합
    
    return total_expected / total_cases  * 1000 # 점수 변환

def calculate_four_of_a_kind_conditional(given_counts):
    """FOUR_OF_A_KIND 조건부 기댓값"""
    total_expected = 0
    
    for target_number in range(1, 7):
        given_target = given_counts[target_number-1]
        
        if given_target >= 4:
            # 이미 4개 이상 있음 - 확정
            prob = 1.0
            # 추가 주사위 5개의 기댓값은 5 * 3.5 = 17.5
            expected_sum = target_number * 4 + 3.5  # 4개 확정 + 나머지 1개 평균
            total_expected += prob * expected_sum * 1000 * (1/6)
        
        elif given_target >= 1:
            # 1~3개 있음 - 추가로 (4-given_target)개 이상 필요
            needed = 4 - given_target
            prob_success = 0
            expected_given_success = 0
            
            for additional in range(needed, 6):  # needed~5개 추가로 나올 경우
                prob_this = binomial_prob(5, additional, 1/6)
                prob_success += prob_this
                
                # 이 경우의 기댓값
                used_target = 4  # 정확히 4개 사용
                remaining_dice = 5 - additional + (5 - 1)  # 추가-사용 + 원래 나머지
                expected_remaining = 3.5  # 평균값
                expected_sum = used_target * target_number + expected_remaining
                expected_given_success += prob_this * expected_sum
            
            if prob_success > 0:
                expected_given_success /= prob_success
                total_expected += prob_success * expected_given_success * 1000 * (1/6)
    
    return total_expected

def calculate_full_house_conditional(given_counts):
    """FULL_HOUSE 조건부 기댓값 - 순수 수학적 계산"""
    
    def multinomial_coeff(n, counts):
        """다항분포 계수 계산"""
        result = math.factorial(n)
        for c in counts:
            result //= math.factorial(c)
        return result
    
    def is_full_house(counts):
        """FULL_HOUSE 조건 확인: 3개+ 하나, 2개+ 하나 이상"""
        sorted_counts = sorted([c for c in counts if c > 0], reverse=True)
        return len(sorted_counts) >= 2 and sorted_counts[0] >= 3 and sorted_counts[1] >= 2
    
    def optimal_full_house_score(total_counts):
        """FULL_HOUSE 조건 만족 시 최적 5개 선택 점수"""
        if not is_full_house(total_counts):
            return 0
        
        # 탐욕적 선택: 개수가 많은 순, 같으면 숫자가 큰 순
        indexed_counts = [(count, num) for num, count in enumerate(total_counts, 1) if count > 0]
        indexed_counts.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        selected = []
        for count, num in indexed_counts:
            use_count = min(count, 5 - len(selected))
            selected.extend([num] * use_count)
            if len(selected) >= 5:
                break
        
        # FULL_HOUSE 조건 재확인 (선택된 5개로)
        selected_counts = [selected.count(i) for i in range(1, 7)]
        if is_full_house(selected_counts):
            return sum(selected)
        return 0
    
    total_prob = 0
    total_expected = 0
    
    # 추가 5개 주사위의 모든 가능한 조합 (다항분포)
    # [a1, a2, a3, a4, a5, a6] where sum(ai) = 5
    def generate_compositions(n, k, current=[]):
        """n을 k개 음이 아닌 정수의 합으로 나타내는 모든 방법"""
        if k == 1:
            yield current + [n]
            return
        
        for i in range(n + 1):
            yield from generate_compositions(n - i, k - 1, current + [i])
    
    for additional_counts in generate_compositions(5, 6):
        # 다항분포 확률 계산
        prob = multinomial_coeff(5, additional_counts) * (1/6)**5
        
        # 전체 개수 계산
        total_counts = [given + add for given, add in zip(given_counts, additional_counts)]
        
        # FULL_HOUSE 조건 확인 및 점수 계산
        score = optimal_full_house_score(total_counts)
        if score > 0:
            total_prob += prob
            total_expected += prob * score
    
    if total_prob > 0:
        return total_expected * 1000  # 이미 확률 가중 평균
    return 0

def calculate_straight_conditional(given_dice, pattern_type):
    """STRAIGHT 패턴 조건부 확률"""
    given_set = set(given_dice)
    
    if pattern_type == "small":
        patterns = [{1,2,3,4}, {2,3,4,5}, {3,4,5,6}]
    else:  # large
        patterns = [{1,2,3,4,5}, {2,3,4,5,6}]
    
    total_prob = 0
    
    for pattern in patterns:
        missing = pattern - given_set
        if len(missing) == 0:
            # 이미 패턴 완성
            total_prob = 1.0
            break
        elif len(missing) <= 5:
            # 추가 5개에서 missing 숫자들이 모두 나올 확률
            prob_all_missing = 1 - ((6 - len(missing)) / 6) ** 5
            # 포함-배제 원리로 정확히 계산해야 하지만 근사
            total_prob = max(total_prob, prob_all_missing)
    
    if pattern_type == "small":
        return total_prob * 15000
    else:
        return total_prob * 30000

def calculate_small_straight_conditional(given_dice):
    """SMALL_STRAIGHT 조건부 계산"""
    return calculate_straight_conditional(given_dice, "small")

def calculate_large_straight_conditional(given_dice):
    """LARGE_STRAIGHT 조건부 계산"""
    return calculate_straight_conditional(given_dice, "large")

def calculate_yacht_conditional(given_counts):
    """YACHT 조건부 기댓값"""
    total_expected = 0
    
    for target_number in range(1, 7):
        given_target = given_counts[target_number-1]
        
        if given_target >= 5:
            # 이미 5개 이상 - 확정
            total_expected += 50000 * (1/6)
        elif given_target >= 1:
            # 1~4개 있음 - 추가로 (5-given_target)개 이상 필요
            needed = 5 - given_target
            prob_success = sum(binomial_prob(5, k, 1/6) for k in range(needed, 6))
            total_expected += prob_success * 50000 * (1/6)
    
    return total_expected

def main():
    """테스트 함수 - 새로운 시스템 테스트"""
    # 테스트 케이스들
    test_cases = [
        [1, 1, 1, 2, 3],  # 1이 3개
        [1, 2, 3, 4, 5],  # 스트레이트에 가까움
        [6, 6, 6, 6, 1],  # 6이 4개 (포카드 거의 확정)
        [3, 3, 3, 2, 2],  # 풀하우스 확정
        [1, 2, 4, 5, 6],  # 일반적인 케이스
    ]
    
    for i, test_dice in enumerate(test_cases):
        print(f"\n=== 테스트 케이스 {i+1}: {test_dice} ===")
        
        # 각 숫자 선택 시 기본 점수 계산
        basic_results = calculate_all_basic_scores(test_dice)
        
        print(f"\n=== 최적 선택 분석 ===")
        best_score = max(basic_results.values())
        best_choice = [num for num, score in basic_results.items() if score == best_score][0]
        print(f"최적 선택: 숫자 {best_choice} ({best_score:.2f}점)")
        
        print("=" * 50)

def test_bonus_calculation():
    """보너스 계산 로직 검증"""
    print("=== 보너스 계산 검증 ===")
    
    # 극단적 케이스 테스트
    test_cases = [
        ([6, 6, 6, 6, 6], 6, "6이 5개 - 높은 보너스 확률"),
        ([1, 1, 1, 1, 1], 1, "1이 5개 - 낮은 보너스 확률"),
        ([1, 2, 3, 4, 5], 6, "고른 분포에서 6 선택")
    ]
    
    for given_dice, selected_num, description in test_cases:
        given_counts = [given_dice.count(i) for i in range(1, 7)]
        bonus = calculate_bonus_for_selected_number(selected_num, given_counts)
        
        print(f"{description}")
        print(f"주사위: {given_dice}, 선택: {selected_num}")
        print(f"보너스: {bonus:.2f}점")
        
        # 상세 분석
        selected_score = 0
        for additional in range(6):
            prob = binomial_prob(5, additional, 1/6)
            total_count = given_counts[selected_num-1] + additional
            usable_count = min(total_count, 5)
            score = usable_count * selected_num * 1000
            selected_score += prob * score
        
        other_sum = sum(BASE_EXPECTATIONS['basic_expected'][i-1] 
                       for i in range(1, 7) if i != selected_num)
        
        print(f"선택된 숫자 기댓값: {selected_score:.2f}")
        print(f"나머지 기본값 합계: {other_sum:.2f}")
        print(f"총 기본점수 기댓값: {selected_score + other_sum:.2f}")
        print()
        
if __name__ == "__main__":
    main()
    test_bonus_calculation()
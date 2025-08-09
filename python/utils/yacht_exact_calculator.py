import math
from fractions import Fraction

def binomial_prob(n, k, p):
    """이항분포 확률 계산"""
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def max_count_distribution():
    """10개 주사위에서 특정 숫자가 n개 나올 확률 분포 (기존 함수)"""
    return {k: binomial_prob(10, k, 1/6) for k in range(11)}

def calculate_basic_distributions():
    """기본 점수 1~6의 확률 분포 (conditional과 동일한 형식)"""
    count_dist = max_count_distribution()
    
    basic_distributions = {}
    basic_expected = []
    
    for number in range(1, 7):
        expected = 0
        prob_distribution = []
        
        # 0~4개는 개별적으로, 5개 이상은 합쳐서 처리
        for count in range(6):  # 0~5
            if count < 5:
                prob = count_dist[count]
                usable_count = count
            else:  # count == 5 (5개 이상 합치기)
                prob = sum(count_dist[k] for k in range(5, 11))  # 5~10개 확률 합
                usable_count = 5
            
            score = usable_count * number * 1000
            expected += score * prob
            prob_distribution.append((usable_count, prob))
        
        basic_distributions[number] = prob_distribution
        basic_expected.append(expected)
    
    return basic_distributions, basic_expected

def calculate_bonus_for_single_number(selected_number):
    """단일 숫자 선택 시 보너스 기댓값 계산 - 올바른 6^6 방식"""
    from itertools import product
    
    # 기본 분포에서 직접 사용 (0~5개만, 5개 이상은 이미 합쳐짐)
    basic_distributions, _ = calculate_basic_distributions()
    
    # 선택된 숫자의 점수 분포 (0~5개 사용)
    selected_distribution = []
    for count, prob in basic_distributions[selected_number]:
        score = count * selected_number * 1000
        selected_distribution.append((score, prob))
    
    # 나머지 5개 숫자들의 점수 분포 (각각 0~5개 사용)
    other_numbers = [i for i in range(1, 7) if i != selected_number]
    other_distributions = []
    
    for num in other_numbers:
        dist = []
        for count, prob in basic_distributions[num]:
            score = count * num * 1000
            dist.append((score, prob))
        other_distributions.append(dist)
    
    # 보너스 달성 확률 계산
    bonus_prob = 0
    total_combinations = 0
    
    for selected_score, selected_prob in selected_distribution:
        for combo in product(*other_distributions):
            other_score = sum(score for score, prob in combo)
            other_prob = 1.0
            for score, prob in combo:
                other_prob *= prob
            
            total_basic_score = selected_score + other_score
            combination_prob = selected_prob * other_prob
            
            if total_basic_score >= 63000:
                bonus_prob += combination_prob
            
            total_combinations += 1
    
    bonus_value = bonus_prob * 35000
    return bonus_value, bonus_prob, total_combinations

def calculate_bonus_exact():
    """보너스 기댓값 계산 - 대칭성을 이용한 단순화"""
    print("보너스 계산 중 (대칭성 이용)...")
    
    # 숫자 1만 계산 (모든 숫자가 동일한 보너스를 가지므로)
    bonus_value, bonus_prob, total_combinations = calculate_bonus_for_single_number(1)
    
    print(f"  보너스 확률: {bonus_prob:.6f}")
    print(f"  보너스 기댓값: {bonus_value:.2f}")
    print(f"  총 조합 수: {total_combinations:,}")
    print("  대칭성으로 인해 모든 숫자(1~6)가 동일한 보너스를 가집니다.")
    
    # 6개 숫자 모두 동일한 값 반환
    return [bonus_value] * 6

def calculate_choice_exact():
    """CHOICE: 10개 중 상위 5개 합의 정확한 기댓값"""
    from itertools import product
    
    # 직접 계산: 모든 가능한 주사위 조합에서 상위 5개의 평균
    total_sum = 0
    total_cases = 6**10  # 모든 가능한 경우의 수
    
    # 메모리 효율을 위해 계산을 분할 (6^10은 너무 큼)
    # 대신 순서통계량의 기댓값 공식 사용
    
    def expected_kth_order_statistic_exact(k, n):
        """n개 이산 균등분포(1~6)에서 k번째 순서통계량의 기댓값"""
        # 베타 함수 기반 공식 사용
        expected = 0
        for value in range(1, 7):
            # P(X_(k:n) = value) 계산
            prob = 0
            
            # value보다 작은 값이 k-1개 미만, value가 적어도 1개, value보다 큰 값이 n-k개 미만
            for less_count in range(k):  # 0 ~ k-1개
                for greater_count in range(n-k+1):  # 0 ~ n-k개
                    equal_count = n - less_count - greater_count
                    if equal_count > 0:  # 적어도 1개는 value와 같아야 함
                        # 다항분포 확률
                        coeff = math.factorial(n) // (math.factorial(less_count) * math.factorial(equal_count) * math.factorial(greater_count))
                        prob_this = coeff * ((value-1)/6)**less_count * (1/6)**equal_count * ((6-value)/6)**greater_count
                        prob += prob_this
            
            expected += value * prob
        
        return expected
    
    # 상위 5개 (6등~10등)의 기댓값 합계
    expected_top_5 = sum(expected_kth_order_statistic_exact(k, 10) for k in range(6, 11))
    
    return expected_top_5 * 1000


def calculate_four_of_a_kind_exact():
    """FOUR_OF_A_KIND: 다항분포 방식으로 정확한 계산"""
    
    def multinomial_coeff(n, counts):
        """다항분포 계수 계산"""
        result = math.factorial(n)
        for c in counts:
            result //= math.factorial(c)
        return result
    
    def generate_compositions(n, k, current=[]):
        """n을 k개 음이 아닌 정수의 합으로 나타내는 모든 방법"""
        if k == 1:
            yield current + [n]
            return
        
        for i in range(n + 1):
            yield from generate_compositions(n - i, k - 1, current + [i])
    
    def find_best_four_of_a_kind(counts):
        """주어진 주사위 개수에서 최적의 포카드 조합 찾기"""
        best_score = 0
        
        # 각 숫자별로 포카드 가능성 확인
        for target_num in range(1, 7):
            if counts[target_num-1] >= 4:
                # 포카드 가능 - 해당 숫자 4개 + 나머지 중 가장 큰 1개
                remaining_dice = []
                for num in range(1, 7):
                    count = counts[num-1]
                    if num == target_num:
                        remaining_dice.extend([num] * (count - 4))  # 4개 제외하고 추가
                    else:
                        remaining_dice.extend([num] * count)
                
                # 나머지 중 가장 큰 1개 선택
                if remaining_dice:
                    largest_remaining = max(remaining_dice)
                    score = target_num * 4 + largest_remaining
                else:
                    score = target_num * 4  # 나머지가 없으면 4개만
                
                best_score = max(best_score, score)
        
        return best_score
    
    total_expected = 0
    
    # 10개 주사위의 모든 가능한 조합
    for counts in generate_compositions(10, 6):
        # 다항분포 확률
        prob = multinomial_coeff(10, counts) * (1/6)**10
        
        # 이 조합에서 최적의 포카드 점수 찾기
        best_score = find_best_four_of_a_kind(counts)
        
        if best_score > 0:
            total_expected += prob * best_score * 1000
    
    return total_expected

def calculate_yacht_exact():
    """YACHT: 다항분포 방식으로 정확한 확률 계산"""
    
    def multinomial_coeff(n, counts):
        """다항분포 계수 계산"""
        result = math.factorial(n)
        for c in counts:
            result //= math.factorial(c)
        return result
    
    def generate_compositions(n, k, current=[]):
        """n을 k개 음이 아닌 정수의 합으로 나타내는 모든 방법"""
        if k == 1:
            yield current + [n]
            return
        
        for i in range(n + 1):
            yield from generate_compositions(n - i, k - 1, current + [i])
    
    def has_yacht(counts):
        """YACHT 조건 확인 (5개 이상 같은 숫자)"""
        return any(count >= 5 for count in counts)
    
    total_expected = 0
    
    # 10개 주사위의 모든 가능한 조합
    for counts in generate_compositions(10, 6):
        # 다항분포 확률
        prob = multinomial_coeff(10, counts) * (1/6)**10
        
        # YACHT 조건 확인
        if has_yacht(counts):
            total_expected += prob * 50000
    
    return total_expected

def calculate_straight_patterns_exact():
    """SMALL/LARGE STRAIGHT 포함-배제 원리로 정확한 확률 계산"""
    
    def prob_contains_all(required_numbers):
        """10개 주사위에서 required_numbers가 모두 포함될 확률 (포함-배제 원리)"""
        n = len(required_numbers)
        total_prob = 0
        
        # 포함-배제: P(A1 ∪ ... ∪ An) = Σ P(Ai) - Σ P(Ai ∩ Aj) + ...
        # 여기서는 반대로 여집합을 계산: 1 - P(적어도 하나 없음)
        for i in range(1, 2**n):
            subset = [num for j, num in enumerate(required_numbers) if i & (1 << j)]
            subset_size = len(subset)
            
            # 이 subset의 숫자들이 모두 없을 확률
            prob_all_missing = ((6 - subset_size) / 6) ** 10
            
            # 포함-배제 원리: 홀수개면 더하고, 짝수개면 빼기
            if subset_size % 2 == 1:
                total_prob += prob_all_missing
            else:
                total_prob -= prob_all_missing
        
        return 1 - total_prob
    
    # SMALL STRAIGHT: 연속 4개 (1234, 2345, 3456)
    # 공식: P(1234) + P(2345) + P(3456) - P(12345) - P(23456) + P(123456)
    
    prob_1234 = prob_contains_all([1,2,3,4])
    prob_2345 = prob_contains_all([2,3,4,5])  
    prob_3456 = prob_contains_all([3,4,5,6])
    prob_12345 = prob_contains_all([1,2,3,4,5])
    prob_23456 = prob_contains_all([2,3,4,5,6])
    prob_123456 = 0  # 10개 주사위로 6개 다른 숫자 불가능
    
    prob_small = prob_1234 + prob_2345 + prob_3456 - prob_12345 - prob_23456 + prob_123456
    
    # LARGE STRAIGHT: 연속 5개 (12345, 23456)  
    # 공식: P(12345) + P(23456) - P(123456)
    prob_large = prob_12345 + prob_23456 - prob_123456
    
    return prob_small * 15000, prob_large * 30000

def calculate_full_house_exact():
    """FULL_HOUSE: 다항분포 방식으로 정확한 계산 (서로 다른 숫자 3+2만)"""
    
    def multinomial_coeff(n, counts):
        """다항분포 계수 계산"""
        result = math.factorial(n)
        for c in counts:
            result //= math.factorial(c)
        return result
    
    def generate_compositions(n, k, current=[]):
        """n을 k개 음이 아닌 정수의 합으로 나타내는 모든 방법"""
        if k == 1:
            yield current + [n]
            return
        
        for i in range(n + 1):
            yield from generate_compositions(n - i, k - 1, current + [i])
    
    def find_best_full_house(counts):
        """주어진 주사위 개수에서 최적의 풀하우스 조합 찾기 (서로 다른 숫자만)"""
        best_score = 0
        
        # 모든 가능한 풀하우스 조합 (서로 다른 숫자의 3개 + 2개)
        for three_num in range(1, 7):
            if counts[three_num-1] >= 3:
                for two_num in range(1, 7):
                    if two_num != three_num and counts[two_num-1] >= 2:
                        score = three_num * 3 + two_num * 2
                        best_score = max(best_score, score)
        
        return best_score
    
    total_expected = 0
    
    # 10개 주사위의 모든 가능한 조합
    for counts in generate_compositions(10, 6):
        # 다항분포 확률
        prob = multinomial_coeff(10, counts) * (1/6)**10
        
        # 이 조합에서 최적의 풀하우스 점수 찾기
        best_score = find_best_full_house(counts)
        
        if best_score > 0:
            total_expected += prob * best_score * 1000
    
    return total_expected

def main():
    """모든 점수 기댓값을 정확히 계산 (보너스 포함)"""
    import time
    print("=== 요트 게임 정확한 기댓값 계산 ===\n")
    
    # 기본 점수 분포 계산 (보너스 제외)
    basic_distributions, basic_expected = calculate_basic_distributions()
    print("기본 점수 (보너스 제외):", [round(score, 2) for score in basic_expected])
    
    # 보너스 계산
    print("\n=== 보너스 계산 시작 ===")
    start_time = time.time()
    bonus_expected = calculate_bonus_exact()
    end_time = time.time()
    print(f"보너스 계산 완료! 소요 시간: {end_time - start_time:.2f}초")
    
    # 보너스 포함 기본 점수
    basic_expected_with_bonus = [basic + bonus for basic, bonus in zip(basic_expected, bonus_expected)]
    print("\n보너스 각 숫자별 기댓값:", [round(bonus, 2) for bonus in bonus_expected])
    print("기본 점수 (보너스 포함):", [round(score, 2) for score in basic_expected_with_bonus])
    
    # 기본 점수 확률 분포 출력 (conditional과 동일한 형식)
    print("\n기본 점수 확률 분포:")
    for num in range(1, 7):
        print(f"숫자 {num}: {basic_distributions[num]}")
    
    # 조합 점수들 (정확 계산) - CHOICE 포함
    choice = calculate_choice_exact()
    four_kind = calculate_four_of_a_kind_exact()
    full_house = calculate_full_house_exact()
    small_straight, large_straight = calculate_straight_patterns_exact()
    yacht = calculate_yacht_exact()
    
    print(f"\nCHOICE: {choice:.2f}")
    print(f"FOUR_OF_A_KIND: {four_kind:.2f}")
    print(f"FULL_HOUSE: {full_house:.2f}")
    print(f"SMALL_STRAIGHT: {small_straight:.2f}")
    print(f"LARGE_STRAIGHT: {large_straight:.2f}")
    print(f"YACHT: {yacht:.2f}")
    
    # 최종 리스트 (보너스 포함)
    combination_expected = {
        'CHOICE': choice,
        'FOUR_OF_A_KIND': four_kind,
        'FULL_HOUSE': full_house,
        'SMALL_STRAIGHT': small_straight,
        'LARGE_STRAIGHT': large_straight,
        'YACHT': yacht
    }
    
    result = {
        'basic_distributions': basic_distributions,
        'basic_expected': basic_expected_with_bonus,  # 보너스 포함
        'basic_expected_no_bonus': basic_expected,    # 보너스 제외 (참고용)
        'bonus_expected': bonus_expected,             # 보너스만
        'combination_expected': combination_expected
    }
    
    print(f"\n=== 최종 정확 결과 (보너스 포함) ===")
    print("Basic Expected (보너스 포함):", [round(x, 2) for x in basic_expected_with_bonus])
    print("Bonus Expected:", [round(x, 2) for x in bonus_expected])
    print("Combination Expected:", {k: round(v, 2) for k, v in combination_expected.items()})
    
    return result

if __name__ == "__main__":
    main()
import math

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
    'basic_expected': [2027.34, 3691.28, 5355.22, 7019.17, 8683.11, 10347.05],  # 보너스 포함
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


def calculate_choice_conditional(given_dice):
    """CHOICE 조건부 기댓값 - 다항분포 방식으로 효율적 계산"""
    
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
    
    given_counts = [given_dice.count(i) for i in range(1, 7)]
    total_expected = 0
    
    # 추가 5개 주사위의 모든 가능한 조합
    for additional_counts in generate_compositions(5, 6):
        # 다항분포 확률
        prob = multinomial_coeff(5, additional_counts) * (1/6)**5
        
        # 전체 개수 계산
        total_counts = [given + add for given, add in zip(given_counts, additional_counts)]
        
        # 각 숫자의 개수만큼 해당 숫자들을 생성
        all_dice = []
        for num in range(1, 7):
            all_dice.extend([num] * total_counts[num-1])
        
        # 상위 5개 선택
        top_5_sum = sum(sorted(all_dice, reverse=True)[:5])
        total_expected += prob * top_5_sum
    
    return total_expected * 1000  # 점수 변환

def calculate_four_of_a_kind_conditional(given_counts):
    """FOUR_OF_A_KIND 조건부 기댓값 - 다항분포 방식으로 확률 가중 합산"""
    
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
    
    def find_best_four_of_a_kind(total_counts):
        """주어진 주사위 개수에서 최적의 포카드 조합 찾기"""
        best_score = 0
        
        # 각 숫자별로 포카드 가능성 확인
        for target_num in range(1, 7):
            if total_counts[target_num-1] >= 4:
                # 포카드 가능 - 해당 숫자 4개 + 나머지 중 가장 큰 1개
                remaining_dice = []
                for num in range(1, 7):
                    count = total_counts[num-1]
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
    
    # 추가 5개 주사위의 모든 가능한 조합
    for additional_counts in generate_compositions(5, 6):
        # 다항분포 확률
        prob = multinomial_coeff(5, additional_counts) * (1/6)**5
        
        # 전체 개수 계산
        total_counts = [given + add for given, add in zip(given_counts, additional_counts)]
        
        # 이 경우에서 최적의 포카드 점수 찾기
        best_score = find_best_four_of_a_kind(total_counts)
        
        if best_score > 0:
            total_expected += prob * best_score * 1000
    
    return total_expected

def calculate_full_house_conditional(given_counts):
    """FULL_HOUSE 조건부 기댓값 - 모든 경우 통합 계산"""
    
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
    
    def find_best_full_house(total_counts):
        """주어진 주사위 개수에서 최적의 풀하우스 조합 찾기"""
        best_score = 0
        
        # 모든 가능한 풀하우스 조합 (서로 다른 숫자의 3개 + 2개)
        for three_num in range(1, 7):
            if total_counts[three_num-1] >= 3:
                for two_num in range(1, 7):
                    if two_num != three_num and total_counts[two_num-1] >= 2:
                        score = three_num * 3 + two_num * 2
                        best_score = max(best_score, score)
        
        return best_score
    
    total_expected = 0
    
    # 추가 5개 주사위의 모든 가능한 조합
    for additional_counts in generate_compositions(5, 6):
        # 다항분포 확률
        prob = multinomial_coeff(5, additional_counts) * (1/6)**5
        
        # 전체 개수 계산
        total_counts = [given + add for given, add in zip(given_counts, additional_counts)]
        
        # 이 경우에서 최적의 풀하우스 점수 찾기
        best_score = find_best_full_house(total_counts)
        
        if best_score > 0:
            total_expected += prob * best_score * 1000
    
    return total_expected

def calculate_small_straight_conditional(given_dice):
    """SMALL_STRAIGHT 조건부 기댓값 - 정확한 다항분포 계산"""
    
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
    
    def check_small_straight(total_counts):
        """스몰 스트레이트 조건 확인 (1234, 2345, 3456)"""
        patterns = [
            [1,2,3,4],    # 1234
            [2,3,4,5],    # 2345  
            [3,4,5,6]     # 3456
        ]
        
        for pattern in patterns:
            if all(total_counts[num-1] > 0 for num in pattern):
                return True
        return False
    
    given_counts = [given_dice.count(i) for i in range(1, 7)]
    total_expected = 0
    
    # 추가 5개 주사위의 모든 가능한 조합
    for additional_counts in generate_compositions(5, 6):
        # 다항분포 확률
        prob = multinomial_coeff(5, additional_counts) * (1/6)**5
        
        # 전체 개수 계산
        total_counts = [given + add for given, add in zip(given_counts, additional_counts)]
        
        # 스몰 스트레이트 조건 확인
        if check_small_straight(total_counts):
            total_expected += prob * 15000
    
    return total_expected

def calculate_large_straight_conditional(given_dice):
    """LARGE_STRAIGHT 조건부 기댓값 - 정확한 다항분포 계산"""
    
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
    
    def check_large_straight(total_counts):
        """라지 스트레이트 조건 확인 (12345, 23456)"""
        patterns = [
            [1,2,3,4,5],  # 12345
            [2,3,4,5,6]   # 23456
        ]
        
        for pattern in patterns:
            if all(total_counts[num-1] > 0 for num in pattern):
                return True
        return False
    
    given_counts = [given_dice.count(i) for i in range(1, 7)]
    total_expected = 0
    
    # 추가 5개 주사위의 모든 가능한 조합
    for additional_counts in generate_compositions(5, 6):
        # 다항분포 확률
        prob = multinomial_coeff(5, additional_counts) * (1/6)**5
        
        # 전체 개수 계산
        total_counts = [given + add for given, add in zip(given_counts, additional_counts)]
        
        # 라지 스트레이트 조건 확인
        if check_large_straight(total_counts):
            total_expected += prob * 30000
    
    return total_expected

def calculate_yacht_conditional(given_counts):
    """YACHT 조건부 기댓값 - 각 숫자별 개별 계산 후 최적 선택"""
    max_expected = 0
    
    # 각 숫자(1~6)별로 YACHT가 될 때의 기댓값을 개별 계산
    for target_number in range(1, 7):
        given_target = given_counts[target_number-1]
        
        if given_target >= 5:
            # 이미 5개 이상 있음 - 확정
            this_expected = 50000
        elif given_target >= 1:
            # 1~4개 있음 - 추가로 (5-given_target)개 이상 필요
            needed = 5 - given_target
            prob_success = sum(binomial_prob(5, k, 1/6) for k in range(needed, 6))
            this_expected = prob_success * 50000
        else:
            # 해당 숫자가 하나도 없음
            # 5개 모두가 해당 숫자가 될 확률
            prob_success = binomial_prob(5, 5, 1/6)
            this_expected = prob_success * 50000
        
        # 최대값 업데이트
        max_expected = max(max_expected, this_expected)
    
    return max_expected

def calculate_all_categories_conditional(given_dice):
    """
    주어진 주사위에 대해 모든 족보별 조건부 기댓값 계산
    returns: dict with all category expected values
    """
    given_counts = [given_dice.count(i) for i in range(1, 7)]
    
    results = {}
    
    # 기본 족보 (ONE~SIX)
    for num in range(1, 7):
        results[f"one" if num == 1 else
               f"two" if num == 2 else
               f"three" if num == 3 else
               f"four" if num == 4 else
               f"five" if num == 5 else
               f"six"] = calculate_score_for_selection(given_dice, num)
    
    # 조합 족보들
    results["choice"] = calculate_choice_conditional(given_dice)
    results["four_of_a_kind"] = calculate_four_of_a_kind_conditional(given_counts)
    results["full_house"] = calculate_full_house_conditional(given_counts)
    results["small_straight"] = calculate_small_straight_conditional(given_dice)
    results["large_straight"] = calculate_large_straight_conditional(given_dice)
    results["yacht"] = calculate_yacht_conditional(given_counts)
    
    return results

if __name__ == "__main__":
    # 테스트 케이스들
    test_cases = [
        [1, 1, 1, 1, 1],  # 야트 확정
        [2, 2, 2, 3, 4],  # 2가 3개
        [1, 2, 3, 4, 5],  # 스트레이트에 가까움
        [3, 3, 3, 4, 4],  # 풀하우스 확정
    ]
    
    for i, test_dice in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"테스트 케이스 {i}: {test_dice}")
        print(f"{'='*50}")
        
        results = calculate_all_categories_conditional(test_dice)
        
        print("\n=== 각 족보별 기댓값 ===")
        for category, expected_value in results.items():
            print(f"{category:15}: {expected_value:8.2f}")
        
        # 최적 선택 분석
        best_category = max(results.keys(), key=lambda x: results[x])
        best_value = results[best_category]
        print(f"\n최적 선택: {best_category} ({best_value:.2f}점)")
        print(f"{'='*50}")
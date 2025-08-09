#!/usr/bin/env python3
"""
Expectation-based Yacht AI (Combinatorial Expectation)
기댓값 기반 야치 인공지능 시스템 (조합적 기댓값 계산 버전)
- 몬테카를로 시뮬레이션을 모든 경우의 수를 따지는 조합적 계산 방식으로 대체
- 턴 상황(0->5개 vs 5->10개)에 따른 정교한 가치 평가 로직 적용
- '기회비용' (기댓값 대비 이득) 기반의 최종 의사결정 구조 유지
"""

import sys
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict
from itertools import product

# ============================= BASE DATA =============================

BASE_EXPECTATIONS = {
    'ONE': 2027.34, 'TWO': 3691.28, 'THREE': 5355.22, 'FOUR': 7019.17,
    'FIVE': 8683.11, 'SIX': 10347.05, 'CHOICE': 24179.31, 'FOUR_OF_A_KIND': 7321.42,
    'FULL_HOUSE': 17631.83, 'SMALL_STRAIGHT': 9750.00, 'LARGE_STRAIGHT': 21372.39, 'YACHT': 4638.59
}

# ========================= GAME RULES & STATE =========================

class DiceRule(Enum):
    ONE, TWO, THREE, FOUR, FIVE, SIX, CHOICE, FOUR_OF_A_KIND, FULL_HOUSE, SMALL_STRAIGHT, LARGE_STRAIGHT, YACHT = range(12)

@dataclass
class Bid:
    group: str
    amount: int

@dataclass
class DicePut:
    rule: DiceRule
    dice: List[int]

class GameState:
    def __init__(self):
        self.dice: List[int] = []
        self.rule_score: List[Optional[int]] = [None] * 12
        self.bid_score: int = 0
        self.basic_score_sum: int = 0

    def get_total_score(self) -> int:
        bonus = 35000 if self.basic_score_sum >= 63000 else 0
        combination = sum(s for i, s in enumerate(self.rule_score) if i >= 6 and s is not None)
        return self.basic_score_sum + bonus + combination + self.bid_score

    def bid(self, is_successful: bool, amount: int):
        self.bid_score += -amount if is_successful else amount

    def add_dice(self, new_dice: List[int]):
        self.dice.extend(new_dice)

    def use_dice(self, put: DicePut):
        assert put.rule is not None and self.rule_score[put.rule.value] is None
        for d in put.dice: self.dice.remove(d)
        score = self.calculate_score(put)
        self.rule_score[put.rule.value] = score
        if put.rule.value < 6: self.basic_score_sum += score

    @staticmethod
    def calculate_score(put: DicePut) -> int:
        rule, dice = put.rule, sorted(put.dice)
        counts = {i: dice.count(i) for i in range(1, 7)}
        if rule.value <= 5: return (rule.value + 1) * counts.get(rule.value + 1, 0) * 1000
        if rule == DiceRule.CHOICE: return sum(dice) * 1000
        if rule == DiceRule.FOUR_OF_A_KIND: return sum(dice) * 1000 if any(c >= 4 for c in counts.values()) else 0
        if rule == DiceRule.FULL_HOUSE:
            vals = list(counts.values())
            return sum(dice) * 1000 if (3 in vals and 2 in vals) or 5 in vals else 0
        if rule == DiceRule.SMALL_STRAIGHT:
            s = "".join(map(str, sorted(list(set(dice)))))
            return 15000 if "1234" in s or "2345" in s or "3456" in s else 0
        if rule == DiceRule.LARGE_STRAIGHT:
            s = "".join(map(str, sorted(list(set(dice)))))
            return 30000 if "12345" in s or "23456" in s else 0
        if rule == DiceRule.YACHT: return 50000 if 5 in counts.values() else 0
        return 0

# ========================= COMBINATORIAL EXPECTATION CALCULATOR =========================

class ExpectationCalculator:
    """조합적 방식을 통해 기댓값 및 점수 가치를 정교하게 계산"""

    @staticmethod
    def _binomial_prob(n, k, p):
        """이항분포 확률 계산"""
        if not (0 <= k <= n): return 0
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    def _get_future_score_distribution(self, rule_value: int, current_dice: List[int]) -> List[tuple[int, float]]:
        """
        특정 기본 규칙에 대해, 미래에 얻게 될 점수의 확률 분포를 계산.
        (현재 주사위 + 미래의 5개 주사위 고려)
        """
        distribution = []
        num_to_match = rule_value + 1
        current_count = current_dice.count(num_to_match)
        
        # 미래에 굴릴 5개 주사위에서 해당 숫자가 나올 모든 경우(0~5개)를 계산
        for k in range(6): # k: 새로 나올 주사위의 개수
            prob = self._binomial_prob(5, k, 1/6)
            total_count = current_count + k
            score = min(5, total_count) * num_to_match * 1000
            distribution.append((score, prob))
        return distribution

    def calculate_bonus_expectation(self, player_state: GameState, candidate_rule: DiceRule, candidate_score: int, all_10_dice: List[int]) -> float:
        """조합적 방식으로 모든 경우의 수를 따져 보너스 기댓값을 계산"""
        fixed_score = player_state.basic_score_sum + candidate_score
        if fixed_score >= 63000: return 35000

        unfilled_basic_rules = [
            DiceRule(i) for i in range(6)
            if player_state.rule_score[i] is None and i != candidate_rule.value
        ]
        if not unfilled_basic_rules: return 0

        # 각 비어있는 규칙의 미래 점수 분포를 계산
        distributions = [self._get_future_score_distribution(r.value, all_10_dice) for r in unfilled_basic_rules]
        
        total_bonus_prob = 0
        # 모든 분포의 조합을 순회
        for combo in product(*distributions):
            future_score = sum(score for score, prob in combo)
            combo_prob = math.prod(prob for score, prob in combo)
            
            if fixed_score + future_score >= 63000:
                total_bonus_prob += combo_prob
        
        return total_bonus_prob * 35000

    def calculate_potential_score(self, all_10_dice: List[int], rule: DiceRule, player_state: GameState) -> float:
        """10개 주사위로 얻을 수 있는 잠재적 총 점수(보너스 기댓값 포함) 계산"""
        best_dice_selection = Game._select_best_dice_for_rule(all_10_dice, rule)
        base_score = GameState.calculate_score(DicePut(rule, best_dice_selection))

        if rule.value < 6:
            bonus_exp = self.calculate_bonus_expectation(player_state, rule, base_score, all_10_dice)
            return base_score + bonus_exp
        return base_score

# ============================== AI BRAIN CLASS ==============================

class Game:
    def __init__(self):
        self.my_state = GameState()
        self.opp_state = GameState()
        self.current_round = 0
        self.calculator = ExpectationCalculator()

    @staticmethod
    def _select_best_dice_for_rule(all_dice: List[int], rule: DiceRule) -> List[int]:
        """특정 규칙에 최적인 주사위 5개 선택"""
        if len(all_dice) <= 5: return all_dice
        if rule.value >= 6: return sorted(all_dice, reverse=True)[:5]
        target_num = rule.value + 1
        target_dice = [d for d in all_dice if d == target_num]
        other_dice = sorted([d for d in all_dice if d != target_num], reverse=True)
        return (target_dice + other_dice)[:5]

    def _evaluate_dice_option(self, dice_option: List[int], player_state: GameState) -> float:
        """턴 상황에 맞게 주사위 묶음의 가치를 평가"""
        available_rules = [DiceRule(i) for i, s in enumerate(player_state.rule_score) if s is None]
        
        # 첫 턴 (0 -> 5개): 기댓값 총합의 변화량으로 가치 평가
        if not player_state.dice:
            # (미래에 얻을 총 기댓값) - (현재 5개 주사위로 인해 변하는 미래 총 기댓값)
            # 이 변화량이 '적을수록' 미래 가치를 덜 훼손하는 것이므로, 더 좋은 패임.
            base_exp_sum = sum(BASE_EXPECTATIONS[r.name] for r in available_rules)
            cond_exp_sum = sum(GameState.calculate_score(DicePut(r, dice_option)) for r in available_rules)
            return base_exp_sum - cond_exp_sum

        # 이후 턴 (5 -> 10개): 잠재적 점수 이득으로 가치 평가
        all_dice = player_state.dice + dice_option
        max_potential_score = -1
        for rule in available_rules:
            score = self.calculator.calculate_potential_score(all_dice, rule, player_state)
            if score > max_potential_score:
                max_potential_score = score
        return max_potential_score

    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        """입찰 로직"""
        my_value_a = self._evaluate_dice_option(dice_a, self.my_state)
        my_value_b = self._evaluate_dice_option(dice_b, self.my_state)
        opp_value_a = self._evaluate_dice_option(dice_a, self.opp_state)
        opp_value_b = self._evaluate_dice_option(dice_b, self.opp_state)

        my_preference = "A" if my_value_a > my_value_b else "B"
        opp_preference = "A" if opp_value_a > opp_value_b else "B"
        
        if my_preference != opp_preference:
            return Bid(my_preference, 0)

        my_gain_diff = abs(my_value_a - my_value_b)
        base_bid = my_gain_diff * 0.5 
        factor = 1.0
        score_diff = self.my_state.get_total_score() - self.opp_state.get_total_score()

        if self.current_round >= 9 and score_diff < -15000: factor = 2.0
        elif self.current_round >= 9 and score_diff > 15000: factor = 0.5
        elif self.current_round <= 4: factor = 1.2
        
        final_bid_amount = int(base_bid * factor)
        return Bid(my_preference, max(0, min(100000, final_bid_amount)))

    def calculate_put(self) -> DicePut:
        """'기댓값 대비 이득'을 고려한 최적의 주사위 배치 결정"""
        available_rules = [DiceRule(i) for i, s in enumerate(self.my_state.rule_score) if s is None]
        if not available_rules: return DicePut(DiceRule.CHOICE, self.my_state.dice[:5])

        best_rule, best_gain = None, -float('inf')

        for rule in available_rules:
            potential_score = self.calculator.calculate_potential_score(self.my_state.dice, rule, self.my_state)
            gain = potential_score - BASE_EXPECTATIONS.get(rule.name, 0)
            
            if gain > best_gain:
                best_gain = gain
                best_rule = rule

        if best_rule is None:
            best_rule = sorted(available_rules, key=lambda r: BASE_EXPECTATIONS.get(r.name, 0))[0]

        final_dice_selection = self._select_best_dice_for_rule(self.my_state.dice, best_rule)
        return DicePut(best_rule, final_dice_selection)

    def update_get(self, dice_a: List[int], dice_b: List[int], my_bid: Bid, opp_bid: Bid, my_group: str):
        self.current_round += 1
        if my_group == "A":
            self.my_state.add_dice(dice_a); self.opp_state.add_dice(dice_b)
        else:
            self.my_state.add_dice(dice_b); self.opp_state.add_dice(dice_a)
        self.my_state.bid(my_bid.group == my_group, my_bid.amount)
        self.opp_state.bid(opp_bid.group == ("B" if my_group == "A" else "A"), opp_bid.amount)

    def update_put(self, put: DicePut): self.my_state.use_dice(put)
    def update_set(self, put: DicePut): self.opp_state.use_dice(put)

def main():
    game = Game()
    dice_a, dice_b = [], []
    my_bid = Bid("", 0)
    while True:
        try:
            line = input().strip()
            if not line: continue
            command, *args = line.split()
            if command == "READY": print("OK")
            elif command == "ROLL":
                dice_a, dice_b = [int(c) for c in args[0]], [int(c) for c in args[1]]
                my_bid = game.calculate_bid(dice_a, dice_b)
                print(f"BID {my_bid.group} {my_bid.amount}")
            elif command == "GET":
                game.update_get(dice_a, dice_b, my_bid, Bid(args[1], int(args[2])), args[0])
            elif command == "SCORE":
                put = game.calculate_put()
                game.update_put(put)
                print(f"PUT {put.rule.name} {''.join(map(str, sorted(put.dice)))}")
            elif command == "SET":
                game.update_set(DicePut(DiceRule[args[0]], [int(c) for c in args[1]]))
            elif command == "FINISH": break
        except (EOFError, IndexError): break

if __name__ == "__main__":
    main()
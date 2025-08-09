#!/usr/bin/env python3
"""
Neural Network-based Yacht AI
4-Way Parallel Branch Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random

# 주사위 규칙들을 나타내는 enum
class DiceRule(Enum):
    ONE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    CHOICE = 6
    FOUR_OF_A_KIND = 7
    FULL_HOUSE = 8
    SMALL_STRAIGHT = 9
    LARGE_STRAIGHT = 10
    YACHT = 11


# 입찰 방법을 나타내는 데이터클래스
@dataclass
class Bid:
    group: str  # 입찰 그룹 ('A' 또는 'B')
    amount: int  # 입찰 금액


# 주사위 배치 방법을 나타내는 데이터클래스
@dataclass
class DicePut:
    rule: DiceRule  # 배치 규칙
    dice: List[int]  # 배치할 주사위 목록


class YachtBranchNet(nn.Module):
    """각 경우의 수를 담당하는 브랜치 (중간 감독 학습 포함)"""
    
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 중간 출력: 12개 조합 선호도 (학습용)
        self.preference_head = nn.Linear(32, 12)
        
        # 최종 출력: 입찰 비용 (의사결정용)
        self.bid_value_head = nn.Linear(32, 1)
    
    def forward(self, x):
        shared_features = self.shared(x)
        
        # 중간 출력 - 12개 조합별 선호도
        preferences = self.preference_head(shared_features)
        
        # 최종 출력 - 입찰 비용 (음수 방지)
        bid_value = F.relu(self.bid_value_head(shared_features))
        
        return preferences, bid_value


class YachtMasterNet(nn.Module):
    """4개 브랜치를 통합하는 마스터 네트워크"""
    
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([YachtBranchNet() for _ in range(4)])
        
        # 4개 브랜치의 비용만 사용 (4차원)
        self.decision_layer = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # [A선택 로짓, B선택 로짓, 최종 입찰비용]
        )
        
    def forward(self, inputs):
        """
        inputs: 4개 브랜치용 입력 [my_A, my_B, opp_A, opp_B]
        각 입력은 18차원 (보드 12 + 주사위 6)
        shape: (batch_size, 4, 18)
        """
        branch_preferences = []  # 학습용 중간 출력
        branch_costs = []        # 의사결정용 최종 출력
        
        # 4개 브랜치 실행
        for i, branch in enumerate(self.branches):
            # inputs[:, i, :]로 각 브랜치 입력 추출
            branch_input = inputs[:, i, :]  # (batch_size, 18)
            preferences, cost = branch(branch_input)
            branch_preferences.append(preferences)
            branch_costs.append(cost.squeeze(-1))
        
        # 최종 결정은 4개 비용만으로
        costs_tensor = torch.stack(branch_costs, dim=-1)  # shape: (batch_size, 4)
        
        output = self.decision_layer(costs_tensor)
        
        # A/B 선택 확률 + 최종 입찰비용
        choice_logits = output[:, :2]  # A, B 로짓
        choice_probs = F.softmax(choice_logits, dim=-1)
        final_bid = F.relu(output[:, 2])  # 입찰비용 (음수 방지)
        
        return choice_probs, final_bid, branch_preferences, branch_costs


class NeuralYachtGame:
    """신경망 기반 Yacht 게임 클래스"""
    
    def __init__(self, model_path=None):
        self.my_state = GameState()
        self.opp_state = GameState()
        
        # 신경망 모델 초기화
        self.device = torch.device('cpu')  # 실전에서는 CPU 사용
        self.model = YachtMasterNet()
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"모델 로드 완료: {model_path}", file=sys.stderr)
            except:
                print(f"모델 로드 실패, 기본 가중치 사용", file=sys.stderr)
        
        self.model.eval()
        
        # 상대 보드 추정 (게임 진행하면서 업데이트)
        self.estimated_opp_board = [0] * 12
        self.round_count = 0
    
    def encode_board_state(self, board_state):
        """보드 상태를 신경망 입력으로 인코딩 (12차원)"""
        return [1 if score is not None else 0 for score in board_state]
    
    def encode_dice(self, dice):
        """주사위를 신경망 입력으로 인코딩 (6차원)"""
        dice_counts = [0] * 6
        for d in dice:
            if 1 <= d <= 6:
                dice_counts[d-1] += 1
        
        # 개수를 5로 나누어 정규화 (0~1 범위)
        return [count / 5.0 for count in dice_counts]
    
    def create_network_input(self, board_state, dice):
        """신경망 입력 생성 (18차원)"""
        board_encoded = self.encode_board_state(board_state)
        dice_encoded = self.encode_dice(dice)
        return torch.tensor(board_encoded + dice_encoded, dtype=torch.float32)
    
    def estimate_opponent_dice(self, my_dice, group):
        """상대방이 선택할 주사위 추정 (간단한 휴리스틱)"""
        # 실제로는 상대방 주사위를 모르므로 평균적인 분포로 추정
        if group == 'A':
            # A그룹 주사위를 내가 선택하지 않았다면, 상대가 선택할 가능성이 높음
            return [random.randint(1, 6) for _ in range(5)]  # 임시 추정
        else:
            return [random.randint(1, 6) for _ in range(5)]  # 임시 추정
    
    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        """신경망을 사용한 입찰 결정"""
        self.round_count += 1
        
        # 4개 브랜치용 입력 준비
        my_board = self.my_state.rule_score
        
        # 1. 내가 A 선택시
        my_a_input = self.create_network_input(my_board, dice_a)
        
        # 2. 내가 B 선택시  
        my_b_input = self.create_network_input(my_board, dice_b)
        
        # 3. 상대가 A 선택시 (상대 보드 추정 + A 주사위)
        opp_a_input = self.create_network_input(self.estimated_opp_board, dice_a)
        
        # 4. 상대가 B 선택시
        opp_b_input = self.create_network_input(self.estimated_opp_board, dice_b)
        
        # 신경망 입력 준비 - (4, 18) -> (1, 4, 18)
        network_inputs = torch.stack([my_a_input, my_b_input, opp_a_input, opp_b_input]).unsqueeze(0)
        
        with torch.no_grad():
            choice_probs, bid_amount, branch_preferences, branch_costs = self.model(network_inputs)
            
            # 선택 결정 (A=0, B=1)
            choice_idx = torch.argmax(choice_probs[0]).item()
            group = "A" if choice_idx == 0 else "B"
            
            # 입찰 금액 (0~100000 범위로 조정)
            amount = int(torch.clamp(bid_amount[0] * 50000, 0, 100000).item())
            
            # 디버그 정보
            print(f"신경망 결정: {group}, 금액: {amount}, 확률: A={choice_probs[0][0]:.3f}, B={choice_probs[0][1]:.3f}", 
                  file=sys.stderr)
        
        return Bid(group, amount)
    
    def calculate_put(self) -> DicePut:
        """신경망 선호도를 사용한 주사위 배치 결정"""
        available_rules = [rule for rule in DiceRule if self.my_state.rule_score[rule.value] is None]
        
        if not available_rules:
            return DicePut(DiceRule.CHOICE, self.my_state.dice[:5])
        
        # 신경망을 사용하여 선호도 계산
        try:
            # 현재 내 상황에서 선호도 계산 (브랜치 0만 사용)
            my_input = self.create_network_input(self.my_state.rule_score, self.my_state.dice)
            
            with torch.no_grad():
                preferences, _ = self.model.branches[0](my_input.unsqueeze(0))
                preferences = preferences[0]  # 배치 차원 제거
            
            # 사용 가능한 규칙 중에서 가장 선호도 높은 것 선택
            best_rule = None
            best_preference = float('-inf')
            
            for rule in available_rules:
                rule_preference = preferences[rule.value].item()
                if rule_preference > best_preference:
                    best_preference = rule_preference
                    best_rule = rule
            
            if best_rule is None:
                best_rule = available_rules[0]
            
            # 디버그 정보
            print(f"배치 결정: {best_rule.name}, 선호도: {best_preference:.3f}", file=sys.stderr)
                
        except Exception as e:
            # 신경망 오류시 fallback
            print(f"신경망 배치 실패, fallback 사용: {e}", file=sys.stderr)
            best_rule = available_rules[0]
        
        # 해당 규칙에 최적인 주사위 선택
        best_dice = self.select_best_dice_for_rule(self.my_state.dice, best_rule)
        
        return DicePut(best_rule, best_dice)
    
    def calculate_rule_score(self, rule: DiceRule, dice: List[int]) -> int:
        """특정 규칙으로 얻을 수 있는 점수 계산"""
        test_put = DicePut(rule, dice[:5])
        return GameState.calculate_score(test_put)
    
    def select_best_dice_for_rule(self, dice: List[int], rule: DiceRule) -> List[int]:
        """특정 규칙에 최적인 주사위 5개 선택"""
        if len(dice) <= 5:
            result = dice[:]
        elif rule in [DiceRule.ONE, DiceRule.TWO, DiceRule.THREE, DiceRule.FOUR, DiceRule.FIVE, DiceRule.SIX]:
            # 해당 숫자 최대한 많이
            target = rule.value + 1
            target_dice = [d for d in dice if d == target]
            other_dice = [d for d in dice if d != target]
            result = (target_dice + other_dice)[:5]
        else:
            # 기타 규칙은 큰 숫자부터
            result = sorted(dice, reverse=True)[:5]
        
        # 5개 보장
        while len(result) < 5:
            result.append(1)
            
        return result[:5]
    
    def update_get(self, dice_a: List[int], dice_b: List[int], my_bid: Bid, opp_bid: Bid, my_group: str):
        """입찰 결과 업데이트"""
        if my_group == "A":
            self.my_state.add_dice(dice_a)
            self.opp_state.add_dice(dice_b)
        else:
            self.my_state.add_dice(dice_b)
            self.opp_state.add_dice(dice_a)
        
        # 입찰 점수 반영
        my_bid_success = my_bid.group == my_group
        self.my_state.bid(my_bid_success, my_bid.amount)
        
        opp_group = "B" if my_group == "A" else "A"
        opp_bid_success = opp_bid.group == opp_group
        self.opp_state.bid(opp_bid_success, opp_bid.amount)
    
    def update_put(self, put: DicePut):
        """내 주사위 배치 결과 반영"""
        self.my_state.use_dice(put)
    
    def update_set(self, put: DicePut):
        """상대 주사위 배치 결과 반영 (상대 보드 추정 업데이트)"""
        self.opp_state.use_dice(put)
        # 상대가 사용한 규칙 기록
        self.estimated_opp_board[put.rule.value] = 1


# 팀의 현재 상태를 관리하는 클래스 (기존과 동일)
class GameState:
    def __init__(self):
        self.dice = []
        self.rule_score: List[Optional[int]] = [None] * 12
        self.bid_score = 0

    def get_total_score(self) -> int:
        basic = bonus = combination = 0
        basic = sum(score for score in self.rule_score[0:6] if score is not None)
        bonus = 35000 if basic >= 63000 else 0
        combination = sum(score for score in self.rule_score[6:12] if score is not None)
        return basic + bonus + combination + self.bid_score

    def bid(self, is_successful: bool, amount: int):
        if is_successful:
            self.bid_score -= amount
        else:
            self.bid_score += amount

    def add_dice(self, new_dice: List[int]):
        self.dice.extend(new_dice)

    def use_dice(self, put: DicePut):
        assert put.rule is not None and self.rule_score[put.rule.value] is None, "Rule already used"
        
        for d in put.dice:
            self.dice.remove(d)
        
        assert put.rule is not None
        self.rule_score[put.rule.value] = self.calculate_score(put)

    @staticmethod
    def calculate_score(put: DicePut) -> int:
        """규칙에 따른 점수 계산"""
        rule, dice = put.rule, put.dice

        if rule == DiceRule.ONE:
            return sum(d for d in dice if d == 1) * 1000
        elif rule == DiceRule.TWO:
            return sum(d for d in dice if d == 2) * 1000
        elif rule == DiceRule.THREE:
            return sum(d for d in dice if d == 3) * 1000
        elif rule == DiceRule.FOUR:
            return sum(d for d in dice if d == 4) * 1000
        elif rule == DiceRule.FIVE:
            return sum(d for d in dice if d == 5) * 1000
        elif rule == DiceRule.SIX:
            return sum(d for d in dice if d == 6) * 1000
        elif rule == DiceRule.CHOICE:
            return sum(dice) * 1000
        elif rule == DiceRule.FOUR_OF_A_KIND:
            ok = any(dice.count(i) >= 4 for i in range(1, 7))
            return sum(dice) * 1000 if ok else 0
        elif rule == DiceRule.FULL_HOUSE:
            pair = triple = False
            for i in range(1, 7):
                cnt = dice.count(i)
                if cnt == 2 or cnt == 5:
                    pair = True
                if cnt == 3 or cnt == 5:
                    triple = True
            return sum(dice) * 1000 if pair and triple else 0
        elif rule == DiceRule.SMALL_STRAIGHT:
            e1, e2, e3, e4, e5, e6 = [dice.count(i) > 0 for i in range(1, 7)]
            ok = ((e1 and e2 and e3 and e4) or 
                  (e2 and e3 and e4 and e5) or 
                  (e3 and e4 and e5 and e6))
            return 15000 if ok else 0
        elif rule == DiceRule.LARGE_STRAIGHT:
            e1, e2, e3, e4, e5, e6 = [dice.count(i) > 0 for i in range(1, 7)]
            ok = ((e1 and e2 and e3 and e4 and e5) or 
                  (e2 and e3 and e4 and e5 and e6))
            return 30000 if ok else 0
        elif rule == DiceRule.YACHT:
            ok = any(dice.count(i) == 5 for i in range(1, 7))
            return 50000 if ok else 0
        
        return 0


def main():
    """메인 게임 루프"""
    # 신경망 모델 경로 (실제 모델이 있다면)
    model_path = "yacht_model.pth" if False else None  # 현재는 없으므로 None
    
    game = NeuralYachtGame(model_path)
    
    dice_a, dice_b = [0] * 5, [0] * 5
    my_bid = Bid("", 0)

    while True:
        try:
            line = input().strip()
            if not line:
                continue

            command, *args = line.split()

            if command == "READY":
                print("OK")
                continue

            if command == "ROLL":
                str_a, str_b = args
                for i, c in enumerate(str_a):
                    dice_a[i] = int(c)
                for i, c in enumerate(str_b):
                    dice_b[i] = int(c)
                    
                my_bid = game.calculate_bid(dice_a, dice_b)
                print(f"BID {my_bid.group} {my_bid.amount}")
                continue

            if command == "GET":
                get_group, opp_group, opp_score = args
                opp_score = int(opp_score)
                game.update_get(dice_a, dice_b, my_bid, Bid(opp_group, opp_score), get_group)
                continue

            if command == "SCORE":
                put = game.calculate_put()
                game.update_put(put)
                assert put.rule is not None
                print(f"PUT {put.rule.name} {''.join(map(str, put.dice))}")
                continue

            if command == "SET":
                rule, str_dice = args
                dice = [int(c) for c in str_dice]
                game.update_set(DicePut(DiceRule[rule], dice))
                continue

            if command == "FINISH":
                break

            print(f"Invalid command: {command}", file=sys.stderr)
            sys.exit(1)

        except EOFError:
            break


if __name__ == "__main__":
    main()
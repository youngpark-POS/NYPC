#!/usr/bin/env python3
"""
Neural Network Training System for Yacht AI
Self-play data generation and training pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
import sys
import os
import copy

# 상위 디렉토리에서 모듈 import를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from yacht_nn_submission import YachtMasterNet, NeuralYachtGame, DiceRule, DicePut, Bid


def generate_game_data():
    """간단한 게임 데이터 생성"""
    game_data = []
    for _ in range(13):  # 13라운드
        dice_a = [random.randint(1, 6) for _ in range(5)]
        dice_b = [random.randint(1, 6) for _ in range(5)]
        tie_breaker = random.randint(0, 1)
        game_data.append((dice_a, dice_b, tie_breaker))
    return game_data


class YachtTrainingData:
    """학습 데이터를 담는 클래스"""
    
    def __init__(self):
        self.situations = []  # 게임 상황들
        self.labels = []      # 정답 레이블들
        self.weights = []     # 가중치들
    
    def add_sample(self, situation: Dict, label: Dict, weight: float = 1.0):
        self.situations.append(situation)
        self.labels.append(label)
        self.weights.append(weight)
    
    def __len__(self):
        return len(self.situations)
    
    def get_batch(self, indices: List[int]):
        batch_situations = [self.situations[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]
        batch_weights = [self.weights[i] for i in indices]
        return batch_situations, batch_labels, batch_weights


class SelfPlayEngine:
    """자기대국 엔진"""
    
    def __init__(self, model1: YachtMasterNet = None, model2: YachtMasterNet = None):
        self.model1 = model1 if model1 else YachtMasterNet()
        self.model2 = model2 if model2 else YachtMasterNet()
        
        # 모델을 평가 모드로 설정
        self.model1.eval()
        self.model2.eval()
        
        self.game_logs = []
    
    def play_single_game(self, game_data: List = None) -> Dict:
        """단일 게임 실행"""
        if game_data is None:
            game_data = generate_game_data()
        
        # 두 플레이어 초기화
        player1 = MockYachtPlayer(self.model1)
        player2 = MockYachtPlayer(self.model2)
        
        game_log = {
            'player1_log': [],
            'player2_log': [],
            'final_scores': [0, 0],
            'winner': 0
        }
        
        # 13라운드 진행
        for round_num in range(13):
            if round_num >= len(game_data):
                break
                
            dice_a, dice_b, tie_breaker = game_data[round_num]
            
            # 플레이어 1 턴
            p1_situation, p1_result = self.play_round(player1, player2, dice_a, dice_b, tie_breaker)
            game_log['player1_log'].append({
                'round': round_num,
                'situation': p1_situation,
                'result': p1_result
            })
            
            # 플레이어 2 턴 (주사위 순서 바뀜)
            p2_situation, p2_result = self.play_round(player2, player1, dice_b, dice_a, 1 - tie_breaker)
            game_log['player2_log'].append({
                'round': round_num,
                'situation': p2_situation,
                'result': p2_result
            })
        
        # 최종 점수 계산
        game_log['final_scores'] = [player1.get_total_score(), player2.get_total_score()]
        game_log['winner'] = 1 if game_log['final_scores'][0] > game_log['final_scores'][1] else 2
        
        return game_log
    
    def play_round(self, current_player, opponent, dice_a, dice_b, tie_breaker):
        """한 라운드 진행"""
        # 입찰 단계
        bid = current_player.make_bid(dice_a, dice_b, opponent.get_board_info())
        opp_bid = opponent.make_bid(dice_a, dice_b, current_player.get_board_info())
        
        # 승부 결정
        if bid.amount > opp_bid.amount:
            winner_group = bid.group
            winner_bid = bid.amount
        elif bid.amount < opp_bid.amount:
            winner_group = opp_bid.group
            winner_bid = opp_bid.amount  
        else:  # 동점
            winner_group = "A" if tie_breaker == 1 else "B"
            winner_bid = bid.amount
        
        # 주사위 분배
        if winner_group == "A":
            current_player.receive_dice(dice_a)
            opponent.receive_dice(dice_b)
        else:
            current_player.receive_dice(dice_b)
            opponent.receive_dice(dice_a)
        
        # 입찰 결과 반영
        current_player.apply_bid_result(bid, winner_group == bid.group)
        opponent.apply_bid_result(opp_bid, winner_group == opp_bid.group)
        
        # 주사위 배치
        put = current_player.place_dice()
        
        # 상황과 결과 기록
        situation = {
            'board_state': copy.deepcopy(current_player.board_state),
            'dice_a': dice_a,
            'dice_b': dice_b,
            'opp_board': copy.deepcopy(opponent.board_state),
            'bid_choice': bid.group,
            'bid_amount': bid.amount,
            'network_inputs': current_player.last_network_inputs
        }
        
        result = {
            'won_bid': winner_group == bid.group,
            'received_dice': dice_a if winner_group == "A" else dice_b,
            'placed_rule': put.rule.value if put.rule else None,
            'round_score': current_player.get_last_round_score()
        }
        
        return situation, result
    
    def generate_training_data(self, num_games: int = 1000) -> YachtTrainingData:
        """자기대국을 통한 학습 데이터 생성"""
        training_data = YachtTrainingData()
        
        print(f"자기대국 {num_games}게임 시작...")
        
        for game_idx in range(num_games):
            if game_idx % 100 == 0:
                print(f"게임 진행: {game_idx}/{num_games}")
            
            game_log = self.play_single_game()
            
            # 게임 결과를 바탕으로 학습 데이터 생성
            self.extract_training_samples(game_log, training_data)
        
        print(f"학습 데이터 생성 완료: {len(training_data)} 샘플")
        return training_data
    
    def extract_training_samples(self, game_log: Dict, training_data: YachtTrainingData):
        """게임 로그에서 학습 샘플 추출 (중간 감독 학습 포함)"""
        winner = game_log['winner']
        final_scores = game_log['final_scores']
        
        # 플레이어 1의 선택들 분석
        for round_data in game_log['player1_log']:
            situation = round_data['situation']
            result = round_data['result']
            
            # 게임 결과 기반 레이블 (승자의 선택이 positive)
            game_weight = 2.0 if winner == 1 else 0.5
            
            # 즉시 보상 기반 레이블 (라운드 성과)
            round_performance = self.calculate_round_performance(result)
            
            # 선택 레이블 (A=0, B=1)
            choice_label = 0 if situation['bid_choice'] == 'A' else 1
            
            # 입찰 금액 레이블 (성공한 입찰만 학습)
            if result['won_bid']:
                bid_label = min(situation['bid_amount'] / 100000.0, 1.0)
            else:
                bid_label = 0.0
            
            # 중간 단계 레이블 (실제 선택한 조합)
            placed_rule_idx = result.get('placed_rule', 0)  # 실제 배치한 규칙
            
            # 최종 가중치 (게임 결과 + 라운드 성과)
            final_weight = game_weight * (0.5 + round_performance)
            
            training_data.add_sample(
                situation=situation,
                label={
                    'choice': choice_label,
                    'bid': bid_label,
                    'performance': round_performance,
                    'placed_rule': placed_rule_idx  # 중간 감독 학습용
                },
                weight=final_weight
            )
    
    def calculate_round_performance(self, result: Dict) -> float:
        """라운드 성과 계산 (0~1)"""
        base_performance = 0.5
        
        # 입찰 성공시 보너스
        if result['won_bid']:
            base_performance += 0.3
        
        # 획득한 점수에 따른 보너스
        round_score = result.get('round_score', 0)
        if round_score >= 25000:  # 높은 점수
            base_performance += 0.2
        elif round_score >= 15000:  # 중간 점수
            base_performance += 0.1
        
        return min(base_performance, 1.0)


class MockYachtPlayer:
    """자기대국용 Mock 플레이어"""
    
    def __init__(self, model: YachtMasterNet):
        self.model = model
        self.board_state = [None] * 12  # 12개 규칙별 점수
        self.dice_inventory = []
        self.bid_score = 0
        self.last_round_score = 0
        self.last_network_inputs = None
        
    def get_board_info(self):
        """보드 정보 반환"""
        return {
            'board_state': [1 if score is not None else 0 for score in self.board_state],
            'total_score': self.get_total_score()
        }
    
    def make_bid(self, dice_a, dice_b, opp_info):
        """입찰 결정"""
        # 신경망 입력 준비 (18차원씩 4개)
        my_board = [1 if score is not None else 0 for score in self.board_state]
        opp_board = opp_info['board_state']
        
        def encode_dice(dice):
            counts = [0] * 6
            for d in dice:
                if 1 <= d <= 6:
                    counts[d-1] += 1
            return [c / 5.0 for c in counts]
        
        # 4개 브랜치 입력
        my_a_input = torch.tensor(my_board + encode_dice(dice_a), dtype=torch.float32)
        my_b_input = torch.tensor(my_board + encode_dice(dice_b), dtype=torch.float32)
        opp_a_input = torch.tensor(opp_board + encode_dice(dice_a), dtype=torch.float32)
        opp_b_input = torch.tensor(opp_board + encode_dice(dice_b), dtype=torch.float32)
        
        # (4, 18) 형태로 저장 (배치 차원은 나중에 DataLoader에서 추가)
        network_inputs = torch.stack([my_a_input, my_b_input, opp_a_input, opp_b_input])
        self.last_network_inputs = network_inputs
        
        with torch.no_grad():
            # 추론시에는 배치 차원 추가
            choice_probs, bid_amount, branch_preferences, branch_costs = self.model(network_inputs.unsqueeze(0))
            
            choice_idx = torch.argmax(choice_probs[0]).item()
            group = "A" if choice_idx == 0 else "B"
            amount = int(torch.clamp(bid_amount[0] * 50000, 0, 100000).item())
            
        return Bid(group, amount)
    
    def receive_dice(self, dice):
        """주사위 받기"""
        self.dice_inventory.extend(dice)
    
    def apply_bid_result(self, bid, success):
        """입찰 결과 적용"""
        if success:
            self.bid_score -= bid.amount
        else:
            self.bid_score += bid.amount
    
    def place_dice(self):
        """주사위 배치 (간단한 규칙 기반)"""
        if not self.dice_inventory:
            return DicePut(DiceRule.CHOICE, [1, 1, 1, 1, 1])
        
        available_rules = [DiceRule(i) for i in range(12) if self.board_state[i] is None]
        
        if not available_rules:
            return DicePut(DiceRule.CHOICE, self.dice_inventory[:5])
        
        # 가장 높은 점수를 얻을 수 있는 규칙 선택
        best_rule = None
        best_score = -1
        
        for rule in available_rules:
            test_dice = self.dice_inventory[:5]
            test_put = DicePut(rule, test_dice)
            score = self.calculate_rule_score(test_put)
            
            if score > best_score:
                best_score = score
                best_rule = rule
        
        if best_rule is None:
            best_rule = available_rules[0]
        
        # 최적 주사위 선택
        best_dice = self.select_optimal_dice(best_rule)
        
        # 상태 업데이트
        self.board_state[best_rule.value] = best_score
        self.last_round_score = best_score
        
        # 사용한 주사위 제거
        for d in best_dice:
            if d in self.dice_inventory:
                self.dice_inventory.remove(d)
        
        return DicePut(best_rule, best_dice)
    
    def select_optimal_dice(self, rule: DiceRule) -> List[int]:
        """규칙에 최적인 주사위 선택"""
        if len(self.dice_inventory) <= 5:
            result = self.dice_inventory[:]
        elif rule.value <= 5:  # 숫자 규칙
            target = rule.value + 1
            target_dice = [d for d in self.dice_inventory if d == target]
            other_dice = [d for d in self.dice_inventory if d != target]
            result = (target_dice + other_dice)[:5]
        else:  # 조합 규칙
            result = sorted(self.dice_inventory, reverse=True)[:5]
        
        # 5개 보장
        while len(result) < 5:
            result.append(1)
        
        return result[:5]
    
    def calculate_rule_score(self, put: DicePut) -> int:
        """규칙별 점수 계산 (간단화)"""
        rule, dice = put.rule, put.dice
        
        if rule.value <= 5:  # ONE ~ SIX
            target = rule.value + 1
            return sum(d for d in dice if d == target) * 1000
        elif rule == DiceRule.CHOICE:
            return sum(dice) * 1000
        elif rule == DiceRule.YACHT:
            if any(dice.count(i) == 5 for i in range(1, 7)):
                return 50000
        elif rule == DiceRule.LARGE_STRAIGHT:
            dice_set = set(dice)
            if {1,2,3,4,5}.issubset(dice_set) or {2,3,4,5,6}.issubset(dice_set):
                return 30000
        # 기타 규칙들은 단순화
        
        return 0
    
    def get_total_score(self) -> int:
        """총 점수 계산"""
        basic = sum(score for score in self.board_state[:6] if score is not None)
        bonus = 35000 if basic >= 63000 else 0
        combination = sum(score for score in self.board_state[6:] if score is not None)
        return basic + bonus + combination + self.bid_score
    
    def get_last_round_score(self) -> int:
        """마지막 라운드 점수"""
        return self.last_round_score


class YachtDataset(Dataset):
    """PyTorch Dataset for Yacht training data"""
    
    def __init__(self, training_data: YachtTrainingData):
        self.data = training_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        situation = self.data.situations[idx]
        label = self.data.labels[idx]
        weight = self.data.weights[idx]
        
        # 신경망 입력 추출
        network_inputs = situation['network_inputs']  # 이미 tensor 형태
        
        # 레이블 추출
        choice_label = label['choice']
        bid_label = label['bid']
        
        return {
            'inputs': network_inputs,
            'choice_target': torch.tensor(choice_label, dtype=torch.long),
            'bid_target': torch.tensor(bid_label, dtype=torch.float32),
            'preference_target': torch.tensor(label.get('placed_rule', 0), dtype=torch.long),
            'weight': torch.tensor(weight, dtype=torch.float32)
        }


def multi_stage_loss(predictions, targets):
    """다중 단계 손실 함수 (최종 결정 + 중간 감독 학습)"""
    choice_probs, final_bid, branch_preferences, branch_costs = predictions
    choice_targets = targets['choice_target']
    bid_targets = targets['bid_target'] 
    preference_targets = targets['preference_target']
    weights = targets['weight']
    
    # 1. 최종 결정 손실 (A/B 선택 + 입찰)
    choice_loss = F.cross_entropy(choice_probs, choice_targets, reduction='none')
    bid_loss = F.mse_loss(final_bid.squeeze(), bid_targets, reduction='none')
    
    # 2. 중간 단계 손실 (브랜치 0의 선호도만 사용)
    # 브랜치 0 = 내가 현재 상황에서의 선호도
    branch_0_preferences = branch_preferences[0]  # (batch_size, 12)
    preference_loss = F.cross_entropy(branch_0_preferences, preference_targets, reduction='none')
    
    # 3. 가중 합산
    total_loss = (2.0 * choice_loss + 
                  1.0 * bid_loss + 
                  2.0 * preference_loss) * weights
    
    return total_loss.mean()


def train_model(model: YachtMasterNet, training_data: YachtTrainingData, 
                epochs: int = 100, batch_size: int = 32, lr: float = 0.001):
    """모델 학습"""
    dataset = YachtDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    print(f"학습 시작: {epochs} 에포크, 배치 크기: {batch_size}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            inputs = batch['inputs']
            targets = {
                'choice_target': batch['choice_target'],
                'bid_target': batch['bid_target'],
                'preference_target': batch['preference_target'],
                'weight': batch['weight']
            }
            
            # Forward pass
            predictions = model(inputs)
            
            # 다중 단계 손실 계산
            loss = multi_stage_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if epoch % 10 == 0:
            print(f"에포크 {epoch}: 평균 손실 = {avg_loss:.4f}")
    
    model.eval()
    print("학습 완료!")


def main():
    """메인 학습 파이프라인"""
    print("Yacht 신경망 학습 시스템")
    
    # 자기대국 엔진 초기화
    engine = SelfPlayEngine()
    
    # 학습 데이터 생성
    training_data = engine.generate_training_data(num_games=500)
    
    # 모델 학습
    model = YachtMasterNet()
    train_model(model, training_data, epochs=50)
    
    # 모델 저장
    model_path = Path(__file__).parent.parent / "yacht_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"모델 저장 완료: {model_path}")


if __name__ == "__main__":
    main()
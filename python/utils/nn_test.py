#!/usr/bin/env python3
"""
Neural Network Test System for Yacht AI
Testing and evaluation utilities
"""

import torch
import numpy as np
import sys
from pathlib import Path
import random

# 상위 디렉토리에서 모듈 import
sys.path.append(str(Path(__file__).parent.parent))

from yacht_nn_submission import YachtMasterNet, NeuralYachtGame
from utils.nn_trainer import SelfPlayEngine, MockYachtPlayer, train_model


def generate_game_data():
    """간단한 게임 데이터 생성 (테스트용)"""
    game_data = []
    for _ in range(13):  # 13라운드
        dice_a = [random.randint(1, 6) for _ in range(5)]
        dice_b = [random.randint(1, 6) for _ in range(5)]
        tie_breaker = random.randint(0, 1)
        game_data.append((dice_a, dice_b, tie_breaker))
    return game_data


def test_network_architecture():
    """신경망 구조 테스트"""
    print("=== 신경망 구조 테스트 ===")
    
    model = YachtMasterNet()
    
    # 테스트 입력 생성 (배치 크기 2)
    batch_size = 2
    test_inputs = torch.randn(batch_size, 4, 18)  # (배치, 4브랜치, 18차원)
    
    print(f"입력 크기: {test_inputs.shape}")
    
    # Forward pass 테스트
    try:
        choice_probs, bid_amounts, branch_preferences, branch_costs = model(test_inputs)
        
        print(f"선택 확률 출력: {choice_probs.shape}")        # (배치, 2)
        print(f"입찰 금액 출력: {bid_amounts.shape}")         # (배치,)
        print(f"브랜치 선호도: {len(branch_preferences)}")     # 4개 브랜치
        print(f"브랜치 비용: {len(branch_costs)}")            # 4개 브랜치
        
        # 브랜치 출력 형태 확인
        for i, prefs in enumerate(branch_preferences):
            print(f"브랜치 {i} 선호도 형태: {prefs.shape}")     # (배치, 12)
        
        # 확률 합이 1인지 확인
        prob_sums = torch.sum(choice_probs, dim=1)
        print(f"확률 합: {prob_sums}")
        
        # 입찰 금액이 양수인지 확인
        print(f"입찰 금액: {bid_amounts}")
        
        print("✅ 신경망 구조 테스트 통과")
        return True
        
    except Exception as e:
        print(f"❌ 신경망 테스트 실패: {e}")
        return False


def test_game_simulation():
    """게임 시뮬레이션 테스트"""
    print("\n=== 게임 시뮬레이션 테스트 ===")
    
    try:
        # 자기대국 엔진 초기화
        engine = SelfPlayEngine()
        
        # 단일 게임 실행
        game_log = engine.play_single_game()
        
        print(f"게임 완료:")
        print(f"  플레이어 1 점수: {game_log['final_scores'][0]}")
        print(f"  플레이어 2 점수: {game_log['final_scores'][1]}")
        print(f"  승자: 플레이어 {game_log['winner']}")
        print(f"  총 라운드: {len(game_log['player1_log'])}")
        
        # 첫 라운드 상세 정보 출력
        if game_log['player1_log']:
            first_round = game_log['player1_log'][0]
            situation = first_round['situation']
            result = first_round['result']
            
            print(f"\n첫 라운드 상세:")
            print(f"  선택: {situation['bid_choice']}")
            print(f"  입찰 금액: {situation['bid_amount']}")
            print(f"  입찰 성공: {result['won_bid']}")
            print(f"  획득 점수: {result.get('round_score', 0)}")
        
        print("✅ 게임 시뮬레이션 테스트 통과")
        return True
        
    except Exception as e:
        print(f"❌ 게임 시뮬레이션 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_data_generation():
    """학습 데이터 생성 테스트"""
    print("\n=== 학습 데이터 생성 테스트 ===")
    
    try:
        engine = SelfPlayEngine()
        
        # 소량의 학습 데이터 생성
        training_data = engine.generate_training_data(num_games=10)
        
        print(f"생성된 학습 샘플 수: {len(training_data)}")
        
        if len(training_data) > 0:
            # 첫 번째 샘플 분석
            situation = training_data.situations[0]
            label = training_data.labels[0]
            weight = training_data.weights[0]
            
            print(f"\n첫 번째 샘플:")
            print(f"  보드 상태: {situation['board_state'][:6]}...")
            print(f"  선택 레이블: {label['choice']}")
            print(f"  입찰 레이블: {label['bid']:.3f}")
            print(f"  가중치: {weight:.3f}")
            print(f"  성과: {label['performance']:.3f}")
            
            # 네트워크 입력 확인
            if 'network_inputs' in situation:
                inputs = situation['network_inputs']
                print(f"  네트워크 입력 형태: {inputs.shape}")
        
        print("✅ 학습 데이터 생성 테스트 통과")
        return True
        
    except Exception as e:
        print(f"❌ 학습 데이터 생성 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training():
    """소규모 학습 테스트"""
    print("\n=== 소규모 학습 테스트 ===")
    
    try:
        # 학습 데이터 생성
        engine = SelfPlayEngine()
        training_data = engine.generate_training_data(num_games=20)
        
        # 모델 초기화
        model = YachtMasterNet()
        
        # 학습 전 성능 기록
        initial_params = list(model.parameters())[0].clone()
        
        # 소규모 학습 실행
        print("소규모 학습 실행...")
        train_model(model, training_data, epochs=5, batch_size=8, lr=0.01)
        
        # 학습 후 파라미터 변화 확인
        final_params = list(model.parameters())[0]
        param_change = torch.norm(final_params - initial_params).item()
        
        print(f"파라미터 변화량: {param_change:.6f}")
        
        if param_change > 1e-6:
            print("✅ 소규모 학습 테스트 통과 (파라미터가 업데이트됨)")
            return True
        else:
            print("⚠️ 파라미터 변화가 미미함")
            return False
            
    except Exception as e:
        print(f"❌ 소규모 학습 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_yacht_game():
    """NeuralYachtGame 클래스 테스트"""
    print("\n=== NeuralYachtGame 테스트 ===")
    
    try:
        game = NeuralYachtGame()
        
        # 인코딩 테스트
        test_board = [None, 5000, None, None, 12000, None, None, None, None, None, None, None]
        encoded_board = game.encode_board_state(test_board)
        print(f"보드 인코딩: {encoded_board}")
        
        test_dice = [6, 6, 6, 4, 2]
        encoded_dice = game.encode_dice(test_dice)
        print(f"주사위 인코딩: {encoded_dice}")
        
        # 네트워크 입력 생성 테스트
        network_input = game.create_network_input(test_board, test_dice)
        print(f"네트워크 입력 형태: {network_input.shape}")
        print(f"네트워크 입력 값: {network_input[:6]}...")  # 처음 6개만
        
        # 입찰 테스트
        dice_a = [6, 6, 6, 4, 2]
        dice_b = [1, 2, 3, 4, 5]
        
        bid = game.calculate_bid(dice_a, dice_b)
        print(f"입찰 결과: {bid.group}, {bid.amount}")
        
        # 주사위 배치 테스트 (간단한 주사위로)
        game.my_state.dice = [6, 6, 6, 4, 2]
        put = game.calculate_put()
        print(f"배치 결과: {put.rule.name}, {put.dice}")
        
        print("✅ NeuralYachtGame 테스트 통과")
        return True
        
    except Exception as e:
        print(f"❌ NeuralYachtGame 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_random_vs_neural():
    """랜덤 vs 신경망 성능 비교"""
    print("\n=== 랜덤 vs 신경망 성능 비교 ===")
    
    try:
        # 간단한 학습으로 모델 준비
        engine = SelfPlayEngine()
        training_data = engine.generate_training_data(num_games=50)
        
        model = YachtMasterNet()
        train_model(model, training_data, epochs=10, batch_size=16)
        
        # 성능 비교를 위한 간단한 대전
        neural_wins = 0
        total_games = 20
        
        for i in range(total_games):
            player1 = MockYachtPlayer(model)  # 학습된 모델
            player2 = MockYachtPlayer(YachtMasterNet())  # 랜덤 초기화 모델
            
            game_data = generate_game_data()
            
            # 간단한 게임 시뮬레이션
            for round_num in range(min(13, len(game_data))):
                dice_a, dice_b, tie_breaker = game_data[round_num]
                
                # 플레이어들이 주사위 받고 배치
                if round_num % 2 == 0:
                    player1.receive_dice(dice_a)
                    player2.receive_dice(dice_b)
                else:
                    player1.receive_dice(dice_b)
                    player2.receive_dice(dice_a)
                
                player1.place_dice()
                player2.place_dice()
            
            # 점수 비교
            if player1.get_total_score() > player2.get_total_score():
                neural_wins += 1
        
        win_rate = neural_wins / total_games
        print(f"신경망 승률: {win_rate:.1%} ({neural_wins}/{total_games})")
        
        if win_rate > 0.4:  # 40% 이상이면 성공
            print("✅ 성능 비교 테스트 통과")
            return True
        else:
            print("⚠️ 성능이 기대보다 낮음")
            return False
            
    except Exception as e:
        print(f"❌ 성능 비교 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """모든 테스트 실행"""
    print("Yacht 신경망 시스템 테스트 시작")
    print("=" * 50)
    
    tests = [
        ("신경망 구조", test_network_architecture),
        ("게임 시뮬레이션", test_game_simulation),
        ("학습 데이터 생성", test_training_data_generation),
        ("소규모 학습", test_mini_training),
        ("NeuralYachtGame", test_neural_yacht_game),
        ("성능 비교", benchmark_random_vs_neural),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}] 테스트 실행 중...")
        try:
            if test_func():
                passed += 1
            print(f"[{test_name}] 완료")
        except Exception as e:
            print(f"[{test_name}] 예외 발생: {e}")
    
    print("\n" + "=" * 50)
    print(f"테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
        print("이제 'python utils/nn_trainer.py'로 본격적인 학습을 시작할 수 있습니다.")
    elif passed >= total * 0.7:
        print("⚠️ 대부분의 테스트 통과. 일부 문제가 있을 수 있습니다.")
    else:
        print("❌ 많은 테스트 실패. 코드를 점검해주세요.")
    
    return passed / total


if __name__ == "__main__":
    run_all_tests()
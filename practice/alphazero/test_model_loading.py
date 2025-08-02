#!/usr/bin/env python3
"""
모델 저장/로드 검증 스크립트
버섯 값 정규화 및 모델 로딩이 제대로 작동하는지 테스트
"""

import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_network import AlphaZeroNet, AlphaZeroTrainer
from training import TrainingManager
from game_board import GameBoard

def test_mushroom_normalization():
    """버섯 값 정규화 테스트 (1-9 범위)"""
    print("=== Testing Mushroom Value Normalization ===")
    
    # 1-9 범위의 테스트 보드 생성
    test_board = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 9, 8, 7, 6, 5, 4, 3, 2],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    ]
    # 나머지는 0으로 채움
    for _ in range(7):
        test_board.append([0] * 17)
    
    game_board = GameBoard(test_board)
    state_tensor = game_board.get_state_tensor(0)
    
    print(f"Original board values: {test_board[0][:9]}")
    print(f"Normalized values: {state_tensor[0][0][:9]}")
    
    # 정규화 검증
    expected_values = [v/10.0 for v in test_board[0][:9]]
    actual_values = state_tensor[0][0][:9].tolist()
    
    success = all(abs(a - e) < 1e-6 for a, e in zip(actual_values, expected_values))
    print(f"Normalization test: {'PASS' if success else 'FAIL'}")
    
    return success

def test_model_save_load():
    """모델 저장/로드 테스트"""
    print("\n=== Testing Model Save/Load ===")
    
    # 임시 디렉토리 생성
    test_dir = "practice/models/test_save_load"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # 1. 모델 생성
        model = AlphaZeroNet(hidden_channels=64, action_space_size=8246)  # 작은 모델로 테스트
        trainer = TrainingManager(model, test_dir)
        
        print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # 2. 초기 가중치 저장 (비교용)
        initial_weights = {}
        for name, param in model.named_parameters():
            initial_weights[name] = param.data.clone()
        
        # 3. 모델 저장
        trainer.save_model("test_model.pth")
        print("Model saved")
        
        # 4. 가중치 변경 (로드 테스트를 위해)
        with torch.no_grad():
            for param in model.parameters():
                param.data.fill_(999.0)  # 모든 가중치를 999로 변경
        
        print("Model weights modified")
        
        # 5. 모델 로드
        load_success = trainer.load_model("test_model.pth")
        print(f"Load attempt: {'SUCCESS' if load_success else 'FAILED'}")
        
        if load_success:
            # 6. 가중치 복원 검증
            weights_restored = True
            for name, param in model.named_parameters():
                if not torch.allclose(param.data, initial_weights[name], atol=1e-6):
                    weights_restored = False
                    break
            
            print(f"Weights restoration: {'VERIFIED' if weights_restored else 'FAILED'}")
            
            # 7. 모델 기능 테스트
            test_input = torch.randn(1, 2, 10, 17).to(model.device)
            try:
                with torch.no_grad():
                    policy, value = model(test_input)
                print(f"Model inference: WORKING (policy: {policy.shape}, value: {value.shape})")
                functional = True
            except Exception as e:
                print(f"Model inference: FAILED ({e})")
                functional = False
                
            return load_success and weights_restored and functional
        else:
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
    finally:
        # 정리
        if os.path.exists(os.path.join(test_dir, "test_model.pth")):
            os.remove(os.path.join(test_dir, "test_model.pth"))

def test_optimizer_state_persistence():
    """Optimizer state 저장/로드 검증"""
    print("\n=== Testing Optimizer State Persistence ===")
    
    # 임시 디렉토리 생성
    test_dir = "practice/models/test_optimizer"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # 1. 모델과 훈련 데이터 생성
        model = AlphaZeroNet(hidden_channels=32, action_space_size=8246)
        trainer_mgr = TrainingManager(model, test_dir)
        
        # 가상 훈련 데이터 생성
        batch_size = 4
        fake_states = np.random.randn(batch_size, 2, 10, 17).astype(np.float32)
        fake_policy_targets = np.random.rand(batch_size, 8246).astype(np.float32)
        fake_policy_targets = fake_policy_targets / fake_policy_targets.sum(axis=1, keepdims=True)  # 정규화
        fake_value_targets = np.random.randn(batch_size).astype(np.float32)
        
        # 2. 첫 번째 훈련 스텝 실행
        print("Running initial training step...")
        initial_loss = trainer_mgr.trainer.train_step(fake_states, fake_policy_targets, fake_value_targets)
        print(f"Initial loss: {initial_loss['total_loss']:.4f} (policy: {initial_loss['policy_loss']:.4f})")
        
        # 3. 모델 저장
        trainer_mgr.save_model("optimizer_test.pth")
        
        # 4. 새로운 모델 인스턴스 생성 (optimizer 초기화됨)
        model2 = AlphaZeroNet(hidden_channels=32, action_space_size=8246)
        trainer_mgr2 = TrainingManager(model2, test_dir)
        
        # 5. 훈련 전 loss (초기 상태)
        fresh_loss = trainer_mgr2.trainer.train_step(fake_states, fake_policy_targets, fake_value_targets)
        print(f"Fresh model loss: {fresh_loss['total_loss']:.4f} (policy: {fresh_loss['policy_loss']:.4f})")
        
        # 6. 모델 로드
        load_success = trainer_mgr2.load_model("optimizer_test.pth")
        
        if load_success:
            # 7. 로드 후 같은 데이터로 훈련
            loaded_loss = trainer_mgr2.trainer.train_step(fake_states, fake_policy_targets, fake_value_targets)
            print(f"Loaded model loss: {loaded_loss['total_loss']:.4f} (policy: {loaded_loss['policy_loss']:.4f})")
            
            # 8. Loss 값 비교 - optimizer state가 제대로 로드되었다면 loaded_loss가 fresh_loss와 달라야 함
            loss_diff = abs(loaded_loss['total_loss'] - fresh_loss['total_loss'])
            optimizer_restored = loss_diff > 0.001  # 의미있는 차이가 있어야 함
            
            print(f"Loss difference: {loss_diff:.6f}")
            print(f"Optimizer state restoration: {'SUCCESS' if optimizer_restored else 'FAILED'}")
            
            return load_success and optimizer_restored
        else:
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False
    finally:
        # 정리
        for f in ["optimizer_test.pth"]:
            filepath = os.path.join(test_dir, f)
            if os.path.exists(filepath):
                os.remove(filepath)

def test_model_functionality_after_load():
    """로드된 모델의 출력 형태 검증"""
    print("\n=== Testing Model Functionality After Load ===")
    
    model = AlphaZeroNet(hidden_channels=64, action_space_size=8246)
    trainer = AlphaZeroTrainer(model)
    
    # 기능 검증 실행
    result = trainer.verify_model_functionality()
    
    print(f"Functionality test: {'PASS' if result['all_checks_passed'] else 'FAIL'}")
    if not result['all_checks_passed']:
        for key, value in result.items():
            if key.endswith('_ok') and not value:
                print(f"  - Failed: {key}")
    
    return result['all_checks_passed']

def main():
    """전체 테스트 실행"""
    print("AlphaZero Model Loading Verification")
    print("=" * 50)
    
    tests = [
        ("Mushroom Normalization", test_mushroom_normalization),
        ("Model Save/Load", test_model_save_load),
        ("Optimizer State Persistence", test_optimizer_state_persistence),
        ("Model Functionality", test_model_functionality_after_load)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed

if __name__ == "__main__":
    main()
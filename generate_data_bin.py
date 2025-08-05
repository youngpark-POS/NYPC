#!/usr/bin/env python3
"""
latest_model.pth에서 submission.py용 data.bin 생성
"""

import sys
import os
sys.path.append('practice/alphazero')

from neural_network import AlphaZeroNet, AlphaZeroTrainer

def main():
    print("Creating data.bin from latest_model.pth...")
    
    # 모델 로드
    model = AlphaZeroNet()
    trainer = AlphaZeroTrainer(model)
    
    model_path = "practice/models/test/latest_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
    
    # 모델 로드
    trainer.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # data.bin으로 저장
    output_path = "practice/alphazero/data.bin"
    trainer.save_model_as_binary(output_path)
    print(f"Binary model saved to {output_path}")

if __name__ == "__main__":
    main()
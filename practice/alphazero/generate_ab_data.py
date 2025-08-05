#!/usr/bin/env python3
"""
AB Pruning 자기대국 훈련 데이터 생성 메인 스크립트
"""

import argparse
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any

from ab_self_play import ABSelfPlayGenerator
from game_history import GameHistoryManager
from compact_data import CompactDataConverter


class ABDataGenerator:
    """AB pruning 데이터 생성 메인 클래스"""
    
    def __init__(self, config_path: str = "practice/alphazero/ab_config.json"):
        """설정 파일 로드"""
        self.config = self._load_config(config_path)
        self.converter = CompactDataConverter()
        # generator와 history_manager는 설정 완료 후 initialize_components에서 초기화
        self.generator = None
        self.history_manager = None
    
    def initialize_components(self):
        """설정 완료 후 컴포넌트 초기화"""
        # AB 정책 분배 설정
        policy_config = self.config.get("policy_distribution", {})
        ab_move_ratio = policy_config.get("ab_move_ratio", 0.85)
        noise_ratio = policy_config.get("noise_ratio", 0.15)
        
        self.generator = ABSelfPlayGenerator(
            ab_executable=self.config["ab_executable"],
            ab_move_ratio=ab_move_ratio,
            noise_ratio=noise_ratio
        )
        
        # 게임 히스토리 매니저 초기화
        self.history_manager = GameHistoryManager(
            storage_path=self.config["output_path"]
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using default configuration...")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            print("Using default configuration...")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            "ab_executable": "practice/testing/ab_pruning.exe",
            "input_boards_file": "practice/testing/input.txt",
            "output_path": "practice/models/test/game_history.h5",
            "games_to_generate": 50,
            "timeout_per_game": 300,
            "verbose": True,
            "policy_distribution": {
                "ab_move_ratio": 0.85,
                "noise_ratio": 0.15
            },
            "board_generation": {
                "use_random_boards": True,
                "random_seed": None,
                "board_count": 10,
                "min_mushroom_value": 1,
                "max_mushroom_value": 9
            }
        }
    
    def generate_boards(self, count: int) -> List[List[List[int]]]:
        """게임 보드 생성 (항상 랜덤 보드)"""
        boards = []
        
        if self.config["verbose"]:
            print(f"Generating {count} random boards...")
        
        seed = self.config["board_generation"]["random_seed"]
        if seed is not None:
            np.random.seed(seed)
        
        min_val = self.config["board_generation"]["min_mushroom_value"]
        max_val = self.config["board_generation"]["max_mushroom_value"]
        
        for i in range(count):
            board = np.random.randint(min_val, max_val + 1, size=(10, 17)).tolist()
            boards.append(board)
        
        return boards
    
    def _load_board_from_file(self, input_file: Path) -> List[List[int]]:
        """파일에서 보드 로드"""
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        board = []
        for line in lines:
            line = line.strip()
            if line:
                row = [int(c) for c in line if c.isdigit()]
                if len(row) == 17:  # 올바른 열 수 확인
                    board.append(row)
        
        if len(board) != 10:  # 올바른 행 수 확인
            raise ValueError(f"Invalid board dimensions: {len(board)}x{len(board[0]) if board else 0}")
        
        return board
    
    def generate_data(self, games_count: int = None) -> Dict[str, Any]:
        """AB 자기대국 데이터 생성"""
        # 설정 완료 후 컴포넌트 초기화
        self.initialize_components()
        
        if games_count is None:
            games_count = self.config["games_to_generate"]
        
        if self.config["verbose"]:
            print(f"=== AB Pruning 자기대국 데이터 생성 ===")
            print(f"생성할 게임 수: {games_count}")
            print(f"AB 실행파일: {self.config['ab_executable']}")
            print(f"출력 경로: {self.config['output_path']}")
            print()
        
        # 보드 생성
        boards = self.generate_boards(games_count)
        
        if not boards:
            print("Error: No boards available for game generation")
            return {"success": False, "error": "No boards available"}
        
        # 기존 데이터 통계
        try:
            existing_stats = self.history_manager.get_storage_stats()
            if self.config["verbose"]:
                print(f"기존 데이터: {existing_stats['total_games']}게임")
        except:
            existing_stats = {"total_games": 0}
        
        # 게임 생성 시작
        start_time = time.time()
        total_games_saved = 0
        batch_size = 500  # 500게임마다 저장
        
        try:
            if self.config["verbose"]:
                print(f"AB 자기대국 게임 생성 중... (배치 크기: {batch_size})")
            
            # 배치별로 게임 생성 및 저장
            for batch_start in range(0, games_count, batch_size):
                batch_end = min(batch_start + batch_size, games_count)
                batch_games_count = batch_end - batch_start
                
                if self.config["verbose"]:
                    print(f"\n=== 배치 {batch_start//batch_size + 1} ({batch_start+1}-{batch_end}/{games_count}) ===")
                
                # 현재 배치용 보드 생성
                batch_boards = boards[batch_start:batch_end]
                
                # 배치 게임 생성 (내부 배치 크기 50으로 제한)
                batch_games_data = self.generator.generate_games(
                    batch_boards, 
                    timeout_per_game=self.config["timeout_per_game"],
                    verbose=self.config["verbose"],
                    batch_size=50  # 내부 배치 크기 제한으로 메모리 안정성 확보
                )
                
                if not batch_games_data:
                    if self.config["verbose"]:
                        print(f"배치 {batch_start//batch_size + 1}: 생성된 게임이 없습니다.")
                    continue
                
                if self.config["verbose"]:
                    print(f"배치 {batch_start//batch_size + 1}: {len(batch_games_data)}개 게임 생성 완료. 변환 및 저장 중...")
                
                # CompactSelfPlayData로 변환
                batch_generated_games = []
                for i, game_data in enumerate(batch_games_data):
                    if self.config["verbose"] and (i % 50 == 0 or i == len(batch_games_data) - 1):
                        print(f"  변환 진행중... ({i+1}/{len(batch_games_data)})")
                    
                    compact_data = self.converter.from_self_play_data(game_data)
                    batch_generated_games.append(compact_data)
                
                # 배치 저장
                if batch_generated_games:
                    try:
                        self.history_manager.save_games(batch_generated_games)
                        total_games_saved += len(batch_generated_games)
                        
                        if self.config["verbose"]:
                            print(f"배치 {batch_start//batch_size + 1}: {len(batch_generated_games)}개 게임 저장 완료!")
                            print(f"누적 저장: {total_games_saved}개 게임")
                    
                    except Exception as e:
                        print(f"배치 {batch_start//batch_size + 1} 저장 중 오류: {e}")
                        # 부분 실패 시에도 계속 진행
                        continue
                
                # 메모리 해제 및 가비지 컬렉션
                del batch_games_data
                del batch_generated_games
                import gc
                gc.collect()
            
        except KeyboardInterrupt:
            if self.config["verbose"]:
                print(f"\n사용자 중단. 현재까지 {total_games_saved}개 게임이 저장되었습니다.")
        
        except Exception as e:
            print(f"Error during data generation: {e}")
            if self.config["verbose"] and total_games_saved > 0:
                print(f"부분적으로 {total_games_saved}개 게임이 저장되었습니다.")
        
        # 결과 확인
        if total_games_saved == 0:
            if self.config["verbose"]:
                print("저장된 게임이 없습니다.")
            return {"success": False, "error": "No games saved"}
        
        # 결과 통계
        end_time = time.time()
        total_time = end_time - start_time
        
        final_stats = self.history_manager.get_storage_stats()
        
        result = {
            "success": True,
            "games_generated": total_games_saved,
            "total_time": total_time,
            "avg_time_per_game": total_time / max(total_games_saved, 1),
            "existing_games_before": existing_stats["total_games"],
            "total_games_after": final_stats["total_games"],
            "output_path": self.config["output_path"],
            "batch_size": batch_size,
            "batches_processed": (total_games_saved + batch_size - 1) // batch_size
        }
        
        if self.config["verbose"]:
            print(f"\n=== 배치 생성 완료 ===")
            print(f"저장된 게임: {result['games_generated']}개")
            print(f"처리된 배치: {result['batches_processed']}개 (배치당 {batch_size}게임)")
            print(f"총 소요시간: {total_time:.1f}초")
            print(f"게임당 평균시간: {result['avg_time_per_game']:.1f}초")
            print(f"배치당 평균시간: {total_time / max(result['batches_processed'], 1):.1f}초")
            print(f"전체 데이터: {result['existing_games_before']} → {result['total_games_after']}게임")
            print(f"저장 위치: {result['output_path']}")
        
        return result


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AB Pruning 자기대국 훈련 데이터 생성")
    parser.add_argument("--games", "-g", type=int, help="생성할 게임 수")
    parser.add_argument("--config", "-c", default="practice/alphazero/ab_config.json", 
                       help="설정 파일 경로")
    parser.add_argument("--output", "-o", help="출력 파일 경로")
    parser.add_argument("--ab-ratio", type=float, help="AB 선택 움직임 확률 (0.0-1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 출력")
    parser.add_argument("--quiet", "-q", action="store_true", help="최소 출력")
    
    args = parser.parse_args()
    
    try:
        # 데이터 생성기 초기화
        generator = ABDataGenerator(args.config)
        
        # 명령행 인수로 설정 오버라이드
        if args.games:
            generator.config["games_to_generate"] = args.games
        if args.output:
            generator.config["output_path"] = args.output
        if args.ab_ratio:
            # AB 비율 조정 (노이즈 비율은 자동 계산)
            ab_ratio = max(0.0, min(1.0, args.ab_ratio))  # 0-1 범위로 제한
            noise_ratio = 1.0 - ab_ratio
            generator.config["policy_distribution"]["ab_move_ratio"] = ab_ratio
            generator.config["policy_distribution"]["noise_ratio"] = noise_ratio
        if args.verbose:
            generator.config["verbose"] = True
        if args.quiet:
            generator.config["verbose"] = False
        
        # 데이터 생성 실행
        result = generator.generate_data()
        
        if result["success"]:
            if not args.quiet:
                batches_info = f" ({result['batches_processed']}배치)" if result.get('batches_processed', 0) > 1 else ""
                print(f"성공: {result['games_generated']}개 게임 생성됨{batches_info}")
            exit(0)
        else:
            print(f"실패: {result.get('error', 'Unknown error')}")
            exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
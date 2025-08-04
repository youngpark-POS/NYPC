#!/usr/bin/env python3
"""
게임 히스토리 관리 시스템
HDF5 기반으로 최신 10,000게임을 순환 버퍼 형태로 저장/관리
"""

import h5py
import numpy as np
import os
import time
import json
from typing import List, Dict, Tuple, Optional, Generator
from pathlib import Path
import threading
from dataclasses import asdict

from compact_data import CompactSelfPlayData, CompactGameState, CompactDataConverter


class GameHistoryManager:
    """게임 히스토리 저장 및 관리 클래스"""
    
    def __init__(self, 
                 storage_path: str = "practice/models/game_history.h5",
                 max_games: int = 10000,
                 compression: str = "gzip",
                 compression_level: int = 6):
        """
        Args:
            storage_path: HDF5 파일 저장 경로
            max_games: 최대 저장할 게임 수 (순환 버퍼)
            compression: 압축 방식 ('gzip', 'lzf', None)
            compression_level: 압축 레벨 (0-9, gzip 전용)
        """
        self.storage_path = Path(storage_path)
        self.max_games = max_games
        self.compression = compression
        self.compression_opts = compression_level if compression == 'gzip' else None
        
        # 스레드 안전성을 위한 락
        self._lock = threading.RLock()
        
        # 메타데이터 캐시
        self._metadata_cache = {
            'total_games': 0,
            'next_index': 0,
            'schema_version': '1.0'
        }
        
        # 저장 디렉토리 생성
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # HDF5 파일 초기화
        self._initialize_storage()
    
    def _initialize_storage(self):
        """HDF5 파일 및 기본 구조 초기화"""
        with self._lock:
            if not self.storage_path.exists():
                self._create_new_storage()
            else:
                self._load_metadata()
                self._validate_storage()
    
    def _create_new_storage(self):
        """새로운 HDF5 저장소 생성"""
        print(f"Creating new game history storage: {self.storage_path}")
        
        with h5py.File(self.storage_path, 'w') as f:
            # 메타데이터 그룹 생성
            meta_group = f.create_group('metadata')
            meta_group.attrs['total_games'] = 0
            meta_group.attrs['next_index'] = 0
            meta_group.attrs['max_games'] = self.max_games
            meta_group.attrs['schema_version'] = '1.0'
            meta_group.attrs['created_at'] = time.time()
            
            # 게임 저장용 그룹 생성
            f.create_group('games')
            
            # 인덱스 그룹 생성
            index_group = f.create_group('index')
            
            # 게임 ID 순서 저장 (순환 버퍼 관리용)
            index_group.create_dataset(
                'game_order', 
                (self.max_games,), 
                dtype='i4', 
                fillvalue=-1,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            
            # 타임스탬프 저장
            index_group.create_dataset(
                'timestamps',
                (self.max_games,),
                dtype='f8',
                fillvalue=0.0,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
        
        self._metadata_cache = {
            'total_games': 0,
            'next_index': 0,
            'schema_version': '1.0'
        }
    
    def _load_metadata(self):
        """저장된 메타데이터 로드"""
        with h5py.File(self.storage_path, 'r') as f:
            meta_group = f['metadata']
            self._metadata_cache = {
                'total_games': meta_group.attrs['total_games'],
                'next_index': meta_group.attrs['next_index'],
                'schema_version': meta_group.attrs.get('schema_version', '1.0')
            }
    
    def _save_metadata(self):
        """메타데이터를 HDF5 파일에 저장"""
        with h5py.File(self.storage_path, 'r+') as f:
            meta_group = f['metadata']
            meta_group.attrs['total_games'] = self._metadata_cache['total_games']
            meta_group.attrs['next_index'] = self._metadata_cache['next_index']
    
    def _validate_storage(self):
        """저장소 무결성 검증"""
        try:
            with h5py.File(self.storage_path, 'r') as f:
                # 필수 그룹들이 존재하는지 확인
                required_groups = ['metadata', 'games', 'index']
                for group_name in required_groups:
                    if group_name not in f:
                        raise ValueError(f"Missing required group: {group_name}")
                
                # 스키마 버전 확인
                schema_version = f['metadata'].attrs.get('schema_version', '1.0')
                if schema_version != '1.0':
                    print(f"Warning: Schema version mismatch. Expected 1.0, got {schema_version}")
        
        except Exception as e:
            print(f"Storage validation failed: {e}")
            print("Creating new storage...")
            self.storage_path.unlink(missing_ok=True)
            self._create_new_storage()
    
    def save_games(self, games_data: List) -> Dict[str, int]:
        """게임 데이터들을 저장 (SelfPlayData 또는 CompactSelfPlayData)"""
        with self._lock:
            saved_count = 0
            skipped_count = 0
            
            for game_data in games_data:
                # CompactSelfPlayData로 변환
                if hasattr(game_data, 'game_states'):  # SelfPlayData
                    compact_game = CompactDataConverter.from_self_play_data(game_data)
                elif hasattr(game_data, 'compact_game_states'):  # 이미 CompactSelfPlayData
                    compact_game = game_data
                else:
                    print(f"Warning: Unknown game data type: {type(game_data)}")
                    skipped_count += 1
                    continue
                
                # 개별 게임 저장
                if self._save_single_game(compact_game):
                    saved_count += 1
                else:
                    skipped_count += 1
            
            # 메타데이터 업데이트
            self._save_metadata()
            
            return {
                'saved': saved_count,
                'skipped': skipped_count,
                'total_games': self._metadata_cache['total_games']
            }
    
    def _save_single_game(self, compact_game: CompactSelfPlayData) -> bool:
        """단일 게임을 HDF5에 저장"""
        try:
            current_index = self._metadata_cache['next_index']
            game_id = f"game_{current_index:06d}"
            
            with h5py.File(self.storage_path, 'r+') as f:
                games_group = f['games']
                
                # 이전 게임 데이터가 있다면 삭제 (순환 버퍼)
                if game_id in games_group:
                    del games_group[game_id]
                
                # 새 게임 그룹 생성
                game_group = games_group.create_group(game_id)
                
                # 게임 메타데이터 저장
                game_group.attrs['winner'] = compact_game.winner
                game_group.attrs['game_length'] = compact_game.game_length
                game_group.attrs['final_score_0'] = compact_game.final_score[0]
                game_group.attrs['final_score_1'] = compact_game.final_score[1]
                game_group.attrs['total_simulations'] = compact_game.total_simulations
                game_group.attrs['average_simulations'] = compact_game.average_simulations
                
                # final_result 저장 (JSON 형태)
                game_group.attrs['final_result'] = json.dumps(compact_game.final_result)
                
                # 게임 상태들 저장
                num_states = len(compact_game.compact_game_states)
                if num_states == 0:
                    return False
                
                # 보드 상태들 저장 (N, 10, 17)
                board_states = np.array([state.board_state for state in compact_game.compact_game_states], dtype=np.int8)
                game_group.create_dataset(
                    'board_states',
                    data=board_states,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                
                # 선택된 움직임들 저장 (N, 4)
                selected_moves = np.array([state.move_coords for state in compact_game.compact_game_states], dtype=np.int8)
                game_group.create_dataset(
                    'selected_moves',
                    data=selected_moves,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                
                # 플레이어 정보 저장 (N,)
                players = np.array([state.player for state in compact_game.compact_game_states], dtype=np.int8)
                game_group.create_dataset(
                    'players',
                    data=players,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                
                # 움직임 번호 저장 (N,)
                move_numbers = np.array([state.move_number for state in compact_game.compact_game_states], dtype=np.int16)
                game_group.create_dataset(
                    'move_numbers',
                    data=move_numbers,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                
                # MCTS 시뮬레이션 횟수 저장 (N,)
                simulations = np.array([state.mcts_simulations for state in compact_game.compact_game_states], dtype=np.int16)
                game_group.create_dataset(
                    'mcts_simulations',
                    data=simulations,
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                
                # Sparse 유효 움직임 및 확률 저장 (가변 길이)
                self._save_sparse_moves_data(game_group, compact_game.compact_game_states)
                
                # 인덱스 업데이트
                index_group = f['index']
                index_group['game_order'][current_index] = current_index
                index_group['timestamps'][current_index] = time.time()
            
            # 메타데이터 업데이트
            self._metadata_cache['next_index'] = (current_index + 1) % self.max_games
            if self._metadata_cache['total_games'] < self.max_games:
                self._metadata_cache['total_games'] += 1
            
            return True
            
        except Exception as e:
            print(f"Error saving game: {e}")
            return False
    
    def _save_sparse_moves_data(self, game_group, compact_states: List[CompactGameState]):
        """가변 길이 유효 움직임 데이터를 효율적으로 저장"""
        # 각 상태별 유효 움직임 개수 저장
        moves_counts = [len(state.valid_moves) for state in compact_states]
        game_group.create_dataset(
            'valid_moves_counts',
            data=np.array(moves_counts, dtype=np.int16),
            compression=self.compression,
            compression_opts=self.compression_opts
        )
        
        # 모든 유효 움직임을 하나의 배열로 결합
        all_moves = []
        all_probs = []
        
        for state in compact_states:
            all_moves.extend(state.valid_moves)
            all_probs.extend(state.move_probabilities)
        
        if all_moves:
            # 움직임 좌표들 저장 (total_moves, 4)
            moves_array = np.array(all_moves, dtype=np.int8)
            game_group.create_dataset(
                'valid_moves_data',
                data=moves_array,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            
            # 확률들 저장 (total_moves,)
            probs_array = np.array(all_probs, dtype=np.float32)
            game_group.create_dataset(
                'move_probabilities_data',
                data=probs_array,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
    
    def load_recent_games(self, count: int = 1000) -> List[CompactSelfPlayData]:
        """최근 게임들을 로드"""
        with self._lock:
            if count <= 0:
                return []
            
            # 실제 로드할 게임 수 결정
            available_games = min(count, self._metadata_cache['total_games'])
            if available_games == 0:
                return []
            
            games = []
            current_index = self._metadata_cache['next_index']
            
            # 최근 게임부터 역순으로 로드
            for i in range(available_games):
                game_index = (current_index - 1 - i) % self.max_games
                if game_index < 0:
                    game_index += self.max_games
                
                game_data = self._load_single_game(game_index)
                if game_data:
                    games.append(game_data)
            
            print(f"Loaded {len(games)} recent games")
            return games
    
    def _load_single_game(self, game_index: int) -> Optional[CompactSelfPlayData]:
        """단일 게임을 로드"""
        try:
            game_id = f"game_{game_index:06d}"
            
            with h5py.File(self.storage_path, 'r') as f:
                games_group = f['games']
                
                if game_id not in games_group:
                    return None
                
                game_group = games_group[game_id]
                
                # 게임 메타데이터 로드
                winner = game_group.attrs['winner']
                game_length = game_group.attrs['game_length']
                final_score = (
                    game_group.attrs['final_score_0'],
                    game_group.attrs['final_score_1']
                )
                total_simulations = game_group.attrs['total_simulations']
                average_simulations = game_group.attrs['average_simulations']
                final_result = json.loads(game_group.attrs['final_result'])
                
                # 게임 상태 데이터 로드
                board_states = game_group['board_states'][:]  # (N, 10, 17)
                selected_moves = game_group['selected_moves'][:]  # (N, 4)
                players = game_group['players'][:]  # (N,)
                move_numbers = game_group['move_numbers'][:]  # (N,)
                simulations = game_group['mcts_simulations'][:]  # (N,)
                
                # Sparse 움직임 데이터 로드
                moves_counts = game_group['valid_moves_counts'][:]
                valid_moves_data = game_group['valid_moves_data'][:]
                move_probs_data = game_group['move_probabilities_data'][:]
                
                # CompactGameState들 재구성
                compact_states = []
                move_data_offset = 0
                
                for i in range(len(board_states)):
                    moves_count = moves_counts[i]
                    
                    # 해당 상태의 유효 움직임들 추출
                    state_moves = []
                    state_probs = []
                    
                    for j in range(moves_count):
                        idx = move_data_offset + j
                        move = tuple(valid_moves_data[idx])
                        prob = float(move_probs_data[idx])
                        state_moves.append(move)
                        state_probs.append(prob)
                    
                    move_data_offset += moves_count
                    
                    # CompactGameState 생성
                    compact_state = CompactGameState(
                        board_state=board_states[i].tolist(),
                        move_coords=tuple(selected_moves[i]),
                        valid_moves=state_moves,
                        move_probabilities=state_probs,
                        player=int(players[i]),
                        move_number=int(move_numbers[i]),
                        mcts_simulations=int(simulations[i]),
                        valid_moves_count=moves_count
                    )
                    
                    compact_states.append(compact_state)
                
                # CompactSelfPlayData 생성
                return CompactSelfPlayData(
                    compact_game_states=compact_states,
                    final_result=final_result,
                    game_length=game_length,
                    final_score=final_score,
                    winner=winner,
                    total_simulations=total_simulations,
                    average_simulations=average_simulations
                )
                
        except Exception as e:
            print(f"Error loading game {game_index}: {e}")
            return None
    
    def get_training_batch(self, batch_size: int = 1000, mix_recent: bool = True) -> List:
        """훈련용 배치 데이터 생성 (기존 SelfPlayData 형태로 반환)"""
        # 최근 게임들 로드
        if mix_recent:
            recent_games = self.load_recent_games(batch_size)
        else:
            # 랜덤 샘플링 (구현 예정)
            recent_games = self.load_recent_games(batch_size)
        
        # CompactSelfPlayData를 SelfPlayData로 변환
        training_data = []
        for compact_game in recent_games:
            original_game = CompactDataConverter.to_self_play_data(compact_game)
            training_data.append(original_game)
        
        return training_data
    
    def get_storage_stats(self) -> Dict:
        """저장소 통계 정보 반환"""
        with self._lock:
            file_size = self.storage_path.stat().st_size if self.storage_path.exists() else 0
            
            return {
                'storage_path': str(self.storage_path),
                'total_games': self._metadata_cache['total_games'],
                'max_games': self.max_games,
                'next_index': self._metadata_cache['next_index'],
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'schema_version': self._metadata_cache['schema_version'],
                'is_full': self._metadata_cache['total_games'] >= self.max_games
            }
    
    def cleanup_old_games(self, keep_games: int = None) -> int:
        """오래된 게임들 정리 (수동 호출용)"""
        if keep_games is None:
            keep_games = self.max_games
        
        with self._lock:
            current_games = self._metadata_cache['total_games']
            if current_games <= keep_games:
                return 0
            
            # 실제로는 순환 버퍼가 자동으로 처리하므로 여기서는 통계만 반환
            return max(0, current_games - keep_games)
    
    def __enter__(self):
        """Context manager 지원"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 지원"""
        # 필요시 정리 작업
        pass


if __name__ == "__main__":
    # 간단한 테스트
    print("GameHistoryManager 테스트")
    print("=" * 50)
    
    # 테스트용 매니저 생성
    test_path = "test_game_history.h5"
    manager = GameHistoryManager(test_path, max_games=100)
    
    # 통계 출력
    stats = manager.get_storage_stats()
    print(f"Storage stats: {stats}")
    
    # 정리
    if Path(test_path).exists():
        Path(test_path).unlink()
    
    print("테스트 완료!")
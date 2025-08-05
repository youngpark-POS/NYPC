#!/usr/bin/env python3
"""
AB Pruning 자기대국을 통한 훈련 데이터 생성 (testing_tool 로직 내장)
"""

import subprocess
import time
import queue
import threading
from typing import List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from threading import Lock
import gc

from ab_data_converter import ABLogConverter
from self_play import SelfPlayData


class Player:
    """testing_tool.py의 Player 클래스 개선 (스레드 정리 추가)"""
    def __init__(self, exec: str):
        self.exec = exec
        self._shutdown = False
        try:
            self.process = subprocess.Popen(
                self.exec,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                shell=True
            )
        except Exception as e:
            print(f'Error: Failed to start process: {e}')
            raise e
        self.reads = queue.Queue()
        self.writes = queue.Queue()

        self.stdin_thread = threading.Thread(target=self.__handle_stdin)
        self.stdout_thread = threading.Thread(target=self.__handle_stdout)
        self.stdin_thread.daemon = True
        self.stdout_thread.daemon = True
        self.stdin_thread.start()
        self.stdout_thread.start()

    def __handle_stdin(self):
        while not self._shutdown:
            try:
                item = self.writes.get(timeout=0.1)
                if item is None:  # Poison pill to stop thread
                    break
                self.process.stdin.write(item)
                self.process.stdin.flush()
            except queue.Empty:
                continue
            except:
                break

    def __handle_stdout(self):
        while not self._shutdown:
            try:
                line = self.process.stdout.readline()
                if not line:  # EOF
                    break
                self.reads.put(line)
            except:
                break

    def print(self, message: str):
        if not self._shutdown:
            self.writes.put(f'{message}\n')

    def readline(self, timeout: float) -> Tuple[float, str] | None:
        try:
            start = time.time()
            content = self.reads.get(timeout=timeout)
            return (time.time() - start, content)
        except queue.Empty:
            return None
    
    def cleanup(self):
        """리소스 정리"""
        if self._shutdown:
            return
            
        self._shutdown = True
        
        # 스레드 종료 신호
        try:
            self.writes.put(None)  # Poison pill for stdin thread
        except:
            pass
        
        # 프로세스 종료
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        except:
            pass
        
        # 스레드 종료 대기 (최대 1초)
        for thread in [self.stdin_thread, self.stdout_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=1)
        
        # Queue 정리
        try:
            while not self.reads.empty():
                self.reads.get_nowait()
        except:
            pass
        try:
            while not self.writes.empty():
                self.writes.get_nowait()
        except:
            pass
    
    def __del__(self):
        """소멸자"""
        self.cleanup()

    @classmethod
    def readAll(cls, selfs: List['Player'], timeout: float) -> List[Tuple[float, str] | None]:
        def __readline_thread(p: 'Player', timeout: float, idx: int, arr: List[Tuple[float, str]|None]):
            arr[idx] = p.readline(timeout)

        readline_threads = []
        returns = [None] * len(selfs)
        for i, p in enumerate(selfs):
            readline_threads.append(threading.Thread(target=__readline_thread, args=(p, timeout, i, returns)))

        for thread in readline_threads:
            thread.start()

        for thread in readline_threads:
            thread.join()

        return returns


class ABProcessPool:
    """AB 프로세스 풀 관리"""
    
    def __init__(self, ab_executable: str, pool_size: int = None):
        self.ab_executable = ab_executable
        self.pool_size = pool_size or min(8, multiprocessing.cpu_count())
        self.available_processes = queue.Queue()
        self.lock = Lock()
        self.is_initialized = False
        
    def initialize(self):
        """프로세스 풀 초기화"""
        if self.is_initialized:
            return
            
        for i in range(self.pool_size):
            try:
                player1 = Player(self.ab_executable)
                player2 = Player(self.ab_executable)
                process_pair = (player1, player2, i)
                self.available_processes.put(process_pair)
            except Exception as e:
                print(f"Failed to create process pair {i}: {e}")
                
        self.is_initialized = True
        
    def get_process_pair(self, timeout: float = 30.0) -> Optional[Tuple]:
        """프로세스 쌍 대여"""
        try:
            return self.available_processes.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def return_process_pair(self, process_pair: Tuple):
        """프로세스 쌍 반납"""
        self.available_processes.put(process_pair)
        
    def cleanup(self):
        """프로세스 풀 정리"""
        while not self.available_processes.empty():
            try:
                player1, player2, pair_id = self.available_processes.get_nowait()
                
                # 프로세스 종료
                for player in [player1, player2]:
                    try:
                        player.print("FINISH")
                        player.process.terminate()
                        player.process.wait(timeout=5)
                    except:
                        try:
                            player.process.kill()
                        except:
                            pass
            except queue.Empty:
                break
                
        self.is_initialized = False


class ThreadSafeGameCollector:
    """스레드 안전한 게임 데이터 수집기"""
    
    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size
        self.games = []
        self.lock = Lock()
        self.total_collected = 0
        
    def add_game(self, game_data) -> bool:
        """게임 데이터 추가, 배치가 찼으면 True 반환"""
        with self.lock:
            self.games.append(game_data)
            self.total_collected += 1
            
            if len(self.games) >= self.batch_size:
                return True
        return False
        
    def get_batch(self) -> List:
        """현재 배치 가져오기 및 초기화"""
        with self.lock:
            batch = self.games.copy()
            self.games.clear()
            return batch
            
    def get_remaining(self) -> List:
        """남은 게임들 가져오기"""
        with self.lock:
            remaining = self.games.copy()
            self.games.clear()
            return remaining


class ABSelfPlayGenerator:
    """AB Pruning 자기대국 데이터 생성기 (내장 게임 실행)"""
    
    def __init__(self, 
                 ab_executable: str = "practice/testing/ab_pruning.exe",
                 ab_move_ratio: float = 0.85,
                 noise_ratio: float = 0.15,
                 num_workers: int = None):
        """
        Args:
            ab_executable: AB pruning 실행 파일 경로
            ab_move_ratio: AB가 선택한 움직임의 확률
            noise_ratio: 다른 유효 움직임들에 분배할 확률
            num_workers: 워커 스레드 수 (None이면 CPU 코어 수 기반)
        """
        self.ab_executable = Path(ab_executable)
        self.converter = ABLogConverter(ab_move_ratio, noise_ratio)
        self.R = 10
        self.C = 17
        self.num_workers = num_workers or min(6, multiprocessing.cpu_count() - 1)
        
        # 파일 존재 확인
        if not self.ab_executable.exists():
            raise FileNotFoundError(f"AB executable not found: {self.ab_executable}")
    
    def generate_single_game(self, initial_board: List[List[int]], 
                           timeout: int = 300, verbose: bool = False) -> Optional[SelfPlayData]:
        """단일 게임 생성 (내장 실행 로직 사용)"""
        
        try:
            # AB vs AB 게임 실행
            game_log = self._run_ab_vs_ab_game(initial_board, timeout, verbose)
            
            if game_log:
                # 로그를 SelfPlayData로 변환
                return self.converter.convert_log_to_selfplay_data(game_log, initial_board)
            else:
                if verbose:
                    print("Game execution failed")
                return None
                
        except Exception as e:
            if verbose:
                print(f"Error in game generation: {e}")
            return None
    
    def generate_games(self, initial_boards: List[List[List[int]]], 
                      timeout_per_game: int = 300, 
                      verbose: bool = False,
                      batch_size: int = 50) -> List[SelfPlayData]:
        """여러 게임 생성 (배치 단위 멀티스레드)"""
        if not initial_boards:
            return []
        
        if verbose:
            print(f"배치 단위 멀티스레드 게임 생성 시작 (워커: {self.num_workers}개, 배치크기: {batch_size})")
        
        all_games_data = []
        total_boards = len(initial_boards)
        
        # 배치 단위로 처리
        for batch_start in range(0, total_boards, batch_size):
            batch_end = min(batch_start + batch_size, total_boards)
            batch_boards = initial_boards[batch_start:batch_end]
            
            if verbose:
                print(f"  배치 {batch_start//batch_size + 1}: {batch_start+1}-{batch_end}/{total_boards} 게임 처리중...")
            
            batch_games = self._generate_batch_games(batch_boards, timeout_per_game, verbose, batch_start)
            all_games_data.extend(batch_games)
            
            # 배치 완료 후 가비지 컬렉션
            import gc
            gc.collect()
        
        return all_games_data
    
    def _generate_batch_games(self, batch_boards: List[List[List[int]]], 
                             timeout_per_game: int,
                             verbose: bool,
                             batch_offset: int = 0) -> List[SelfPlayData]:
        """단일 배치 게임 생성"""
        games_data = []
        
        # 배치마다 새로운 ThreadPoolExecutor 사용
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            try:
                # 배치 내 모든 게임을 병렬로 제출
                future_to_board = {
                    executor.submit(self._generate_single_game_threaded, board, timeout_per_game): (i, board)
                    for i, board in enumerate(batch_boards)
                }
                
                # 완료된 게임들 수집
                completed = 0
                for future in as_completed(future_to_board):
                    board_idx, board = future_to_board[future]
                    completed += 1
                    
                    try:
                        game_data = future.result()
                        if game_data:
                            games_data.append(game_data)
                            if verbose:
                                print(f"    Game {batch_offset + board_idx+1} completed: "
                                      f"{game_data.game_length} moves, "
                                      f"score {game_data.final_score[0]}-{game_data.final_score[1]} "
                                      f"({completed}/{len(batch_boards)})")
                        else:
                            if verbose:
                                print(f"    Game {batch_offset + board_idx+1} failed ({completed}/{len(batch_boards)})")
                    except Exception as e:
                        if verbose:
                            print(f"    Game {batch_offset + board_idx+1} error: {e} ({completed}/{len(batch_boards)})")
                
                # Future 객체들 명시적 정리
                del future_to_board
                        
            except Exception as e:
                if verbose:
                    print(f"    배치 처리 중 오류: {e}")
        
        return games_data
    
    def _generate_single_game_threaded(self, initial_board: List[List[int]], 
                                     timeout: int = 300) -> Optional[SelfPlayData]:
        """스레드에서 실행되는 단일 게임 생성 (새 프로세스 사용)"""
        # 각 게임마다 새로운 프로세스 생성하여 상태 오염 방지
        user1 = None
        user2 = None
        try:
            user1 = Player(str(self.ab_executable))
            user2 = Player(str(self.ab_executable))
            
            game_log = self._run_ab_vs_ab_game_with_processes(
                initial_board, [user1, user2], timeout, False
            )
            
            if game_log:
                game_data = self.converter.convert_log_to_selfplay_data(game_log, initial_board)
                return game_data
            return None
            
        except Exception as e:
            return None
            
        finally:
            # Player 리소스 완전 정리
            for user in [user1, user2]:
                if user:
                    try:
                        user.print("FINISH")
                    except:
                        pass
                    user.cleanup()
            
            # 명시적 메모리 해제
            del user1, user2
    
    def _run_ab_vs_ab_game_with_processes(self, board: List[List[int]], 
                                        user: List[Player], timeout: int, verbose: bool) -> Optional[List[str]]:
        """기존 프로세스를 사용한 AB vs AB 게임 실행"""
        game_log = []
        
        try:
            # 보드 복사 (원본 보드 보호)
            board = [row[:] for row in board]
            
            # READY 명령
            user[0].print("READY FIRST")
            user[1].print("READY SECOND")
            lines = Player.readAll(user, 3.0)
            
            # 초기화 확인
            for i, line in enumerate(lines):
                if line is None or line[1] != "OK\n":
                    if verbose:
                        print(f'ABORT {i} TLE during READY')
                    return None
            
            # 보드 초기화
            board_str = ' '.join(''.join(map(str, row)) for row in board)
            for u in user:
                u.print(f'INIT {board_str}')
            game_log.append(f'INIT {board_str}')
            
            # 게임 진행
            timeout_limits = [10000, 10000]  # 각 플레이어 타임아웃
            passed = False
            
            for i in range(999):  # 최대 999턴
                u = i % 2
                name = ['FIRST', 'SECOND'][u]
                
                # TIME 명령 전송
                user[u].print(f'TIME {timeout_limits[u]} {timeout_limits[1 - u]}')
                read = user[u].readline(timeout_limits[u])
                
                if read is None:
                    return None
                
                readTime, readStr = read
                readTime = min(int(readTime*1000), timeout_limits[u])
                timeout_limits[u] -= readTime
                
                # 움직임 파싱
                try:
                    r1, c1, r2, c2 = map(int, readStr.split())
                except:
                    return None
                
                # 패스 처리
                if r1 == -1 and c1 == -1 and r2 == -1 and c2 == -1:
                    user[1-u].print(f'OPP {r1} {c1} {r2} {c2} {readTime}')
                    game_log.append(f'{name} {r1} {c1} {r2} {c2} {readTime}')
                    if passed:
                        break  # 연속 패스로 게임 종료
                    passed = True
                else:
                    passed = False
                    
                    # 유효성 검사 (간소화)
                    if not (0 <= r1 <= r2 < self.R and 0 <= c1 <= c2 < self.C):
                        return None
                    
                    # 합 검사
                    sum_val = 0
                    for r in range(r1, r2+1):
                        for c in range(c1, c2+1):
                            if board[r][c] > 0:
                                sum_val += board[r][c]
                    
                    if sum_val != 10:
                        return None
                    
                    # 모서리 검사
                    top, down, left, right = False, False, False, False
                    for r in range(r1, r2+1):
                        if board[r][c1] > 0:
                            left = True
                        if board[r][c2] > 0:
                            right = True
                    for c in range(c1, c2+1):
                        if board[r1][c] > 0:
                            top = True
                        if board[r2][c] > 0:
                            down = True
                    
                    if not (left and right and top and down):
                        return None
                    
                    # 보드 업데이트
                    for r in range(r1, r2+1):
                        for c in range(c1, c2+1):
                            board[r][c] = -u-1
                    
                    # 상대방에게 움직임 전달
                    user[1-u].print(f'OPP {r1} {c1} {r2} {c2} {int(readTime)}')
                    game_log.append(f'{name} {r1} {c1} {r2} {c2} {readTime}')
            
            # 점수 계산
            score = [0, 0]
            for row in board:
                for num in row:
                    if num == -1:
                        score[0] += 1
                    elif num == -2:
                        score[1] += 1
            
            # 게임 종료 로그
            game_log.append('FINISH')
            game_log.append(f'SCOREFIRST {score[0]}')
            game_log.append(f'SCORESECOND {score[1]}')
            
            return game_log
            
        except Exception as e:
            return None
    
    def _run_ab_vs_ab_game(self, board: List[List[int]], timeout: int, verbose: bool) -> Optional[List[str]]:
        """AB vs AB 게임 실행 (testing_tool 로직 내장)"""
        
        # AB 프로세스 두 개 시작
        try:
            user = [Player(str(self.ab_executable)), Player(str(self.ab_executable))]
        except Exception as e:
            if verbose:
                print(f"Failed to start AB processes: {e}")
            return None
        
        game_log = []
        
        try:
            # READY 명령
            user[0].print("READY FIRST")
            user[1].print("READY SECOND")
            lines = Player.readAll(user, 3.0)
            
            # 초기화 확인
            aborted = False
            for i, line in enumerate(lines):
                if line is None or line[1] != "OK\n":
                    if verbose:
                        print(f'ABORT {i} TLE during READY')
                    game_log.append(f'ABORT {i} TLE')
                    aborted = True
            
            if aborted:
                return None
            
            # 보드 초기화
            board_str = ' '.join(''.join(map(str, row)) for row in board)
            for u in user:
                u.print(f'INIT {board_str}')
            game_log.append(f'INIT {board_str}')
            
            # 게임 진행
            timeout_limits = [10000, 10000]  # 각 플레이어 타임아웃
            passed = False
            
            for i in range(999):  # 최대 999턴
                u = i % 2
                name = ['FIRST', 'SECOND'][u]
                
                # TIME 명령 전송
                user[u].print(f'TIME {timeout_limits[u]} {timeout_limits[1 - u]}')
                read = user[u].readline(timeout_limits[u])
                
                if read is None:
                    if verbose:
                        print(f'ABORT {u} TLE during move')
                    game_log.append(f'ABORT {u} TLE')
                    return None
                
                readTime, readStr = read
                readTime = min(int(readTime*1000), timeout_limits[u])
                timeout_limits[u] -= readTime
                
                # 움직임 파싱
                try:
                    r1, c1, r2, c2 = map(int, readStr.split())
                except:
                    if verbose:
                        print(f'ABORT {u} Parse failed')
                    game_log.append(f'ABORT {u} Parse failed')
                    return None
                
                # 패스 처리
                if r1 == -1 and c1 == -1 and r2 == -1 and c2 == -1:
                    user[1-u].print(f'OPP {r1} {c1} {r2} {c2} {readTime}')
                    game_log.append(f'{name} {r1} {c1} {r2} {c2} {readTime}')
                    if passed:
                        break  # 연속 패스로 게임 종료
                    passed = True
                else:
                    passed = False
                    
                    # 유효성 검사
                    if not (0 <= r1 <= r2 < self.R and 0 <= c1 <= c2 < self.C):
                        if verbose:
                            print(f'ABORT {u} Out of range')
                        game_log.append(f'ABORT {u} Out of range')
                        return None
                    
                    # 합 검사
                    sum_val = 0
                    for r in range(r1, r2+1):
                        for c in range(c1, c2+1):
                            if board[r][c] > 0:
                                sum_val += board[r][c]
                    
                    if sum_val != 10:
                        if verbose:
                            print(f'ABORT {u} Sum not equals to 10')
                        game_log.append(f'ABORT {u} Sum not equals to 10')
                        return None
                    
                    # 모서리 검사
                    top, down, left, right = False, False, False, False
                    for r in range(r1, r2+1):
                        if board[r][c1] > 0:
                            left = True
                        if board[r][c2] > 0:
                            right = True
                    for c in range(c1, c2+1):
                        if board[r1][c] > 0:
                            top = True
                        if board[r2][c] > 0:
                            down = True
                    
                    if not (left and right and top and down):
                        if verbose:
                            print(f'ABORT {u} Not fit')
                        game_log.append(f'ABORT {u} Not fit')
                        return None
                    
                    # 보드 업데이트
                    for r in range(r1, r2+1):
                        for c in range(c1, c2+1):
                            board[r][c] = -u-1
                    
                    # 상대방에게 움직임 전달
                    user[1-u].print(f'OPP {r1} {c1} {r2} {c2} {int(readTime)}')
                    game_log.append(f'{name} {r1} {c1} {r2} {c2} {readTime}')
            
            # 점수 계산
            score = [0, 0]
            for row in board:
                for num in row:
                    if num == -1:
                        score[0] += 1
                    elif num == -2:
                        score[1] += 1
            
            # 게임 종료 로그
            game_log.append('FINISH')
            game_log.append(f'SCOREFIRST {score[0]}')
            game_log.append(f'SCORESECOND {score[1]}')
            
            # 프로세스 정리
            for u in user:
                u.print("FINISH")
            
            return game_log
            
        except Exception as e:
            if verbose:
                print(f"Error during game execution: {e}")
            return None
    
    def cleanup(self):
        """리소스 정리 (각 게임마다 프로세스를 정리하므로 추가 정리 불필요)"""
        pass
            
    def __del__(self):
        """소멸자"""
        pass


def test_ab_selfplay():
    """테스트 함수"""
    # 샘플 보드 생성
    sample_board = [
        [6, 1, 6, 8, 2, 9, 7, 8, 5, 2, 5, 9, 4, 3, 1, 9, 6],
        [3, 4, 4, 4, 5, 1, 6, 3, 8, 7, 6, 3, 9, 8, 2, 9, 6],
        [6, 9, 2, 3, 2, 7, 9, 9, 7, 9, 1, 4, 1, 7, 2, 4, 4],
        [3, 3, 1, 1, 7, 4, 2, 7, 7, 7, 3, 9, 4, 4, 6, 2, 7],
        [4, 5, 1, 3, 3, 6, 3, 3, 2, 9, 8, 6, 9, 7, 4, 4, 9],
        [8, 8, 1, 5, 6, 7, 5, 6, 5, 5, 3, 8, 2, 2, 8, 7, 5],
        [4, 7, 2, 4, 6, 4, 8, 5, 6, 9, 3, 9, 9, 9, 8, 8, 2],
        [7, 5, 7, 3, 4, 6, 2, 4, 8, 2, 1, 2, 6, 7, 9, 9, 7],
        [4, 5, 4, 1, 1, 6, 6, 7, 9, 3, 8, 5, 5, 1, 7, 1, 6],
        [5, 8, 2, 6, 9, 7, 5, 2, 8, 7, 8, 6, 6, 3, 3, 2, 2]
    ]
    
    try:
        generator = ABSelfPlayGenerator()
        print("AB Self-play generator created successfully")
        
        # 단일 게임 테스트
        print("Testing single game generation...")
        game_data = generator.generate_single_game(sample_board, verbose=True)
        
        if game_data:
            print("Single game generation successful!")
            print(f"  Game length: {game_data.game_length}")
            print(f"  Final score: {game_data.final_score}")
            print(f"  Winner: {game_data.winner}")
        else:
            print("Single game generation failed")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_ab_selfplay()
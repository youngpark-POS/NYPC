# testing-tool-yacht

testing-tool-yacht는 Yacht Dice 문제를 테스팅 하기 위해서 만들어진 도구입니다.

## 실행 방법

`testing-tool-yacht.py`는 다음과 같은 커맨드라인 인자를 받을 수 있습니다.

- `-h`, `--help`: 도움말을 출력합니다.
- `-c CONFIG`, `--config CONFIG`: **설정 파일**로 `CONFIG`를 사용합니다.
- `-i INPUT`, `--input INPUT`: **입력 파일**로 `INPUT`을 사용합니다.
- `-l LOG`, `--log LOG`: **로그 파일**로 `LOG`를 사용합니다.
- `-s`, `--stdio`: **입력 파일**이나 **로그 파일**이 주어지지 않은 경우, 표준 입출력을 대신 사용합니다.
- `-a EXEC1, --exec1 EXEC1`: 선공 플레이어의 실행 커맨드로 `EXEC1`을 사용합니다.
- `-b EXEC2, --exec2 EXEC2`: 후공 플레이어의 실행 커맨드로 `EXEC2`를 사용합니다.

예를 들어 입력 파일을 `input.txt`, 로그 파일을 `log.txt`, 선공 플레이어의 실행 커맨드를 `python3 sample-code.py P1`, 후공 플레이어의 실행 커맨드를 `python3 sample-code.py P2`와 같이 사용하고 싶으면 다음과 같이 실행합니다.

```bash
python3 testing-tool-yacht.py -i input.txt -l log.txt -a "python3 sample-code.py P1" -b "python3 sample-code.py P2"
```

### 설정 파일

설정 파일은 command-line argument를 간단하게 사용하기 위한 방법으로, 다음과 같은 내용을 작성할 수 있습니다.

```
INPUT=<입력 파일 경로>
LOG=<로그 파일 경로>
EXEC1=<선공 플레이어의 프로그램 실행 커맨드>
EXEC2=<후공 플레이어의 프로그램 실행 커맨드>
```

단, 커맨드라인 인자와 내용이 충돌하는 경우 커맨드라인 인자가 우선 실행됩니다.

예를 들어 입력 파일을 `input.txt`, 로그 파일을 `log.txt`, 선공 플레이어의 실행 커맨드를 `python3 sample-code.py P1`, 후공 플레이어의 실행 커맨드를 `python3 sample-code.py P2`와 같이 사용하고 싶으면 `config.ini`를 다음과 같이 작성합니다.

```
INPUT=input.txt
LOG=log.txt
EXEC1=python3 sample-code.py P1
EXEC2=python3 sample-code.py P2
```

그 이후 다음 명령어를 사용합니다.

```bash
python3 testing-tool-yacht.py -c config.ini
```

### 입력 파일

테스팅 툴의 입력 파일은 1라운드부터 12라운드까지의 주사위 정보를 나타냅니다.
파일의 각 줄은 5개의 주사위 A, 5개의 주사위 B, 그리고 타이브레이크 값(0 또는 1)으로 구성되어야 합니다.
같은 값을 입찰했을 때, 0이면 선공, 1이면 후공에게 원하는 주사위가 주어집니다.

예를 들어 줄 `12345 54321 0`은 주사위 A가 [1, 2, 3, 4, 5], 주사위 B가 [5, 4, 3, 2, 1]이며, 같은 값을 입찰했을 때는 선공에게 원하는 주사위가 주어진다는 의미입니다.

### 로그 파일

로그 파일에는 게임에 대한 다음 정보를 출력합니다.

- `[<FIRST/SECOND> "<player>"]`
  - 1P혹은 2P가 어떤 커맨드를 실행했는지를 나타냅니다.
- `[RESULT "<result>"]`
  - 게임의 결과를 나타냅니다. `1-0`은 선공 승, `1/2-1/2`는 무승부, `0-1`은 후공 승을 나타냅니다.
- `ROUND <n>`
  - `n`번째 라운드를 시작함을 의미합니다.
- `ROLL <diceA> <diceB>`
  - 주사위 A와 B가 각각 `<diceA>`, `<diceB>`로 굴려졌음을 의미합니다.
- `BID FIRST/SECOND <A/B> <amount>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 그룹 `<A/B>`에 `<amount>`만큼 입찰했음을 의미합니다.
- `GET FIRST/SECOND <A/B>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 그룹 `<A/B>`의 주사위를 받았음을 의미합니다.
- `PUT FIRST/SECOND <RULE> <dice>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 `<RULE>` 규칙에 `<dice>` 주사위를 배치했음을 의미합니다.
- `FINISH`
  - 게임의 정상적인 종료를 나타냅니다.
- `SCORE<FIRST/SECOND> <score>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 받은 점수가 `<score>`임을 나타냅니다.
- `ABORT <FIRST/SECOND> <reason>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 `<reason>`을 이유로 프로그램이 비정상종료했음을 의미합니다.
- `# Debug <FIRST/SECOND>: <msg>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 표준 에러 출력(stderr)에 출력한 줄입니다.

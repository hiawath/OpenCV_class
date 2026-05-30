import sys
import os
import platform
import getpass

def show_help():
    print("\n--- [공정 관리 시스템 도움말] ---")
    for cmd, desc in commands.items():
        print(f"{cmd:<12} : {desc['info']}")

def show_sysinfo():
    print(f"파이썬 버전: {sys.version}")
    print(f"운영체제(OS): {platform.platform()}")
    print(f"현재 작업 디렉토리: {os.getcwd()}")

# 기능 정의 예시
commands = {
    "help": {"func": show_help, "info": "사용 가능한 모든 명령어를 보여줍니다."},
    "clear": {"func": lambda: print("\033[H\033[J"), "info": "터미널 화면을 깨끗이 지웁니다."},
    # 여기에 아래 10가지 기능을 추가합니다.
    "sysinfo": {"func": show_sysinfo, "info": "현재 파이썬 버전, 운영체제(OS), 현재 작업 디렉토리 경로를 보여줍니다."},
}

def main_cli():
    username = getpass.getuser()
    hostname = platform.node()
    while True:
        user_input = input(f"\n[{username}@{hostname}] >> ").strip().lower()
        if user_input in commands:
            commands[user_input]["func"]()
        elif user_input == "exit":
            break
        else:
            print(f"'{user_input}'은(는) 알 수 없는 명령어입니다. 'help'를 입력하세요.")

if __name__ == "__main__":
    main_cli()
#
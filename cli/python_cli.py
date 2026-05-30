import sys
import os
import platform
import getpass


# 기능 정의 예시
commands = {}
# 자동 등록 데코레이터(Decorator) 생성
def register_command(cmd_name, description):
    """
    함수 위에 @register_command("명령어", "설명") 형태로 붙이면
    자동으로 commands 딕셔너리에 등록해주는 마법의 함수입니다.
    """
    def decorator(func):
        # 딕셔너리에 명령어 이름, 실행할 함수, 설명을 저장
        commands[cmd_name] = {
            "func": func,
            "info": description
        }
        return func # 원래 함수는 그대로 돌려줌
    return decorator

@register_command("hello", "인사말을 출력합니다. (테스트용)")
def say_hello():
    print("안녕하세요! 시스템이 정상 작동 중입니다.")

@register_command("help", "사용 가능한 모든 명령어를 보여줍니다.")
def show_help():
    print("\n--- [공정 관리 시스템 도움말] ---")
    for cmd, desc in commands.items():
        print(f"{cmd:<12} : {desc['info']}")

@register_command("sysinfo", "시스템 버전을 확인합니다.")
def show_sysinfo():
    print(f"파이썬 버전: {sys.version}")
    print(f"운영체제(OS): {platform.platform()}")
    print(f"현재 작업 디렉토리: {os.getcwd()}")

@register_command("clear", "터미널 화면을 깨끗이 지웁니다.")
def clear_screen():
    # 윈도우는 'cls', 맥/리눅스는 'clear'
    os.system('cls' if os.name == 'nt' else 'clear')

def main_cli():
    username = getpass.getuser()
    hostname = platform.node()
    while True:
        try:
            user_input = input(f"\n[{username}@{hostname}] >> ").strip().lower()
            if user_input in commands:
                commands[user_input]["func"]()
            elif user_input == "exit":
                break
            else:
                print(f"'{user_input}'은(는) 알 수 없는 명령어입니다. 'help'를 입력하세요.")
        except KeyboardInterrupt:
            # Ctrl+C 를 눌렀을 때 튕기지 않고 안전하게 종료
            print("\n프로그램을 강제 종료합니다.")
            break
        except Exception as e:
            print(f"명령어 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main_cli()
#
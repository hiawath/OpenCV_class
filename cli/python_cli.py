import sys
import os
import platform
import getpass
import shutil
from datetime import datetime
import fnmatch


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

@register_command("ls", "현재 폴더에 있는 파일과 디렉토리 목록을 보기 좋게 출력합니다.")
def list_directory():
    current_path = os.getcwd()
    print(f"\n[{current_path}] 디렉토리 내용:")
    try:
        for item in sorted(os.listdir(current_path)):
            full_path = os.path.join(current_path, item)
            if os.path.isdir(full_path):
                print(f"  [DIR]  {item}")
            else:
                print(f"  [FILE] {item}")
    except Exception as e:
        print(f"목록을 불러오는 중 오류가 발생했습니다: {e}")

@register_command("calc", "사용자가 입력한 수식(예: 10 * 5 / 2)을 즉석에서 계산해 결과를 보여줍니다.")
def calculate_expression():
    expr = input("계산할 수식을 입력하세요 (예: 10 * 5 / 2) >> ").strip()
    try:
        result = eval(expr)
        print(f"결과: {result}")
    except ZeroDivisionError:
        print("오류: 0으로 나눌 수 없습니다.")
    except Exception as e:
        print(f"수식 오류: 유효한 수식이 아닙니다. ({e})")

@register_command("logview", "log.txt 파일의 마지막 10줄을 읽어서 실시간 공정 상태를 모니터링합니다.")
def view_logs():
    log_file = "./cli/log.txt"
    if not os.path.exists(log_file):
        print(f"[{log_file}] 파일이 존재하지 않습니다.")
        return
        
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            last_lines = lines[-10:] # 리스트 슬라이싱으로 마지막 10줄 가져오기
            print(f"\n--- [최근 로그 {len(last_lines)}줄] ---")
            for line in last_lines:
                print(line.strip())
    except Exception as e:
        print(f"로그 파일을 읽는 중 오류가 발생했습니다: {e}")

@register_command("backup", "지정된 데이터 폴더를 backup_날짜 형태의 폴더로 복사합니다.")
def backup_data():
    target_dir = input("백업할 원본 폴더 경로를 입력하세요 (예: ./temp) >> ").strip()
    if not os.path.exists(target_dir):
        print(f"오류: '{target_dir}' 경로가 존재하지 않습니다.")
        return
    if not os.path.isdir(target_dir):
        print(f"오류: '{target_dir}'은(는) 폴더가 아닙니다.")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    try:
        shutil.copytree(target_dir, backup_dir)
        print(f"백업 성공: '{target_dir}' 폴더가 '{backup_dir}'(으)로 복사되었습니다.")
    except Exception as e:
        print(f"백업 중 오류가 발생했습니다: {e}")

@register_command("find", "지정된 디렉토리(하위 포함)에서 패턴(예: *.py)에 맞는 파일을 검색합니다.")
def find_files():
    target_dir = input("검색할 디렉토리 경로를 입력하세요 (엔터 입력 시 현재 폴더) >> ").strip()
    if not target_dir:
        target_dir = os.getcwd()
        
    if not os.path.isdir(target_dir):
        print(f"오류: '{target_dir}'은(는) 유효한 디렉토리가 아닙니다.")
        return
        
    pattern = input("검색할 파일명 패턴을 입력하세요 (예: *.py, *test*) >> ").strip()
    if not pattern:
        print("패턴을 입력해주세요.")
        return
        
    # 사용자가 와일드카드를 쓰지 않은 경우 키워드 포함 검색(*키워드*)으로 자동 전환
    if '*' not in pattern and '?' not in pattern:
        pattern = f"*{pattern}*"
        
    print(f"\n[{target_dir}] 및 하위 폴더에서 '{pattern}' 검색 결과:")
    try:
        found_count = 0
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if fnmatch.fnmatch(file, pattern):
                    print(f"  - {os.path.join(root, file)}")
                    found_count += 1
        if found_count == 0:
            print("  일치하는 파일이 없습니다.")
        else:
            print(f"\n  총 {found_count}개의 파일을 찾았습니다.")
    except Exception as e:
        print(f"검색 중 오류가 발생했습니다: {e}")

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
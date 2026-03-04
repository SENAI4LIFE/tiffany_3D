import subprocess
import sys

def run_command(command, allow_fail=False):
    try:
        subprocess.run(command, shell=True, check=not allow_fail)
    except subprocess.CalledProcessError:
        sys.exit(1)

run_command("git branch -D latest_branch", allow_fail=True)
run_command("git checkout --orphan latest_branch")
run_command("git add -A")
run_command("git commit -m 'Initial commit'")
run_command("git branch -D main", allow_fail=True)
run_command("git branch -m main")

print("\nReady to push. This will overwrite the history on GitHub.")
confirm = input("Type 'yes' to force push: ")

if confirm.lower() == 'yes':
    run_command("git push -f origin main")
else:
    print("Push cancelled.")

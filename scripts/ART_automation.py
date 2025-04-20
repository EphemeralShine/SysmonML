import subprocess
import random
import time

# Full PowerShell module and Atomic test path
ART_MODULE_PATH = r"C:\AtomicRedTeam\invoke-atomicredteam\Invoke-AtomicRedTeam.psd1"

# Choose Word or Excel randomly
def choose_office_product():
    return random.choice(["Word", "Excel"])

# Run the atomic test
def run_atomic_test(product):
    print(f"[*] Running T1053.005-5 with: {product}")
    ps_command = rf'''
    Import-Module "{ART_MODULE_PATH}"
    Invoke-AtomicTest T1053.005 -TestNumbers 5 -InputArgs @{{ ms_product = "{product}" }} -Confirm:$false
    '''
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_command],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)
    return result.returncode

# Run cleanup
def run_cleanup():
    print("[*] Cleaning up test artifacts...")
    ps_cleanup = rf'''
    Import-Module "{ART_MODULE_PATH}"
    Invoke-AtomicTest T1053.005 -TestNumbers 5 -Cleanup
    '''
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cleanup],
        capture_output=True,
        text=True
    )

    # Force kill notepad.exe if still running
    print("[*] Checking for lingering notepad.exe processes...")
    try:
        subprocess.run(["taskkill", "/IM", "notepad.exe", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[+] notepad.exe terminated (if running).")
    except Exception as e:
        print(f"[-] Failed to kill notepad.exe: {e}")
    print("[+] Cleanup complete.")

# Main loop
if __name__ == "__main__":
    try:
        while True:
            office_product = choose_office_product()
            ret_code = run_atomic_test(office_product)

            if ret_code == 0:
                run_cleanup()
            else:
                print(f"[-] Test failed with return code: {ret_code}. Skipping cleanup.")

            # Sleep between 2 to 5 minutes
            sleep_seconds = random.randint(120, 300)
            print(f"[*] Sleeping for {sleep_seconds} seconds...\n")
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("\n[!] Script terminated by user.")
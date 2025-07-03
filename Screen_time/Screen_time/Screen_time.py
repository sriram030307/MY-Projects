import time
import win32gui
def get_active_window():
    window = win32gui.GetForegroundWindow()
    return win32gui.GetWindowText(window)
start_time = time.time()
active_window = get_active_window()
screen_time = 0
try:
    while True:
        current_window = get_active_window()
        if current_window:
            screen_time = time.time() - start_time
        time.sleep(1) 
        print(f"Screen Time: {int(screen_time)} seconds", end="\r")
except KeyboardInterrupt:
    print(f"\nTotal Screen Time: {int(screen_time)} seconds")

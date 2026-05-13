import platform
import subprocess
try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None 
def get_brightness():
    system_os = platform.system()

    if system_os == "Windows" or system_os == "Linux": 
        if sbc:
            try:
                brightness = sbc.get_brightness(display=0)  
                return f"Screen Brightness: {brightness}%"
            except Exception as e:
                return f"Error getting brightness: {str(e)}"
    elif system_os == "Android":
        try:
            result = subprocess.run(["adb", "shell", "settings get system screen_brightness"],
                                    capture_output=True, text=True)
            brightness = int(result.stdout.strip())
            return f"Screen Brightness (Android): {brightness}/255 ({(brightness/255)*100:.2f}%)"
        except Exception as e:
            return f"Error getting Android brightness: {str(e)}"
    return "Unsupported OS"
if __name__ == "__main__":
    print(get_brightness())


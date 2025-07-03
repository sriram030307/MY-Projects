screen_time = float(input("Enter screen time in hours: "))  
distance_from_screen = float(input("Enter distance from screen in cm: "))  
screen_brightness = int(input("Enter screen brightness (0-100%): "))  
warnings = []
if screen_time > 2:
    warnings.append("⚠️ Warning: You have been using the screen for more than 2 hours. Take a break! ⏳")
elif distance_from_screen < 30:
    warnings.append("⚠️ Warning: You are too close to the screen! Maintain at least 30 cm distance. 👀")
elif screen_brightness == 100:
    warnings.append("⚠️ Warning: Your screen brightness is at 100%! Reduce it to prevent eye strain. 🔆")
if warnings:
    for warning in warnings:
        print(warning)
else:
    print("✅ No warning. Your screen habits are safe! 👍")

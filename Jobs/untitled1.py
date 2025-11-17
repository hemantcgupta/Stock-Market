import pyautogui
import time
import webbrowser
import os

# Step 1: Open Chrome and navigate to the first URL
webbrowser.open('https://www.google.com')  # Opens Chrome or the default browser
time.sleep(5)

# Enter the first URL
pyautogui.hotkey('ctrl', 'l')  # Focus on the address bar (Ctrl+L)
time.sleep(1)
pyautogui.typewrite('https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY', interval=0.05)
pyautogui.press('enter')  # Go to the URL

# Wait for the page to load
time.sleep(10)


# Step 2: Select all content on the page and copy it
pyautogui.hotkey('ctrl', 'a')  # Select all (Ctrl+A)
time.sleep(1)
pyautogui.hotkey('ctrl', 'c')  # Copy selected content (Ctrl+C)
time.sleep(2)

# Step 3: Open Notepad
pyautogui.hotkey('win', 'r')  # Open Run dialog (Win+R)
time.sleep(1)
pyautogui.typewrite('notepad', interval=0.05)  # Type 'notepad'
pyautogui.press('enter')  # Open Notepad
time.sleep(2)

# Step 4: Paste the copied content into Notepad
pyautogui.hotkey('ctrl', 'v')  # Paste (Ctrl+V)
time.sleep(1)

# Step 5: Save the file to the desktop
pyautogui.hotkey('ctrl', 's')  # Open Save dialog (Ctrl+S)
time.sleep(1)

# Path to save the file on the desktop (adjust as needed)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "data.json")
pyautogui.typewrite(desktop_path, interval=0.05)
pyautogui.press('enter')  # Save the file
time.sleep(1)

# Optional: Close Notepad (Ctrl+W) and confirm if prompted (Alt+S)
pyautogui.hotkey('ctrl', 'w')  # Close Notepad
time.sleep(1)
pyautogui.press('alt', 's')  # Confirm saving if prompted





# Open a new tab and enter the second URL
pyautogui.hotkey('ctrl', 't')  # Open a new tab (Ctrl+T)
time.sleep(1)
pyautogui.typewrite('https://www.nseindia.com/api/chart-databyindex?index=OPTIDXNIFTY19-09-2024PE25400.00', interval=0.05)
pyautogui.press('enter')  # Go to the URL

# Wait for the page to load
time.sleep(10)

# Step 2: Select all content on the page and copy it
pyautogui.hotkey('ctrl', 'a')  # Select all (Ctrl+A)
time.sleep(1)
pyautogui.hotkey('ctrl', 'c')  # Copy selected content (Ctrl+C)
time.sleep(2)

# Step 3: Open Notepad
pyautogui.hotkey('win', 'r')  # Open Run dialog (Win+R)
time.sleep(1)
pyautogui.typewrite('notepad', interval=0.05)  # Type 'notepad'
pyautogui.press('enter')  # Open Notepad
time.sleep(2)

# Step 4: Paste the copied content into Notepad
pyautogui.hotkey('ctrl', 'v')  # Paste (Ctrl+V)
time.sleep(1)

# Step 5: Save the file to the desktop
pyautogui.hotkey('ctrl', 's')  # Open Save dialog (Ctrl+S)
time.sleep(1)

# Path to save the file on the desktop (adjust as needed)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "data.json")
pyautogui.typewrite(desktop_path, interval=0.05)
pyautogui.press('enter')  # Save the file
time.sleep(1)

# Optional: Close Notepad (Ctrl+W) and confirm if prompted (Alt+S)
pyautogui.hotkey('ctrl', 'w')  # Close Notepad
time.sleep(1)
pyautogui.press('alt', 's')  # Confirm saving if prompted

print("Data has been saved to the desktop.")

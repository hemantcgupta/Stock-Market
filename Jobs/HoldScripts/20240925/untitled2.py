import pyautogui
import time
import webbrowser

# Open Chrome (this will vary depending on your OS and Chrome installation)
# For Windows, you might need to adjust the path or use a different method
webbrowser.open('https://www.google.com')  # Opens Chrome or the default browser

# Wait for Chrome to open
time.sleep(5)

# Enter the first URL
pyautogui.hotkey('ctrl', 'l')  # Focus on the address bar (Ctrl+L)
time.sleep(1)
pyautogui.typewrite('https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY', interval=0.05)
pyautogui.press('enter')  # Go to the URL

# Wait for the page to load
time.sleep(10)

# Open a new tab
pyautogui.hotkey('ctrl', 't')  # Open a new tab (Ctrl+T)
time.sleep(1)

# Enter the second URL
pyautogui.typewrite('https://www.nseindia.com/api/chart-databyindex?index=OPTIDXNIFTY19-09-2024PE25400.00', interval=0.05)
pyautogui.press('enter')  # Go to the URL

# Wait for the page to load
time.sleep(10)

# Optionally, close the browser after some time
# time.sleep(10)
# pyautogui.hotkey('ctrl', 'w')  # Close the current tab (Ctrl+W)
# pyautogui.hotkey('ctrl', 'shift', 'w')  # Close the browser (Ctrl+Shift+W) - This might vary

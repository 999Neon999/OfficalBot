import time
import pyotp
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from kiteconnect import KiteConnect

# ================== CREDENTIALS & API CONFIG ==================
USER_ID = "THS720"
PASSWORD = "Chits@23"
TOTP_SECRET = "E5HRMPZEN4QOOORKIPR5UX66E5SAG4VY"

API_KEY = "vv25p1x1xjh0gpnr"
API_SECRET = "2mj8gklqv9hjf51a3vuf31mm4xgety5f"
# ==============================================================

def kite_login_and_get_token():
    # 1. Initialize KiteConnect
    kite = KiteConnect(api_key=API_KEY)
    login_url = kite.login_url()
    
    # 2. Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless") # Enabled for headless DietPi

    # 3. Initialize WebDriver
    print("Starting Chrome browser...")
    try:
        driver = webdriver.Chrome(options=chrome_options)
        print("Browser started successfully.")
    except Exception as drv_err:
        print(f"Error during browser setup: {drv_err}")
        return

    try:
        # 4. Open Kite Login Page (via API Login URL)
        print(f"Opening Login URL: {login_url}")
        driver.get(login_url)
        
        wait = WebDriverWait(driver, 20)
        
        # 5. Enter User ID
        print("Entering User ID...")
        user_input = wait.until(EC.presence_of_element_located((By.ID, "userid")))
        user_input.send_keys(USER_ID)
        
        # 6. Enter Password
        print("Entering Password...")
        pass_input = driver.find_element(By.ID, "password")
        pass_input.send_keys(PASSWORD)
        
        # 7. Click Login
        login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        login_button.click()
        
        # 8. Handle TOTP with Retry Logic
        max_retries = 3
        logged_in = False
        for attempt in range(max_retries):
            print(f"TOTP Attempt {attempt + 1}/{max_retries}...")
            
            clean_secret = TOTP_SECRET.replace(" ", "")
            totp = pyotp.TOTP(clean_secret)
            otp_code = totp.now()
            print(f"TOTP Generated: {otp_code}")
            
            # Wait for pin/totp input field
            pin_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='number'], input[label='TOTP']")))
            pin_input.clear()
            pin_input.send_keys(otp_code)
            
            print("Verifying TOTP...")
            time.sleep(5)
            
            # Check for redirect to localhost (which signals success in request_token acquisition)
            current_url = driver.current_url
            if "request_token=" in current_url:
                print("Captured redirect URL successfully!")
                logged_in = True
                break
            else:
                try:
                    error_msg = driver.find_element(By.CLASS_NAME, "su-error").text
                    print(f"Login error: {error_msg}. Retrying...")
                    time.sleep(2)
                except:
                    # Maybe it's still loading or redirected elsewhere
                    if "request_token=" in driver.current_url:
                        logged_in = True
                        break
                    print("Still on login page. Retrying TOTP...")
        
        if not logged_in:
            print("Failed to reach redirect URL after max retries.")
            return

        # 9. Extract Request Token
        final_url = driver.current_url
        print(f"Final URL: {final_url}")
        
        parsed_url = urllib.parse.urlparse(final_url)
        params = urllib.parse.parse_qs(parsed_url.query)
        
        if 'request_token' not in params:
            print("Error: request_token not found in the redirect URL.")
            return
            
        request_token = params['request_token'][0]
        print(f"Captured Request Token: {request_token}")
        
        # 10. Exchange Request Token for Access Token
        print("Exchanging Request Token for Access Token...")
        try:
            session = kite.generate_session(request_token, api_secret=API_SECRET)
            access_token = session["access_token"]
            
            print("==================================================")
            print(f"SUCCESS! Your Access Token for today is:")
            print(f"{access_token}")
            print("==================================================")
            
            # Save it to a file
            with open("access_token.txt", "w") as f:
                f.write(access_token)
            print("Saved access_token.txt")
            
        except Exception as api_err:
            print(f"API Error during token exchange: {api_err}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Closing browser in 5 seconds...")
        time.sleep(5)
        driver.quit()

if __name__ == "__main__":
    kite_login_and_get_token()

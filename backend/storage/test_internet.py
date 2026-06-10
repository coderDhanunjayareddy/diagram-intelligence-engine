import urllib.request
import urllib.error

try:
    print("Testing internet access...")
    response = urllib.request.urlopen("https://www.google.com", timeout=5)
    print("Success! Status code:", response.getcode())
except Exception as e:
    print("Failed to access internet:", e)

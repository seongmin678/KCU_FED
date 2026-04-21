import urllib.request
import json

req = urllib.request.Request("http://localhost:5001/api/chat", method="POST", data=json.dumps({"message": "최근 3년간의 금리 그래프 보여줘"}).encode('utf-8'), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read())
    print("tickers:", result.get("tickers"))
    print("start_yr:", result.get("start_yr"))

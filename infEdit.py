import requests
import json
from base64 import b64encode

url = "https://modelslab.com/api/v6/image_editing/magic_mix"

def image_to_base64(image_path : str) -> str :
    with open(image_path, 'rb') as image_bytes :
        return b64encode(image_bytes.read()).decode('utf-8')
    
payload = json.dumps({
    "key":"AjCN7NlRwc9ARkMVt7XKF9y51ZrNmrsAUcd0LRd3yXJif1SL9pAAhwVVMHW0",
    "prompt":"Bed",
    "height":768,
    "width":768,
    "image":image_to_base64("img.png"),
    "kmax":0.5,
    "kmin":0.3,
    "mix_factor":0.5,
    "samples":1,
    "negative_prompt":"low quality",
    "seed":1829183163,
    "steps":20,
    "webhook": None,
    "track_id": None,
    "base64": True,
})

headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
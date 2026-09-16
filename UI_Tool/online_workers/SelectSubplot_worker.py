import json
import random
import cv2
import numpy as np
from PIL import Image
import uuid
import argparse
import requests
import time
import re

from UI_Tool.utils.utils import *
from UI_Tool.utils.server_utils import *

from UI_Tool.online_workers.base_tool_worker import BaseToolWorker

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_sorted_bbox(img):
    image = np.array(img)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    gaussian = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
    edged = cv2.Canny(gaussian, 100, 200) 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key= len, reverse=True)[:10]
    longest_contour = sorted_contours[0]
    x,y,w,h = cv2.boundingRect(np.concatenate(longest_contour))
    longest_box = w + h

    sorted_bbox = []
    for c in sorted_contours:
        x,y,w,h = cv2.boundingRect(np.concatenate(c))
        if longest_box - 10 < w + h <= longest_box + 10:
            repeat = False
            for bb in sorted_bbox:
                if abs(bb['x1'] - x) < 10 and abs(bb['y1'] - y) < 10 and abs(bb['x2'] - (x+w)) < 10 and abs(bb['y2'] - (y+h)) < 10:
                    repeat = True
            if not repeat:
                sorted_bbox.append({'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h})
    sorted_bbox_json = json.dumps(sorted_bbox)
    
    return sorted_bbox_json

def get_subplot_images(img, sorted_bbox_json):
    sorted_bbox = json.loads(sorted_bbox_json)

    image = np.array(img)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    result_images = []

    for bbox in sorted_bbox:
        
        masked_image = image.copy()
        
        for other_bbox in sorted_bbox:
            if other_bbox != bbox:
                x1, y1, x2, y2 = other_bbox['x1'], other_bbox['y1'], other_bbox['x2'], other_bbox['y2']
                cv2.rectangle(masked_image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        pil_image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        result_images.append(pil_image)

    return result_images
    
def create_llm_payload(param, image_base64, model_name):
    prompt = f"""Find the bounding box of the UI element described as: "{param}".
Output the bounding box in the format: [xmin, ymin, xmax, ymax]
Only output the list of coordinates, nothing else."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                }
            ]
        }
    ]

    data = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.0
    }
    return data

def call_llm(data, api_url):
    headers = {
        "Content-Type": "application/json",
    }
    
    attempt = 0
    while attempt < 3:
        try:
            if os.environ.get("ZOOM_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["ZOOM_API_KEY"]
            response = requests.post(api_url, headers=headers, json=data, timeout=float(os.environ.get("ZOOM_API_TIMEOUT", "120")))
            if response.status_code != 200:
                logger.error(f"Error: {response.status_code}, {response.text}")
                time.sleep(2)
                attempt += 1
                continue
            
            response = response.json()
            content = response["choices"][0]["message"]["content"]
            print("text response: ", content)
            return content

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
            attempt += 1
    
    return "Some thing wrong and there is nothing return"

def extract_bbox(response_text, width, height):
    try:
        # Match [xmin, ymin, xmax, ymax]
        match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response_text)
        if match:
            xmin, ymin, xmax, ymax = map(int, match.groups())
            logger.info(f"Raw bbox from model: {xmin}, {ymin}, {xmax}, {ymax}")
            # Convert normalized coordinates to pixel coordinates
            # Using 999 as the scale factor as requested
            x1 = int(round(xmin / 999 * width))
            y1 = int(round(ymin / 999 * height))
            x2 = int(round(xmax / 999 * width))
            y2 = int(round(ymax / 999 * height))
            
            # Clamp coordinates to image boundaries
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Add padding (20% of the image size on each side, resulting in ~40% context)
            pad_w = int(width * 0.2)
            pad_h = int(height * 0.2)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(width, x2 + pad_w)
            y2 = min(height, y2 + pad_h)
            
            return (x1, y1, x2, y2)
    except Exception as e:
        logger.error(f"Failed to extract bbox: {e}")
    return None

    
class SelectSubplotToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_name = "ZoomInSubfigure",
                 model_path = "", 
                 model_base = "", 
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 api_url = None,
                 llm_model_name = None,
                 ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore
            )
        self.api_url = api_url
        self.llm_model_name = llm_model_name

        
    def init_model(self):
        logger.info(f"No need to initialize model {self.model_name}.")
        self.model = None

        
    def generate(self, params):
        generate_param = params["param"]
        image = params["image"]
        
        if generate_param is None or image is None:
            logger.error("Missing 'param' or 'image' in the input parameters.")
            return {"text": "Missing 'param' or 'image' in the input parameters.", "edited_image": None}
        
        ret = {"text": "", "error_code": 0}
        
        try:
            img = base64_to_pil(image).convert("RGB")
            width, height = img.size
            
            # Create payload with original image
            data = create_llm_payload(generate_param, image, self.llm_model_name)
            
            # Call LLM
            response = call_llm(data, self.api_url)
            
            # Extract bbox
            bbox = extract_bbox(response, width, height)
            
            if bbox:
                x1, y1, x2, y2 = bbox
                # Crop image
                cropped_img = img.crop((x1, y1, x2, y2))
                ret["text"] = f"Detected region: {bbox}"
                ret['edited_image'] = encode_image_to_base64(cropped_img)
            else:
                ret["text"] = f"Could not detect region for: {generate_param}. Response: {response}"
                ret['edited_image'] = None
            
        except Exception as e:
            logger.error(f"Error when selecting subplot: {e}")
            ret["text"] = f"Error when selecting subplot: {e}"
            ret['edited_image'] = None

        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20039)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://127.0.0.1:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-model-name", type=str, default="qwen3-vl-32b-instruct")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = SelectSubplotToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        api_url=args.api_url,
        llm_model_name=args.llm_model_name
    )
    worker.run()
    
    
    

    

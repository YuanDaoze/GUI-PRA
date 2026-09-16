"""
A model worker executes the model.
"""

import uuid
import os
import re
import io
import argparse
import torch
import numpy as np
from PIL import Image

from UI_Tool.utils.utils import *
from UI_Tool.utils.server_utils import *
import matplotlib.pyplot as plt

from UI_Tool.online_workers.base_tool_worker import BaseToolWorker
import traceback

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig


import cv2
from PIL import Image, ImageDraw
from PIL import Image, ImageDraw, ImageColor
import base64
import math # <--- 新增导入

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")
model_semaphore = None


def draw_pixel_marker(image: Image.Image, point: list, color_name: str = 'red', size: int = 5) -> Image.Image:
    """
    使用最底层的putpixel()方法在图像副本上绘制一个实心方块标记。
    这个方法保证了除了标记本身，不会对图像造成任何其他改变。

    Args:
        image: 原始Pillow图像对象。
        point: 标记的中心(x, y)坐标。
        color_name: 标记的颜色。
        size: 标记的边长（像素）。

    Returns:
        一个带有标记的新的Pillow图像对象。
    """
    # 1. 创建一个精确副本进行操作，保护原图
    edited_image = image.copy()
    
    # 2. 获取颜色值
    try:
        color = ImageColor.getrgb(color_name)
    except ValueError:
        color = (255, 0, 0) # 默认为红色

    # 3. 计算绘制区域的左上角坐标
    x, y = int(point[0]), int(point[1])
    offset = size // 2
    
    # 4. 直接在像素级别上色
    # 这个循环是唯一修改图像的地方，它不可能改变图像的分辨率或模式。
    for i in range(x - offset, x + offset + 1):
        for j in range(y - offset, y + offset + 1):
            # 确保不会画出图像边界
            if 0 <= i < edited_image.width and 0 <= j < edited_image.height:
                edited_image.putpixel((i, j), color)

    return edited_image
# ==========================================================

# ==================== 新增的五角星绘制函数 ====================
def draw_star(image: Image.Image, center: list, size: int, color_name: str = 'red') -> Image.Image:
    """
    在图像上绘制一个带黑色边框的五角星标记。

    Args:
        image: 原始Pillow图像对象。
        center: 五角星的中心 (x, y) 坐标。
        size: 五角星外接圆的半径（决定了五角星的大小）。
        color_name: 五角星的填充颜色。

    Returns:
        一个带有五角星标记的新的Pillow图像对象。
    """
    draw = ImageDraw.Draw(image)
    try:
        fill_color = ImageColor.getrgb(color_name)
    except ValueError:
        fill_color = (255, 0, 0)

    x, y = center
    start_angle = -math.pi / 2
    vertices = []
    outer_radius = size
    inner_radius = outer_radius / 2.5

    for i in range(10):
        radius = outer_radius if i % 2 == 0 else inner_radius
        angle = start_angle + i * math.pi / 5
        vx = x + radius * math.cos(angle)
        vy = y + radius * math.sin(angle)
        vertices.append((vx, vy))

    # 先绘制填充
    draw.polygon(vertices, fill=fill_color)
    # 再绘制黑色边框
    # 将顶点列表闭合（最后回到第一个点）
    outline_vertices = vertices + [vertices[0]]
    draw.line(outline_vertices, fill='black', width=max(1, size // 10))

    return image
# ==========================================================


# 新增一个更智能的坐标提取函数，取代旧的 extract_points
def extract_and_process_coordinates(molmo_output, image_w, image_h):
    """
    Parses the model's output string to find coordinates.
    - Handles single <point> tags.
    - Handles <points> (bbox) tags by calculating the center point.
    Returns a list of (x, y) tuples to be drawn.
    """
    points_to_draw = []
    
    # 1. 寻找单个点: <point x="..." y="...">
    single_point_pattern = r'<point\s+x="([0-9.]+)"\s+y="([0-9.]+)"'
    for match in re.finditer(single_point_pattern, molmo_output):
        try:
            x_rel, y_rel = float(match.group(1)), float(match.group(2))
            # 将相对坐标转换为绝对像素坐标
            abs_x = (x_rel / 100.0) * image_w
            abs_y = (y_rel / 100.0) * image_h
            points_to_draw.append((abs_x, abs_y))
        except (ValueError, IndexError):
            continue

    # 2. 寻找边界框: <points x1="..." y1="..." x2="..." y2="...">
    bbox_pattern = r'<points\s+x1="([0-9.]+)"\s+y1="([0-9.]+)"\s+x2="([0-9.]+)"\s+y2="([0-9.]+)"'
    for match in re.finditer(bbox_pattern, molmo_output):
        try:
            x1, y1, x2, y2 = map(float, [match.group(1), match.group(2), match.group(3), match.group(4)])
            # 计算边界框的中心点，并转换为绝对像素坐标
            center_x = ((x1 + x2) / 2 / 100.0) * image_w
            center_y = ((y1 + y2) / 2 / 100.0) * image_h
            points_to_draw.append((center_x, center_y))
        except (ValueError, IndexError):
            continue
            
    return points_to_draw
    
# ==========================================================


def extract_points(molmo_output, image_w, image_h):
    all_points = []
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return np.array(all_points)  # Ensure it's always a NumPy array

def show_points(coords, labels, ax, marker_size=375):
    # Only plot if there are points
    if len(coords) == 0:
        return
    
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def create_image_with_points(image, coords, labels, marker_size=375):
    fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
    ax.imshow(image)
    image_format = image.format.lower()
    if image_format not in ['png', 'jpeg', 'jpg']:
        image_format = 'png'

    # Only show points if there are any valid coordinates
    show_points(coords, labels, ax, marker_size)

    plt.axis('off')  # Turn off axis

    # Convert the figure to a PIL image
    buf = BytesIO()
    plt.savefig(buf, format=image_format, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

class MolmoToolWorker(BaseToolWorker):
    def __init__(self, 
                 controller_addr, 
                 worker_addr = "auto",
                 worker_id = worker_id, 
                 no_register = False,
                 model_path = "allenai/Molmo-7B-D-0924", 
                 model_base = "", 
                 model_name = "Point",
                 load_8bit = False, 
                 load_4bit = False, 
                 device = "auto",
                 limit_model_concurrency = 1,
                 host = "0.0.0.0",
                 port = None,
                 model_semaphore = None,
                 max_length = 2048,
                 args = None,
                 ):
        self.max_length = max_length
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
            model_semaphore,
            args=args
            )

        
    def init_model(self):
        logger.info(f"Initializing model {self.model_name}...")
        quant_config = None
        print(f"load8bitarg:{self.args.load_8bit}\n load4bitarg:{self.args.load_4bit}")
        self.load_4bit = self.args.load_4bit 
        self.load_8bit = self.args.load_8bit 
        
        print(f"load8bit:{self.load_8bit}\n load4bit:{self.load_4bit}")
        if self.load_4bit or self.load_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=self.load_8bit,
                load_in_4bit=self.load_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            logger.info(f"Using quantization config: {quant_config}")

        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto',
            quantization_config=quant_config
        )

    def load_image(self, image_input):
        """
        从文件路径、Base64编码字符串或已有的PIL.Image对象加载图像。
        返回一个 PIL.Image 对象和一个文件路径（如果存在）。
        """
        # ==================== 新增逻辑开始 ====================
        # 1. 检查输入是否已经是PIL图像对象
        if isinstance(image_input, Image.Image):
            # 如果是，直接返回该对象，文件路径为None
            return image_input, None
        # ===================== 新增逻辑结束 =====================

        # 2. 如果输入是字符串，则执行原有的逻辑
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                # 是一个有效的文件路径
                return Image.open(image_input).convert("RGB"), image_input
            else:
                # 尝试作为Base64字符串解码
                try:
                    image_data = base64.b64decode(image_input)
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    
                    # 为Base64图像创建一个临时文件，因为下游函数可能需要路径
                    temp_dir = "temp_images"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.png")
                    image.save(temp_file_path)
                    
                    return image, temp_file_path
                except Exception as e:
                    logger.error(f"Failed to decode base64 image: {e}")
                    raise ValueError("Input string is not a valid file path or base64 encoded image.")
        
        # 3. 如果输入不是PIL图像也不是字符串，则抛出错误
        raise TypeError(f"Unsupported input type: {type(image_input)}. Expected PIL.Image or str.")
    
    def generate(self, params):
        generate_param = params["param"]
        image_b64 = params.get("image")
        image_input = params.get("image")
        
        # if not image_input:
        #     logger.error("Missing required parameter: image")
        #     return {"text": "Missing required parameter: image", "edited_image": None, "error_code": 1}
        
        # ret = {"text": "", "error_code": 0}
    
        # try:
        #     # 1. 解码得到原始的高分辨率图像，并立即创建一个绘图专用的副本
        #     original_high_res_image = base64_to_pil(image_b64)
            
        #     image_1, image_path_for_processing = self.load_image(image_b64)
        #     # output_path_2 = os.path.join("runtime/debug/image_1.jpg")
        #     # image_1.save(output_path_2)
            
        #     image_for_drawing = original_high_res_image.copy()
            
        #     logger.info(f"成功解码图像，原始尺寸: {original_high_res_image.size}")
            
        #     text_prompt = "Point to the {} in the scene.".format(generate_param)
# 注意：这里的变量名虽然叫 image_b64，但实际上它可能是 Base64 也可能是路径
        raw_image_input = params.get("image") 
        
        if not raw_image_input:
            logger.error("Missing required parameter: image")
            return {"text": "Missing required parameter: image", "edited_image": None, "error_code": 1}
        
        ret = {"text": "", "error_code": 0}
    
        try:
            # ============ 修改开始 ============
            # 1. 移除 base64_to_pil 调用，因为它处理不了路径
            # original_high_res_image = base64_to_pil(image_b64) <--- 删除这一行
            
            # 2. 直接使用你写好的 load_image，它能同时处理 Path 和 Base64
            # load_image 返回两个值：PIL对象 和 路径
            original_high_res_image, image_path_for_processing = self.load_image(raw_image_input)
            
            # 3. 检查是否加载成功 (防止 load_image 内部出错但未抛出异常的情况)
            if original_high_res_image is None:
                raise ValueError("Failed to load image from input (neither valid path nor base64).")

            # 4. 创建绘图副本
            image_for_drawing = original_high_res_image.copy()
            
            # 现在这里调用 .size 就安全了
            logger.info(f"成功解码图像，原始尺寸: {original_high_res_image.size}")
            # ============ 修改结束 ============
            
            text_prompt = "Point to the {} in the scene.".format(generate_param)
            
            # 2. 执行模型推理（传入原始图像，因为我们已经确认它不会被修改）
            with torch.no_grad():
                inputs = self.processor.process(
                    images=[original_high_res_image], # 使用原图进行推理
                    text=text_prompt,
                )
                # ... (后续的模型推理代码不变) ...
                inputs["images"] = inputs["images"].to(torch.bfloat16)
                inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = self.model.generate_from_batch(
                        inputs,
                        GenerationConfig(max_new_tokens=self.max_length, stop_strings="<|endoftext|>"),
                        tokenizer=self.processor.tokenizer,
                        use_cache=False
                    )
                    generated_tokens = output[0, inputs['input_ids'].size(1):]
                    response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    if response is not None:
                        ret["text"] = response
    
            logger.info(f"模型响应: {response}")
    
            # 3. 在高分辨率的绘图副本上进行操作
            width, height = image_for_drawing.size
            coords_to_draw = extract_and_process_coordinates(response, width, height)
            # output_path = os.path.join("runtime/debug/image_for_drawing.jpg")
            # image_for_drawing.save(output_path)
            # output_path_2 = os.path.join("runtime/debug/original_high_res_image.jpg")
            # original_high_res_image.save(output_path_2)
            if coords_to_draw:
                logger.info(f"找到 {len(coords_to_draw)} 个点进行绘制。")
                for point_coords in coords_to_draw:
                    # # 【关键】调用最简单、最不可能出错的像素级绘图函数
                    # image_for_drawing = draw_pixel_marker(image_for_drawing, point_coords, color_name='red', size=10)
                    # ==================== 【关键修改】 ====================
                    # 调用新的五角星函数，并动态设置一个更大的尺寸
                    # 五角星的大小为其外接圆半径，设置为图像最长边的 2.5%
                    star_radius = int(max(width, height) * 0.025) 
                    image_for_drawing = draw_star(image_for_drawing, point_coords, size=star_radius, color_name='red')
                    # ======================================================
                
                # 4. 编码最终的绘图结果
                ret['edited_image'] = pil_to_base64(image_for_drawing)
                logger.info("成功在高清图上绘制像素标记并编码返回。")
            else:
                ret["text"] = f"Error. Corrdinates not found."
                ret['edited_image'] = None
                logger.warning("模型未返回有效坐标，不生成编辑图像。")
                
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc()) 
            ret["text"] = f"Error: {e}"
            ret['edited_image'] = None
        
        return ret
    
    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "point",
                "description": "Identify a point in the image based on a natural language description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier or path of the image in which to locate the point, e.g., 'img_1'."
                        },
                        "description": {
                            "type": "string",
                            "description": "A natural language description of the point of interest, e.g., 'the dog’s nose', 'center of the clock', 'the tallest tree'."
                        }
                    },
                    "required": ["image", "description"]
                }
            }
        }
        return instruction 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20027)
    parser.add_argument("--worker-address", type=str,
        default="auto")
    parser.add_argument("--controller-address", type=str,
        default="http://127.0.0.1:20001")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--model_path", type=str, default="allenai/Molmo-7B-D-0924")
    parser.add_argument("--gpu_usage_limit",type=float, default=None)
    parser.add_argument("--load-8bit", type=str2bool, default=False, help="Use 8-bit quantization")
    parser.add_argument("--load-4bit",  type=str2bool, default=False, help="Use 4-bit quantization (nf4)")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = MolmoToolWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        host = args.host,
        port = args.port,
        no_register = args.no_register,
        model_path = args.model_path,
        args = args,
    )
    worker.run()

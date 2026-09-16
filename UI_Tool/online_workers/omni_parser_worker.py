import torch
from PIL import Image
import base64
import argparse
import uuid
import os
from io import BytesIO
import json # <--- 导入json库，用于更好地格式化文本输出

# 导入您工具所需的核心库
from UI_Tool.online_workers.util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

# 导入框架相关的基类和工具
from UI_Tool.online_workers.base_tool_worker import BaseToolWorker
from UI_Tool.utils.server_utils import build_logger
from UI_Tool.utils.utils import *
from UI_Tool.utils.server_utils import *

# --- Boilerplate Initialization ---
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"omni_parser_worker_{worker_id}.log")

# ==================== NEW FUNCTION START ====================
# def encode_pil_to_png_base64(image: Image.Image) -> str:
#     """
#     将 PIL.Image 对象编码为 PNG 格式的 Base64 字符串。
#     这确保了输出格式与 SAM Worker 保持一致。
#     """
#     buffered = BytesIO()
#     image.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return img_str
# ===================== NEW FUNCTION END =====================


class OmniParserWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 som_model_path=None,
                 caption_model_path=None,
                 model_name="omni_parser",
                 device="cuda",
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=120.0,
                 ):
        
        self.som_model_path = som_model_path
        self.caption_model_path = caption_model_path
        
        super().__init__(
            controller_addr, worker_addr, worker_id, no_register,
            None, None, model_name, False, False, device,
            limit_model_concurrency, host, port, model_semaphore,
            wait_timeout, task_timeout,
        )

    def init_model(self):
        """
        加载并初始化 OmniParser 所需的两个模型。
        """
        logger.info(f"Initializing model {self.model_name}...")
        logger.info(f"Loading SOM model from: {self.som_model_path}")
        self.som_model = get_yolo_model(self.som_model_path)
        self.som_model.to(self.device)
        logger.info("SOM model loaded successfully.")

        logger.info(f"Loading Caption model from: {self.caption_model_path}")
        self.caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=self.caption_model_path,
            device=self.device
        )
        logger.info("Caption model loaded successfully.")
        
        self.som_model.eval()

    # def load_image(self, image_input: str):
    #     """
    #     从文件路径或 Base64 编码的字符串加载图像。
    #     返回一个 PIL.Image 对象和一个临时文件路径（如果需要）。
    #     """
    #     if len(image_input) < 4096 and os.path.isfile(image_input):
    #         return Image.open(image_input).convert("RGB"), image_input
    #     else:
    #         try:
    #             image_data = base64.b64decode(image_input)
    #             image = Image.open(BytesIO(image_data)).convert("RGB")
                
    #             temp_dir = "temp_images"
    #             os.makedirs(temp_dir, exist_ok=True)
    #             temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.png")
    #             image.save(temp_file_path)
                
    #             return image, temp_file_path
    #         except Exception as e:
    #             logger.error(f"Failed to decode base64 image: {e}")
    #             raise ValueError("Input is not a valid file path or base64 encoded image.")
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
            if len(image_input) < 4096 and os.path.isfile(image_input):
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



    @torch.inference_mode()
    def generate(self, params):
        """
        执行核心的图像解析任务。
        """
        image_input = params.get("image")
        box_threshold = params.get("box_threshold", 0.05)
        iou_threshold = params.get("iou_threshold", 0.7)
        
        if not image_input:
            logger.error("Missing required parameter: image")
            return {"text": "Missing required parameter: image", "edited_image": None, "error_code": 1}

        temp_file_path_generated = None
        try:
            image, image_path_for_processing = self.load_image(image_input)
            if isinstance(image_input, str) and not (len(image_input) < 4096 and os.path.isfile(image_input)):
                temp_file_path_generated = image_path_for_processing
            
            logger.info(f"Processing image: {image_path_for_processing}, Size: {image.size}")

            logger.info("Performing OCR...")
            ocr_bbox_rslt, _ = check_ocr_box(
                image_path_for_processing, display_img=False,
                output_bb_format='xyxy', use_paddleocr=True
            )
            text, ocr_bbox = ocr_bbox_rslt
            logger.info(f"OCR found {len(text)} text segments.")

            logger.info("Performing SOM detection and captioning...")
            # edited_image, _, parsed_content_list = get_som_labeled_img(
            #     image=image=image_path_for_processing, som_model=self.som_model,
            #     BOX_TRESHOLD=box_threshold, output_coord_in_ratio=True,
            #     ocr_bbox=ocr_bbox, draw_bbox_config={},
            #     caption_model_processor=self.caption_model_processor,
            #     ocr_text=text, use_local_semantics=True,
            #     iou_threshold=iou_threshold, scale_img=False, batch_size=128
            # )
            edited_image, _, parsed_content_list = get_som_labeled_img(
                image,  # <-- 关键修改：将参数名从 image_path 改为 image，并传递图像对象
                self.som_model,
                BOX_TRESHOLD=box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config={'text_scale': 1.0, 'text_thickness': 1, 'text_padding': 2, 'thickness': 2},
                caption_model_processor=self.caption_model_processor,
                ocr_text=text,
                use_local_semantics=True,
                iou_threshold=iou_threshold,
                scale_img=False,
                batch_size=128
            )
            # dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, self.som_model, BOX_TRESHOLD = self.config['BOX_TRESHOLD'], output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)
            logger.info(f"Parsing complete. Found {len(parsed_content_list)} elements.")
            print(edited_image)
            print(parsed_content_list)
            response_text_obj = {
                "tool_response_from": "omni_parser",
                "status": "success",
                "parsed_elements": parsed_content_list
            }
            
            # ==================== MODIFICATION START ====================
            
            # 6. 使用新的函数将处理后的图像编码为PNG Base64，确保格式统一
            # edited_image_base64 = encode_pil_to_png_base64(edited_image)
            # edited_image_pil = pil_to_base64(edited_image)

            # 7. 构建最终返回的字典
            ret = {
                "text": json.dumps(response_text_obj, indent=2), # 将字典转为格式化的JSON字符串
                "edited_image": edited_image,
                "error_code": 0
            }
            
            return ret
            
            # ===================== MODIFICATION END =====================
            
        except Exception as e:
            logger.error(f"Error during OmniParser inference: {e}", exc_info=True)
            response_text_obj = {
                "tool_response_from": "omni_parser",
                "status": "failed",
                "error": str(e)
            }
            return {"text": json.dumps(response_text_obj, indent=2), "edited_image": None, "error_code": 1}
        finally:
            if temp_file_path_generated and os.path.exists(temp_file_path_generated):
                os.remove(temp_file_path_generated)

    # ... get_tool_instruction 和 __main__ 部分保持不变 ...
    def get_tool_instruction(self):
        """
        定义工具的接口规范，供规划模型调用。
        """
        instruction = {
            "type": "function",
            "function": {
                "name": "omni_parser",
                "description": "Parses a UI or general image to detect, label, and locate interactive elements like icons, buttons, and text fields.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier, path, or base64 encoded string of the image to parse, e.g., 'img_1'."
                        },
                        "box_threshold": {
                            "type": "number",
                            "description": "Optional. The confidence threshold for object detection. Defaults to 0.05."
                        },
                        "iou_threshold": {
                             "type": "number",
                             "description": "Optional. The IoU threshold for non-maximum suppression. Defaults to 0.7."
                        }
                    },
                    "required": ["image"]
                }
            }
        }
        return instruction

# ==================== 新增的本地调试函数 ====================
def debug_worker_locally(args):
    """
    在本地直接测试worker的核心功能，不启动Web服务。
    """
    print("--- [本地调试模式] ---")
    
    # 1. 初始化Worker（不注册）
    worker = OmniParserWorker(
        controller_addr="http://localhost:9999",  # 虚拟地址，无影响
        no_register=True,                          # 设置为True，不向控制器注册
        som_model_path=args.som_model_path,
        caption_model_path=args.caption_model_path,
        device=args.device,
        port=args.port                              # <-- 新增这一行，传递端口号
    )
    
    # 2. 加载模型
    worker.init_model()
    
    # 3. 准备输入数据
    image_path = args.debug_image_path
    if not os.path.exists(image_path):
        print(f"!!! 错误: 调试图片未找到，路径: {image_path}")
        return
        
    print(f"正在加载调试图片: {image_path}")
    # 将图片文件编码为base64字符串，模拟真实请求
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    params = {
        "image": image_b64,
        "box_threshold": 0.1, # 可以按需调整测试参数
    }
    
    # 4. 调用核心的 generate 方法
    print("\n--- 调用 worker.generate() ---")
    try:
        result = worker.generate(params)
    except Exception as e:
        print(f"!!! 调用generate时发生未捕获的异常: {e}")
        return

    # 5. 打印和保存结果
    print("\n--- [函数返回结果] ---")
    
    # 打印文本输出
    print("\n[文本部分]:")
    print(result.get("text"))
    
    # 解码并保存图像输出
    edited_image_b64 = result.get("edited_image")
    if edited_image_b64:
        print("\n[图像部分]:")
        output_image_data = base64.b64decode(edited_image_b64)
        output_path = "debug_output.png"
        with open(output_path, "wb") as f:
            f.write(output_image_data)
        print(f"--> 已将标注后的图像保存至: {os.path.abspath(output_path)}")
    else:
        print("\n[图像部分]: 未返回图像。")
    
    print("\n--- [本地调试结束] ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20004)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:30001")
    parser.add_argument("--som-model-path", type=str, required=True, help="Path to the SOM (YOLO) model checkpoint.")
    parser.add_argument("--caption-model-path", type=str, required=True, help="Path to the Florence-2 caption model.")
    parser.add_argument("--model-name", type=str, default="omni_parser")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    # ========= 新增的调试参数 =========
    parser.add_argument("--debug-local", action="store_true", help="在本地运行调试，不启动Web服务。")
    parser.add_argument("--debug-image-path", type=str, default="imgs/google_page.png", help="用于本地调试的图片路径。")
    # =================================

    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.debug_local:
        # 如果提供了 --debug-local 参数，则运行本地调试函数
        debug_worker_locally(args)
    else:
        # 否则，按原方式启动Web服务
        worker = OmniParserWorker(
            controller_addr=args.controller_address,
            worker_addr=args.worker_address,
            worker_id=worker_id,
            no_register=args.no_register,
            som_model_path=args.som_model_path,
            caption_model_path=args.caption_model_path,
            model_name=args.model_name,
            device=args.device,
            limit_model_concurrency=args.limit_model_concurrency,
            host=args.host,
            port=args.port,
        )
        worker.run()
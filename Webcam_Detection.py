import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import gradio as gr
from openai import OpenAI
import time

# YOLO 모델 로드
model = YOLO('yolo12x.pt')

# xAI API 설정
client = OpenAI(
    api_key="API Key",  # 유효한 xAI API 키로 교체
    base_url="https://api.x.ai/v1",  # xAI 문서에서 확인
)

# 객체 설명 함수 (Grok 3 사용)
def describe_object(object_name):
    """Grok 3를 사용해 객체 설명을 생성."""
    prompt = f"Describe in one sentence what a {object_name} is and what it is used for."
    try:
        response = client.chat.completions.create(
            model="grok-3",  # xAI 문서에서 모델 이름 확인
            messages=[{"role": "user", "content": prompt}]
        )
        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        print(f"Grok API error: {e}")
        return f"{object_name}: [description not available]"

# 웹캠 프레임 처리 함수
def process_webcam_frame(frame):
    """웹캠 프레임(BGR)을 처리하고 객체를 감지하며 설명을 생성."""
    if frame is None:
        print("Frame is None - possible causes: webcam not active, permissions denied, or no capture.")
        return None, "Please ensure webcam is active and permissions are granted."
    
    print(f"Webcam frame type: {type(frame)}, Shape: {getattr(frame, 'shape', 'N/A')}")
    
    try:
        # Webcam input is BGR, convert to RGB
        image = Image.fromarray(frame[..., ::-1])  # BGR -> RGB
    except Exception as e:
        print(f"Image conversion error: {e}")
        return None, "Invalid image format."
    
    try:
        results = model(image)
        print("Object detection completed.")
        annotated = results[0].plot()  # Annotated image (BGR)
        if annotated is None:
            print("Annotated image is None - detection failed.")
            return None, "Detection failed."
        
        # Convert annotated image from BGR to RGB
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Detection error: {e}")
        return None, "Error during object detection."
    
    descriptions = {}
    for box in results[0].boxes:
        try:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            if label not in descriptions:
                desc = describe_object(label)
                descriptions[label] = f"**{label.capitalize()}** – {desc}"
                print(f"Detected: {label}, Description: {desc}")
        except Exception as e:
            print(f"Error processing detection: {e}")
            continue
    descriptions_text = "\n\n".join(descriptions.values()) if descriptions else "No objects detected."
    
    return annotated_rgb, descriptions_text

# 업로드 이미지 처리 함수
def process_uploaded_image(frame):
    """업로드 이미지(RGB)를 처리하고 객체를 감지하며 설명을 생성."""
    if frame is None:
        print("Frame is None - invalid upload.")
        return None, "Please upload a valid image."
    
    print(f"Uploaded image type: {type(frame)}, Shape: {getattr(frame, 'shape', 'N/A')}")
    
    try:
        # Uploaded image is RGB
        image = Image.fromarray(np.array(frame)).convert('RGB')
    except Exception as e:
        print(f"Image conversion error: {e}")
        return None, "Invalid image format."
    
    try:
        results = model(image)
        print("Object detection completed.")
        annotated = results[0].plot()  # Annotated image (BGR)
        if annotated is None:
            print("Annotated image is None - detection failed.")
            return None, "Detection failed."
        
        # Convert annotated image from BGR to RGB
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Detection error: {e}")
        return None, "Error during object detection."
    
    descriptions = {}
    for box in results[0].boxes:
        try:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            if label not in descriptions:
                desc = describe_object(label)
                descriptions[label] = f"**{label.capitalize()}** – {desc}"
                print(f"Detected: {label}, Description: {desc}")
        except Exception as e:
            print(f"Error processing detection: {e}")
            continue
    descriptions_text = "\n\n".join(descriptions.values()) if descriptions else "No objects detected."
    
    return annotated_rgb, descriptions_text

# 웹캠 프레임 캡처 함수
def capture_webcam_frame():
    """웹캠에서 단일 프레임을 캡처."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return None, "Webcam could not be opened."
    
    ret, frame = cap.read()
    cap.release()
    if ret:
        return process_webcam_frame(frame)
    return None, "Failed to capture frame."

# Gradio 인터페이스
with gr.Blocks() as demo:
    gr.Markdown("## 👓 AI Smart Glasses – Object Detection Demo\nUpload an image or use the webcam button for detection.")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=["webcam", "upload"], label="Camera Input", interactive=True, width=640, height=480)
            webcam_button = gr.Button("Capture from Webcam")
        with gr.Column():
            output_video = gr.Image(label="Detected Images")
            output_text = gr.Markdown(label="Object Descriptions")
    
    # 웹캠 버튼 이벤트
    webcam_button.click(
        fn=capture_webcam_frame,
        inputs=None,
        outputs=[output_video, output_text]
    )
    
    # 이미지 업로드 이벤트
    input_image.change(
        fn=process_uploaded_image,
        inputs=input_image,
        outputs=[output_video, output_text]
    )

# 앱 실행 (로컬 환경)
demo.launch(share=True)

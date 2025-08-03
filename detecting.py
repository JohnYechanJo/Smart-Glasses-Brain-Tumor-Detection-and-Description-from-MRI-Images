import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
import gradio as gr
import pydicom
from PIL import Image
import time

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded paths
MODEL_WEIGHTS_PATH = '/Users/ivchxn/Desktop/Apple/yolov11l_trained_weights.pt'
TEST_DATASET_PATH = '/Users/ivchxn/Desktop/Apple/BrainTumor/BrainTumorYolov11/test/images'
OUTPUT_DIR = '/Users/ivchxn/Desktop/Apple/BrainTumor/BrainTumorYolov11/test/outputs'
TEMP_WEBCAM_DIR = '/Users/ivchxn/Desktop/Apple/BrainTumor/BrainTumorYolov11/test/webcam_temp'
# Model parameters
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.7

class BrainTumorDetector:
    def __init__(self):
        self.model = self._load_yolo_model()
        logger.info("BrainTumorDetector initialized successfully")

    def _load_yolo_model(self):
        try:
            if not Path(MODEL_WEIGHTS_PATH).exists():
                raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS_PATH}")
            
            model = YOLO(MODEL_WEIGHTS_PATH)
            logger.info("YOLOv11 model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLOv11 model: {e}")
            raise

    def process_image(self, image_input, image_name="uploaded_image"):
        try:
            # Handle both file path (str) and numpy array (from Gradio)
            if isinstance(image_input, str):
                if image_input.lower().endswith('.dcm'):
                    ds = pydicom.dcmread(image_input)
                    image = ds.pixel_array
                    if len(image.shape) == 2:  # Grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image = cv2.imread(image_input)
                    if image is None:
                        raise ValueError(f"Failed to read image: {image_input}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_input  # Assume numpy array from Gradio
            
            results = self.model(image, conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE)
            annotated = results[0].plot()
            detections = results[0].boxes
            
            if len(detections) == 0:
                return annotated, "No tumors detected"
            
            descriptions = []
            for box in detections:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = results[0].names.get(cls_id, "tumor")
                descriptions.append(f"{label} (Conf: {confidence:.2f})")
            
            return annotated, "\n".join(descriptions)
        except Exception as e:
            logger.error(f"Detection error for {image_name}: {e}")
            return None, f"Error: {str(e)}"

def process_dataset():
    try:
        detector = BrainTumorDetector()
        
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "detection_results.txt"
        with open(results_file, 'w') as f:
            f.write("Brain Tumor Detection Results\n\n")
        
        test_dir = Path(TEST_DATASET_PATH)
        if not test_dir.exists():
            raise FileNotFoundError(f"Test dataset directory not found: {test_dir}")
        
        # Support jpg, jpeg, png, and dcm
        image_files = list(test_dir.glob('*.[jjpp][ppnn][gg]')) + list(test_dir.glob('*.dcm'))
        
        if not image_files:
            raise FileNotFoundError("No valid images found in test dataset directory")
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            logger.info(f"Processing {image_path.name}")
            
            annotated, description = detector.process_image(str(image_path), image_path.name)
            
            if annotated is not None:
                output_image_path = output_dir / f"annotated_{image_path.name}"
                if image_path.suffix.lower() == '.dcm':
                    # Save DICOM as PNG
                    output_image_path = output_dir / f"annotated_{image_path.stem}.png"
                    Image.fromarray(annotated).save(output_image_path)
                else:
                    cv2.imwrite(str(output_image_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved annotated image: {output_image_path}")
            
            with open(results_file, 'a') as f:
                f.write(f"Image: {image_path.name}\n{description}\n\n")
        
        return f"Processing complete. Results saved to {OUTPUT_DIR}\nDetection results written to {results_file}"
        
    except Exception as e:
        logger.error(f"Dataset processing error: {e}")
        return f"Error: {str(e)}"

def process_webcam_image(image):
    try:
        detector = BrainTumorDetector()
        
        # Create temporary directory for webcam images
        temp_dir = Path(TEMP_WEBCAM_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save webcam image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_image_path = temp_dir / f"webcam_{timestamp}.png"
        cv2.imwrite(str(temp_image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved webcam image: {temp_image_path}")
        
        # Process the saved image
        annotated, description = detector.process_image(str(temp_image_path), temp_image_path.name)
        
        # Save annotated image to output directory
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = output_dir / f"annotated_{temp_image_path.name}"
        if annotated is not None:
            cv2.imwrite(str(output_image_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved annotated webcam image: {output_image_path}")
        
        # Append results to detection_results.txt
        results_file = output_dir / "detection_results.txt"
        with open(results_file, 'a') as f:
            f.write(f"Webcam Image: {temp_image_path.name}\n{description}\n\n")
        
        return annotated, description
    except Exception as e:
        logger.error(f"Webcam image processing error: {e}")
        return None, f"Error: {str(e)}"
    
def create_gradio_app():
    with gr.Blocks(title="Brain Tumor Detection with YOLOv11", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Brain Tumor Detection with YOLOv11")
        gr.Markdown("Upload an MRI image or capture from webcam to detect brain tumors. For educational purposes only.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Upload MRI Image")
                image_input = gr.Image(label="Upload MRI Image", type="numpy")
                upload_button = gr.Button("Analyze Uploaded Image")
                
                gr.Markdown("## Webcam Capture")
                webcam_input = gr.Image(label="Capture from Webcam", sources=["webcam"], type="numpy")
                webcam_button = gr.Button("Analyze Webcam Image")
            
            with gr.Column():
                image_output = gr.Image(label="Detected Tumors")
                text_output = gr.Textbox(label="Detection Results")
        
        
        upload_button.click(
            fn=process_image,
            inputs=image_input,
            outputs=[image_output, text_output]
        )
        
        webcam_button.click(
            fn=process_webcam_image,
            inputs=webcam_input,
            outputs=[image_output, text_output]
        )
        
    return demo

def process_image(image):
    detector = BrainTumorDetector()
    annotated, description = detector.process_image(image, "uploaded_image")
    if annotated is None:
        return None, "Error processing image"
    return annotated, description

def main():
    try:
        print("🚀 Starting brain tumor detection web interface...")
        print("⚠️ This tool is for educational purposes only!")
        
        demo = create_gradio_app()
        demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        logger.error(f"Main application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
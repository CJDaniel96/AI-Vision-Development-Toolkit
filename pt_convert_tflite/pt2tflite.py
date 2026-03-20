import torch
import torch.nn as nn
from ultralytics import YOLO
import onnx
import os
import tensorflow as tf
import numpy as np
import glob
from PIL import Image
import shutil
import argparse
import sys
from ultralytics.utils.downloads import download

def setup_calibration_data(calib_dir):
    """
    Ensures calibration data exists. If not, downloads COCO128.
    """
    if not calib_dir:
        calib_dir = os.path.join(os.getcwd(), "datasets", "coco128", "images", "train2017")
    
    if os.path.exists(calib_dir) and len(glob.glob(os.path.join(calib_dir, "*.*"))) > 0:
        print(f"Using existing calibration data at: {calib_dir}")
        return calib_dir

    print(f"Calibration data not found. Downloading COCO128 to {calib_dir}...")
    try:
        # Download COCO128 (approx 7MB)
        torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip', 'coco128.zip')
        import zipfile
        with zipfile.ZipFile('coco128.zip', 'r') as zip_ref:
            zip_ref.extractall('datasets')
        os.remove('coco128.zip')
        
        # Verify path after download
        expected_path = os.path.join("datasets", "coco128", "images", "train2017")
        if os.path.exists(expected_path):
             return expected_path
        else:
             print(f"Warning: COCO128 downloaded but path {expected_path} not found.")
             return calib_dir
    except Exception as e:
        print(f"Error downloading calibration data: {e}")
        return None

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchors from features.
    Updated to be more friendly to ONNX export (avoiding scalar issues).
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        
        # Create grid coordinates
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        
        # Check torch version for meshgrid indexing
        if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        else:
            sy, sx = torch.meshgrid(sy, sx)
            
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        
        # Create stride tensor matching the anchor points
        # shape: (h*w, 1)
        # We use multiplication by ones to broadcast the scalar 'stride' tensor.
        # This avoids using torch.full with a tensor (which causes export errors)
        # and avoids explicit Expand ops that onnx2tf might dislike.
        st = torch.ones((h * w, 1), dtype=dtype, device=device) * stride
        stride_tensor.append(st)
        
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def export_custom_yolo_int8(model_path, output_tflite_path, calibration_images_dir, input_shape=(640, 640)):
    # 1. Load and Patch the Model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    detect_module = model.model.model[-1]

    # --- Patching Start ---
    # This patch is required to adapt YOLOv8 export for certain NPUs (like QCS6490)
    # by restructuring the forward pass to be more TFLite friendly.
    def custom_forward_export(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        preds = self.forward_det_post(x)
        return preds, x[0], x[1], x[2]

    def forward_det_post(self, x):
        shape = x[0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y

    import types
    detect_module.forward_det_post = types.MethodType(forward_det_post, detect_module)
    detect_module.forward = types.MethodType(custom_forward_export, detect_module)
    # --- Patching End ---

    # 2. Export to ONNX
    print("Exporting patched model to ONNX...")
    dummy_input = torch.randn(1, 3, *input_shape)
    onnx_filename = output_tflite_path.replace(".tflite", ".onnx")
    
    try:
        # Use opset 11 as it is generally more compatible and avoids some FX graph issues
        torch.onnx.export(
            model.model,
            dummy_input,
            onnx_filename,
            opset_version=11,
            input_names=['image'],
            output_names=['output_0', 'output_1', 'output_2', 'output_3']
        )
        print("ONNX export successful.")
    except Exception as e:
         print(f"Error during ONNX export: {e}")
         return

    # 3. Prepare Calibration Data Generator
    print(f"Preparing calibration data from {calibration_images_dir}...")
    
    def representative_dataset_gen():
        img_paths = glob.glob(os.path.join(calibration_images_dir, "*.*"))
        if not img_paths:
            print("Warning: No images found for calibration!")
            return
            
        # Use up to 100 images for calibration
        for img_path in img_paths[:100]:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((input_shape[1], input_shape[0]))
                img_data = np.array(img, dtype=np.float32) / 255.0
                img_data = np.expand_dims(img_data, axis=0) 
                yield [img_data]
            except Exception:
                continue

    # 4. Convert ONNX to SavedModel using onnx2tf
    print("Converting ONNX to TensorFlow SavedModel (via onnx2tf)...")
    saved_model_dir = output_tflite_path.replace(".tflite", "_saved_model")
    
    # -osd: Output static dimension (optimization for NPU)
    # -oiqt: Optimize for input quantization
    # -v info: Set verbosity to info (required arg)
    cmd = f"onnx2tf -i {onnx_filename} -o {saved_model_dir} -osd -v info"
    ret_code = os.system(cmd)
    
    if ret_code != 0:
        print("Error: onnx2tf conversion failed. Ensure onnx2tf is installed.")
        return

    # 5. Convert SavedModel to W8A8 TFLite
    print("Quantizing to W8A8 TFLite...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # Typical NPU requirement: Int8 Input/Output
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model_quant = converter.convert()

        with open(output_tflite_path, 'wb') as f:
            f.write(tflite_model_quant)
        
        print(f"✅ Success! Encoded file: {output_tflite_path}")

    except Exception as e:
        print(f"Error during TFLite quantization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv8 .pt to TFLite (Int8) for HALCON/Smart Cameras")
    parser.add_argument("--model", type=str, required=True, help="Path to input .pt model")
    parser.add_argument("--output", type=str, default="model_w8a8.tflite", help="Path to output .tflite model")
    parser.add_argument("--calib_dir", type=str, default=None, help="Directory containing calibration images (default: downloads COCO128)")
    parser.add_argument("--img_size", type=int, default=640, help="Input image size (default: 640)")

    args = parser.parse_args()

    # Ensure model exists
    if not os.path.exists(args.model):
        print(f"Error: Input model {args.model} does not exist.")
        # Optional: Download standard model for testing
        if args.model == "yolov8n.pt":
             print("Downloading standard yolov8n.pt...")
             YOLO("yolov8n.pt")
        else:
             sys.exit(1)

    # Setup calibration data
    calib_path = setup_calibration_data(args.calib_dir)
    if not calib_path:
        print("Error: Could not setup calibration data. Aborting.")
        sys.exit(1)

    export_custom_yolo_int8(args.model, args.output, calib_path, (args.img_size, args.img_size))

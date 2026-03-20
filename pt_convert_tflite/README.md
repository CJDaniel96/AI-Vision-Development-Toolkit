# YOLOv8 PT to TFLite Converter for HALCON Smart Camera

This tool converts YOLOv8 PyTorch models (`.pt`) to TensorFlow Lite (`.tflite`) with **INT8 quantization**, which is typically required for AI accelerators (NPUs) on smart cameras (e.g., Qualcomm QCS6490 used in some HALCON-compatible cameras).

## Features

- **Automated Patching**: Patches the YOLOv8 model to ensure TFLite compatibility (replaces dynamic logic with static ops).
- **INT8 Quantization**: Converts weights to INT8 using a calibration dataset.
- **Auto-Calibration**: Automatically downloads COCO128 dataset for calibration if no local images are provided.
- **CLI Support**: Easy to use command-line interface.

## Requirements

The smart camera (or the PC performing the conversion) must have Python 3.8+ and the following libraries:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `ultralytics` (YOLOv8)
- `onnx`, `onnx2tf`, `onnxsim`, `sng4onnx` (Conversion tools)
- `tensorflow` (TFLite Converter)

> **Note:** If your smart camera has limited storage or cannot install these heavy libraries, **run this conversion on a PC**, then transfer the resulting `.tflite` file to the camera.

## Usage

### Basic Usage
Convert a model using default settings (downloads calibration data automatically):

```bash
python pt2tflite.py --model your_model.pt
```

### Advanced Usage

```bash
python pt2tflite.py \
  --model custom_best.pt \
  --output custom_model_int8.tflite \
  --img_size 640 \
  --calib_dir /path/to/your/images
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to the input `.pt` file (Required) | - |
| `--output` | Path to the output `.tflite` file | `model_w8a8.tflite` |
| `--calib_dir` | Directory containing images for calibration. If omitted, downloads COCO128. | `None` (auto-download) |
| `--img_size` | Input image size (square). | `640` |

## Troubleshooting

- **`onnx2tf` not found**: Ensure it is installed via pip and in your PATH.
- **Memory Errors**: Quantization can be memory intensive. If running on a camera fails, run on a powerful PC.
- **Missing Calibration Images**: The script needs at least 1-100 images to calibrate quantization parameters.

## Integration with HALCON

Once you have the `.tflite` file:
1. Transfer it to the smart camera.
2. Use HALCON's Deep Learning operators or the specific camera's AI inference runtime to load the model.

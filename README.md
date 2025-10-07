# Lane segmentation & Vehicle Dashboard 

This project processes a video to detect road lanes and vehicles, overlays a real-time dashboard with lane metrics, vehicle counts, and a mini-map. It leverages **Roboflow's lane detection API** and **YOLOv11-L** object detection model (via SAHI slicing for better detection in high-resolution frames).

---

## Features

* **Road lane detection** RF DETR form roboflow.
* **Vehicle detection** (car, truck, bus, motorbike, bicycle) using YOLOv11-L + SAHI.
* **Dashboard overlay**:

  * Lane visualization with colored polygons
  * Lane metrics (area %, width)
  * Vehicle count
  * Lane quality score
* **Mini-map** with lane positions and vehicles
* **Charts** for lane area percentage and lane width distribution

---

## Requirements

* Python 3.10+
* Packages (install via `pip install -r requirements.txt`):

  ```text
  opencv-python
  numpy
  matplotlib
  pillow
  sahi
  inference_sdk
  ```
* CUDA-enabled GPU recommended for YOLO inference
* Roboflow API key for lane detection

---

## Models Used

### 1. Lane Detection

* **API**: Roboflow Serverless Inference
* **Model ID**: `road-lane-instance-segmentation-y5bzk/1`
* **Usage**: Returns polygon coordinates of road lanes.
* **API Configuration**:

  ```python
  API_URL = "https://serverless.roboflow.com"
  API_KEY = "YOUR_ROBOFLOW_API_KEY"
  LANE_MODEL_ID = "road-lane-instance-segmentation-y5bzk/1"
  ```

### 2. Vehicle Detection

* **Model**: YOLOv11-L (PyTorch `.pt` file)
* **Loaded via**: SAHI `AutoDetectionModel` for slicing large images
* **Classes detected**: car, truck, bus, motorbike, bicycle
* **Configuration**:

  ```python
  YOLO_MODEL_PATH = "path_to_yolov11l.pt"
  confidence_threshold = 0.25
  device = "cuda" or "cpu"
  ```

---

## How to Use

1. **Set paths and API key** in `main_video.py`:

   ```python
   API_KEY = "YOUR_ROBOFLOW_API_KEY"
   VIDEO_PATH = "path_to_input_video.mp4"
   OUTPUT_VIDEO_PATH = "path_to_save_output_video.mp4"
   YOLO_MODEL_PATH = "path_to_yolov11l.pt"
   ```

2. **Run the script**:

   ```bash
   python main_video.py
   ```

3. **Controls**:

   * Press `q` to exit video preview early.

---

## Configurable Parameters

* **Lane detection frequency**: change frame interval for API calls

  ```python
  if frame_idx % 5 == 0:  # every 5 frames
  ```
* **Vehicle detection confidence threshold**

  ```python
  confidence_threshold=0.25
  ```
* **Slicing parameters for SAHI**:

  ```python
  slice_height=512
  slice_width=512
  overlap_height_ratio=0.2
  overlap_width_ratio=0.2
  ```
* **Lane colors**: modify `LANE_COLORS` list
* **Mini-map size**: `MAP_WIDTH`, `MAP_HEIGHT`
* **HUD and charts appearance**: colors, font sizes, transparency in `functions.py`

---

## Output

* Processed video with overlayed dashboard saved to `OUTPUT_VIDEO_PATH`
* Real-time preview displayed during processing

---

## Notes

* Ensure YOLOv11-L `.pt` model exists at specified path.
* Roboflow API requires valid API key.
* GPU usage highly recommended for real-time performance.
* Fonts `arialbd.ttf` are optional; defaults used if unavailable.

# main_video.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import io

from functions import overlay_lanes, draw_vehicle_boxes, draw_hud_box, draw_mini_map

# === CONFIG ===
API_URL = "https://serverless.roboflow.com"
API_KEY = "93HY4MRKeUkMy9zXPDWI"
LANE_MODEL_ID = "road-lane-instance-segmentation-y5bzk/1"

VIDEO_PATH = r"c:\Users\PC\Downloads\Adobe Express - YTDown.com_YouTube_U-K-Travels-4-M1-accident-northbound-bet_Media_labBi572p0k_001_1080p (online-video-cutter.com).mp4"
OUTPUT_VIDEO_PATH = r"c:\Users\PC\Downloads\lane_dashboard_video7.mp4"

YOLO_MODEL_PATH = r"C:\Users\PC\yolo11l.pt"  
vehicle_classes = {"car", "truck", "bus", "motorbike", "bicycle"}

LANE_COLORS = [
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 128),   # Purple
]

# === Initialize Lane Model Client ===
CLIENT = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
print("Lane model client initialized.")

# === Initialize YOLOv11-L + SAHI model ===
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=YOLO_MODEL_PATH,
    confidence_threshold=0.25,
    device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
)
print("YOLOv11-L model loaded.")

# === Open video ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))


ret, first_frame = cap.read()
if not ret:
    raise ValueError("Cannot read video")

frame_idx = 1
h, w = first_frame.shape[:2]

lane_result = CLIENT.infer(first_frame, model_id=LANE_MODEL_ID)


cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
lane_id_colors = {}
previous_lane_polygons = []
next_lane_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    h, w = frame.shape[:2]

    MAX_RETRIES = 3  

    if frame_idx % 5 == 0:
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                lane_result = CLIENT.infer(frame, model_id=LANE_MODEL_ID)
                previous_lane_result = lane_result  
                success = True
                break
            except Exception as e:
                print(f"Lane API failed on frame {frame_idx}, attempt {attempt+1}: {e}")
        if not success:
            print(f"Skipping lane inference for frame {frame_idx}, using previous lanes")
            lane_result = previous_lane_result  


    lane_polygons = []
    lane_x_positions = []

    for pred in lane_result.get("predictions", []):
        if "points" in pred and pred["class"].lower() == "road lane":
            pts = np.array([[p["x"], p["y"]] for p in pred["points"]], dtype=int)
            lane_polygons.append(pts)
            lane_x_positions.append(np.min(pts[:, 0]))  

    
    sorted_indices = np.argsort(lane_x_positions)
    lane_polygons_sorted = [lane_polygons[i] for i in sorted_indices]

    
    lane_colors_for_map = [LANE_COLORS[i % len(LANE_COLORS)] for i in range(len(lane_polygons_sorted))]

    
    overlay, lane_areas, lane_widths, lane_confidences, lane_polygons_sorted = overlay_lanes(
        frame, lane_result, lane_colors_for_map,
        fixed_order=True, lane_polygons_fixed=lane_polygons_sorted
    )




    # --- Vehicle detection ---
    sliced_result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    overlay = draw_vehicle_boxes(overlay, sliced_result, vehicle_classes)

    
    output = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    main_img = Image.fromarray(output_rgb)
    
    # --- Compute metrics for charts using fixed lanes ---
    lane_areas_chart, lane_widths_chart = [], []

    for idx, pts in enumerate(lane_polygons_sorted):
        area = cv2.contourArea(pts)
        lane_areas_chart.append(area / (h * w))
        xs = pts[:, 0]
        lane_widths_chart.append(max(xs) - min(xs))
        
        
        
    
    SMOOTHING_FACTOR = 0.15  

    
    if "prev_lane_areas_chart" not in locals():
        prev_lane_areas_chart = lane_areas_chart.copy()
        prev_lane_widths_chart = lane_widths_chart.copy()

    
    min_len = min(len(prev_lane_areas_chart), len(lane_areas_chart))
    if min_len > 0:
    
        lane_areas_chart = [
            prev_lane_areas_chart[i] + SMOOTHING_FACTOR * (lane_areas_chart[i] - prev_lane_areas_chart[i])
            for i in range(min_len)
        ]
        lane_widths_chart = [
            prev_lane_widths_chart[i] + SMOOTHING_FACTOR * (lane_widths_chart[i] - prev_lane_widths_chart[i])
            for i in range(min_len)
        ]

    
    prev_lane_areas_chart = lane_areas_chart.copy()
    prev_lane_widths_chart = lane_widths_chart.copy()

    # === Charts ===
    fig, axes = plt.subplots(2, 1, figsize=(3, 6), facecolor="#282828")
    if lane_areas:
        labels = [f"Lane {i+1}" for i in range(len(lane_areas))]
        base_colors = ["#00FF00", "#FF0000", "#0000FF", "#00FFFF", "#FF00FF", "#FFFF00", "#800080"]
        colors = [base_colors[i % len(base_colors)] for i in range(len(lane_areas))]
        wedges, _ = axes[0].pie(
            lane_areas_chart,  
            labels=None,
            colors=colors,
            wedgeprops=dict(width=0.4, edgecolor="gray", linewidth=1.5)
        )
        r = 0.75
        for i, wedge_patch in enumerate(wedges):
            theta = (wedge_patch.theta2 + wedge_patch.theta1) / 2
            x = r * np.cos(np.deg2rad(theta))
            y = r * np.sin(np.deg2rad(theta))
            c = colors[i].lstrip('#')
            r_c, g_c, b_c = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            luminance = 0.299*r_c + 0.587*g_c + 0.114*b_c
            text_color = 'black' if luminance > 150 else 'white'
            axes[0].text(
                x, y, labels[i],
                ha='center', va='center',
                color=text_color,
                fontsize=8,
                rotation=theta-90,
                rotation_mode='anchor'
            )
    axes[0].set_title("Lane Area %", color="white")
    axes[1].hist(lane_widths_chart, bins=8, color="#33CCFF", edgecolor="gray", linewidth=1.0)
    axes[1].set_title("Lane Width Distribution", color="white")
    axes[1].tick_params(colors="white")
    axes[1].grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    for ax in axes:
        ax.set_facecolor("#282828")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor="#282828")
    plt.close(fig)
    buf.seek(0)
    charts_img = Image.open(buf)

    # === Mini-map ===
    vehicles_map = []
    for obj in sliced_result.object_prediction_list:
        label = obj.category.name
        if label.lower() not in vehicle_classes:
            continue
        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        lane_pos = None
        if lane_areas:
            for i, pred in enumerate(lane_result["predictions"]):
                if "points" not in pred or pred["class"].lower() != "road lane":
                    continue
                pts = np.array([[p["x"], p["y"]] for p in pred["points"]])
                min_x, max_x = np.min(pts[:,0]), np.max(pts[:,0])
                vehicle_x_center = (x1 + x2)/2
                if min_x <= vehicle_x_center <= max_x:
                    lane_pos = i
                    break
        if lane_pos is not None:
            vehicles_map.append((lane_pos, (x1+x2)/2 / w))

    

    
    lane_polygons = lane_polygons_sorted


    vehicles_map_corrected = []
    for obj in sliced_result.object_prediction_list:
        label = obj.category.name.lower()
        if label not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        vehicle_center_x = (x1 + x2) / 2

        assigned_lane = None
        max_overlap = 0

        
        for lane_idx, pts in enumerate(lane_polygons_sorted):
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            overlap = mask[y1:y2, x1:x2].sum()
            if overlap > max_overlap:
                max_overlap = overlap
                assigned_lane = lane_idx

        if assigned_lane is not None:
            x_ratio = vehicle_center_x / w
            y_center = (y1 + y2) / 2
            y_ratio = y_center / h
            vehicles_map_corrected.append((assigned_lane, (x_ratio, y_ratio)))




    # --- Mini-map ---
    MAP_WIDTH, MAP_HEIGHT = 300, 120
    map_img = Image.new("RGBA", (MAP_WIDTH, MAP_HEIGHT), (0, 0, 0, 180))
    map_draw = ImageDraw.Draw(map_img)
    draw_mini_map(
        map_draw,
        lane_polygons_sorted,       
        lane_colors_for_map,        
        vehicles_map_corrected,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT
    )
    # === Side panel ===
    chart_panel_width = 350
    chart_panel_height = main_img.height
    charts_img_small = charts_img.resize((chart_panel_width, chart_panel_height - MAP_HEIGHT - 20))
    panel_img = Image.new("RGBA", (chart_panel_width, chart_panel_height), (40, 40, 40, 255))
    panel_img.paste(charts_img_small, (0, 0))
    map_x = (chart_panel_width - MAP_WIDTH) // 2
    map_y = chart_panel_height - MAP_HEIGHT - 10
    panel_img.paste(map_img, (map_x, map_y), mask=map_img)

    # === Compose final image ===
    final_width = main_img.width + panel_img.width
    final_img = Image.new("RGB", (final_width, main_img.height), (0, 0, 0))
    final_img.paste(main_img, (0, 0))
    final_img.paste(panel_img, (main_img.width, 0))

    draw = ImageDraw.Draw(final_img)

    # === Draw HUD ===
    
    draw_hud_box(draw, (20, 20, 130, 130), symbol="↑", text="Straight")

    total_vehicles = sum(1 for obj in sliced_result.object_prediction_list
                        if obj.category.name.lower() in vehicle_classes)
    try:
        font_big = ImageFont.truetype("arialbd.ttf", 24)
    except:
        font_big = ImageFont.load_default()

    hud_right_x = 140
    hud_top_y = 20
    
    draw.text((hud_right_x, hud_top_y + 30), f"Vehicles: {total_vehicles}", fill="black", font=font_big)

    mean_conf = np.mean(lane_confidences) if lane_confidences else 0.6
    quality_score = int(mean_conf * 100)
    bar_w, bar_h = 250, 35
    x0, y0 = 25, 190
    
    draw.text((x0 + 10, y0 - 35), f"Lane Quality: {quality_score}/100", fill="black", font=font_big)
    draw.rectangle([x0, y0, x0 + bar_w, y0 + bar_h], outline="white", width=3)
    fill_w = int(bar_w * mean_conf)
    draw.rectangle([x0, y0, x0 + fill_w, y0 + bar_h], fill=(0, 255, 0, 150))

    
    final_img_ratio = final_img.width / final_img.height
    video_ratio = frame_width / frame_height

    if final_img_ratio > video_ratio:
    
        new_width = frame_width
        new_height = int(frame_width / final_img_ratio)
    else:
        
        new_height = frame_height
        new_width = int(frame_height * final_img_ratio)

    resized_final = final_img.resize((new_width, new_height), Image.LANCZOS)

    
    final_frame_padded = Image.new("RGB", (frame_width, frame_height), (0, 0, 0))
    x_offset = (frame_width - new_width) // 2
    y_offset = (frame_height - new_height) // 2
    final_frame_padded.paste(resized_final, (x_offset, y_offset))

    
    final_frame = cv2.cvtColor(np.array(final_frame_padded), cv2.COLOR_RGB2BGR)

    
    cv2.imshow("Lane Dashboard", final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(final_frame)
    



cap.release()
out.release()
print(f"✅ Video processed and saved to {OUTPUT_VIDEO_PATH}")

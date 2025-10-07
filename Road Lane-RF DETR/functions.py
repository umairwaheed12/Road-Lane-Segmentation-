# functions.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def overlay_lanes(image, lane_result, lane_colors, fixed_order=False, lane_polygons_fixed=None):
    overlay = image.copy()
    lane_areas, lane_widths, lane_confidences = [], [], []
    lane_polygons = []
    lane_x_positions = []
    h, w = image.shape[:2]


    if "predictions" in lane_result:
        for pred in lane_result["predictions"]:
            if "points" not in pred or pred["class"].lower() != "road lane":
                continue
            pts = np.array([[p["x"], p["y"]] for p in pred["points"]], dtype=np.int32)
            epsilon = 0.01 * cv2.arcLength(pts, True)
            smooth_pts = cv2.approxPolyDP(pts, epsilon, True)

            lane_polygons.append(smooth_pts)
            lane_x_positions.append(np.min(smooth_pts[:, 0, 0]))
            area = cv2.contourArea(smooth_pts)
            lane_areas.append(area / (h * w))
            lane_confidences.append(pred.get("confidence", 0.5))
            xs = [p[0][0] for p in smooth_pts]
            lane_widths.append(max(xs) - min(xs))


    if not fixed_order:
        sorted_indices = np.argsort(lane_x_positions)
        lane_polygons = [lane_polygons[i] for i in sorted_indices]
        lane_areas = [lane_areas[i] for i in sorted_indices]
        lane_widths = [lane_widths[i] for i in sorted_indices]
        lane_confidences = [lane_confidences[i] for i in sorted_indices]
    else:
        if lane_polygons_fixed is not None:
            lane_polygons = lane_polygons_fixed


    for idx, pts in enumerate(lane_polygons):
        color = lane_colors[idx % len(lane_colors)]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

    
    overlay_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_pil)
    try:
        font_lane = ImageFont.truetype("arialbd.ttf", 20)
    except:
        font_lane = ImageFont.load_default()

    bottom_y = h - 35
    spacing = w // max(len(lane_polygons), 1)

    for idx in range(len(lane_polygons)):
        lane_text = f"Lane {idx + 1}"
        text_bbox = draw.textbbox((0, 0), lane_text, font=font_lane)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = spacing * idx + (spacing - text_w) / 2
        draw.text((text_x, bottom_y), lane_text, fill=(255, 255, 255), font=font_lane)

    overlay = np.array(overlay_pil)
    return overlay, lane_areas, lane_widths, lane_confidences, lane_polygons



def draw_vehicle_boxes(overlay, sliced_result, vehicle_classes):
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay_pil)
    try:
        font_vehicle = ImageFont.truetype("arialbd.ttf", 14)
    except:
        font_vehicle = ImageFont.load_default()

    for obj in sliced_result.object_prediction_list:
        label = obj.category.name
        if label.lower() not in vehicle_classes:
            continue
        conf = obj.score.value
        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        color = (0, 255, 255)  # cyan box

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = label.upper()
        bbox_text = draw.textbbox((0, 0), text, font=font_vehicle)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        draw.rectangle([x1, y1 - text_h - 3, x1 + text_w + 3, y1], fill=(0, 0, 0, 180))
        draw.text((x1 + 2, y1 - text_h - 2), text, fill=color, font=font_vehicle)

    overlay = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)
    return overlay


def draw_hud_box(draw, xy, symbol=None, text=None, font_size=36):
    try:
        font_big = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        font_big = ImageFont.load_default()
    draw.rectangle(xy, fill=(60, 60, 60, 255), outline="white", width=2)
    if symbol:
        bbox = draw.textbbox((0, 0), symbol, font=font_big)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        sx = xy[0] + (xy[2] - xy[0] - w) / 2
        sy = xy[1] + 10
        draw.text((sx, sy), symbol, fill="white", font=font_big)
    if text:
        try:
            font_small = ImageFont.truetype("arialbd.ttf", 20)
        except:
            font_small = ImageFont.load_default()
        tx = xy[0] + (xy[2] - xy[0]) / 2 - 35
        ty = xy[3] - 40
        draw.text((tx, ty), text, fill="white", font=font_small)

def draw_mini_map(draw, lane_polygons, lane_colors, vehicles_map, map_width=300, map_height=120):
    """
    Draws a mini-map of lanes and vehicles with lane geometry scaled from actual segmented lanes.
    """
    if not lane_polygons:
        return

    
    all_pts = np.vstack(lane_polygons)
    x_min, y_min = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
    x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])

    lane_width = max(1, x_max - x_min)
    lane_height = max(1, y_max - y_min)

    
    scale_x = map_width / lane_width
    scale_y = map_height / lane_height

    def transform_point(p):
        """Scale and flip Y-axis so lanes appear upright."""
        x = (p[0] - x_min) * scale_x
        y = map_height - (p[1] - y_min) * scale_y
        return (x, y)


    for idx, pts in enumerate(lane_polygons):
        color = tuple(lane_colors[idx % len(lane_colors)])
        scaled_pts = [transform_point(p) for p in pts]
        draw.polygon(scaled_pts, outline=color, fill=(color[0], color[1], color[2], 60))

    
    for lane_idx, vehicle_data in enumerate(vehicles_map):
        lane_id, (x_ratio, y_ratio) = vehicle_data
        if lane_id >= len(lane_polygons):
            continue

        
        vx = (x_ratio * (x_max - x_min)) * scale_x
        vy = map_height - (y_ratio * (y_max - y_min)) * scale_y

        
        color = lane_colors[lane_id % len(lane_colors)]
        r = 5
        draw.ellipse(
            (vx - r, vy - r, vx + r, vy + r),
            fill=(color[0], color[1], color[2], 255),
            outline="white"
        )

    # --- Optional: border ---
    draw.rectangle([0, 0, map_width - 1, map_height - 1], outline=(255, 255, 255, 150))
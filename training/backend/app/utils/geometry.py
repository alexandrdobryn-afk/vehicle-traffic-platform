def tile_stride(tile_size: int, overlap: float) -> int:
    if overlap < 1:
        return max(1, int(tile_size * (1 - overlap)))
    return max(1, tile_size - int(overlap))


def tile_origins(width: int, height: int, tile_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    origins: list[tuple[int, int, int, int]] = []
    for top in sorted(set(ys)):
        for left in sorted(set(xs)):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            if right > left and bottom > top:
                origins.append((left, top, right, bottom))
    return origins


def annotation_to_abs_box(ann, width: int, height: int):
    if None in (ann.x_center, ann.y_center, ann.bbox_width, ann.bbox_height):
        return None
    box_w = float(ann.bbox_width) * width
    box_h = float(ann.bbox_height) * height
    cx = float(ann.x_center) * width
    cy = float(ann.y_center) * height
    return ann, (cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2)


def clip_box_to_tile(box, tile, min_visibility: float):
    x1, y1, x2, y2 = box
    left, top, right, bottom = tile
    ix1 = max(x1, left)
    iy1 = max(y1, top)
    ix2 = min(x2, right)
    iy2 = min(y2, bottom)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    box_area = max((x2 - x1) * (y2 - y1), 1.0)
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    if inter_area / box_area < min_visibility:
        return None
    return ix1, iy1, ix2, iy2


def norm_box_to_abs(cx, cy, bw, bh, width: int, height: int):
    if None in (cx, cy, bw, bh):
        return None
    box_w = float(bw) * width
    box_h = float(bh) * height
    center_x = float(cx) * width
    center_y = float(cy) * height
    return [center_x - box_w / 2, center_y - box_h / 2, center_x + box_w / 2, center_y + box_h / 2]


def abs_box_to_norm(box, width: int, height: int) -> dict:
    x1, y1, x2, y2 = box
    return {
        "x_center": ((x1 + x2) / 2) / width,
        "y_center": ((y1 + y2) / 2) / height,
        "width": max(x2 - x1, 0) / width,
        "height": max(y2 - y1, 0) / height,
    }


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 0.0)
    area_b = max((bx2 - bx1) * (by2 - by1), 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

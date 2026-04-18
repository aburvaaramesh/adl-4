import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def _extract_kart_name_map(info: dict) -> dict[int, str]:
    """Best-effort extraction of track_id -> kart_name mapping from an info.json dict."""
    name_map: dict[int, str] = {}

    candidate_keys = [
        "karts",
        "kart_info",
        "kart_infos",
        "players",
        "player_info",
        "racers",
    ]

    def maybe_add(entry: dict):
        if not isinstance(entry, dict):
            return
        kart_name = entry.get("kart_name") or entry.get("name") or entry.get("kart")
        kart_id = entry.get("track_id")
        if kart_id is None:
            kart_id = entry.get("instance_id")
        if kart_id is None:
            kart_id = entry.get("id")
        if kart_name is None or kart_id is None:
            return
        try:
            name_map[int(kart_id)] = str(kart_name)
        except (TypeError, ValueError):
            return

    for key in candidate_keys:
        value = info.get(key)
        if isinstance(value, list):
            for item in value:
                maybe_add(item)
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, dict):
                    if "id" not in v:
                        v = {"id": k, **v}
                    maybe_add(v)
                elif isinstance(v, str):
                    try:
                        name_map[int(k)] = v
                    except (TypeError, ValueError):
                        continue

    return name_map


def _relative_position(ego_center: tuple[float, float], other_center: tuple[float, float]) -> tuple[str, str, str]:
    dx = other_center[0] - ego_center[0]
    dy = other_center[1] - ego_center[1]

    left_right = "left" if dx < 0 else "right"
    front_behind = "front" if dy < 0 else "behind"

    if abs(dx) >= abs(dy):
        coarse = left_right
    else:
        coarse = "in front of" if front_behind == "front" else "behind"

    return left_right, front_behind, coarse


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path) as f:
        info = json.load(f)

    detections = info.get("detections", [])
    if view_index < 0 or view_index >= len(detections):
        return []

    frame_detections = detections[view_index]
    kart_name_map = _extract_kart_name_map(info)

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    karts = []
    for detection in frame_detections:
        if len(detection) < 6:
            continue

        class_id, track_id, x1, y1, x2, y2 = detection[:6]
        class_id = int(class_id)
        track_id = int(track_id)
        if class_id != 1:
            continue

        x1_scaled = float(x1) * scale_x
        y1_scaled = float(y1) * scale_y
        x2_scaled = float(x2) * scale_x
        y2_scaled = float(y2) * scale_y

        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2

        if center_x < 0 or center_x >= img_width or center_y < 0 or center_y >= img_height:
            continue

        kart_name = kart_name_map.get(track_id, "ego" if track_id == 0 else f"kart_{track_id}")

        karts.append(
            {
                "instance_id": track_id,
                "kart_name": kart_name,
                "center": (center_x, center_y),
                "is_center_kart": False,
            }
        )

    if not karts:
        return []

    image_center = (img_width / 2, img_height / 2)

    # Prefer id=0 (ego) when available; otherwise use geometric center heuristic.
    ego_idx = next((i for i, k in enumerate(karts) if int(k["instance_id"]) == 0), None)
    if ego_idx is None:
        ego_idx = int(
            np.argmin(
                [
                    (k["center"][0] - image_center[0]) ** 2 + (k["center"][1] - image_center[1]) ** 2
                    for k in karts
                ]
            )
        )

    karts[ego_idx]["is_center_kart"] = True
    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        info = json.load(f)

    direct_candidates = [
        "track",
        "track_name",
        "track_id",
        "map_name",
        "level_name",
        "arena",
    ]

    for key in direct_candidates:
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_key in ["name", "track_name", "id"]:
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()

    metadata = info.get("metadata")
    if isinstance(metadata, dict):
        for key in direct_candidates:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return "unknown"


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    karts = extract_kart_objects(info_path, view_index, img_width=img_width, img_height=img_height)
    if not karts:
        return []

    ego = next((k for k in karts if k["is_center_kart"]), karts[0])
    ego_name = str(ego["kart_name"])

    qa_pairs = [
        {"question": "What kart is the ego car?", "answer": ego_name},
        {"question": "How many karts are there in the scenario?", "answer": str(len(karts))},
        {"question": "What track is this?", "answer": extract_track_info(info_path)},
    ]

    others = [k for k in karts if int(k["instance_id"]) != int(ego["instance_id"])]

    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0

    for kart in others:
        kart_name = str(kart["kart_name"])
        left_right, front_behind, coarse = _relative_position(ego["center"], kart["center"])

        if left_right == "left":
            left_count += 1
        else:
            right_count += 1
        if front_behind == "front":
            front_count += 1
        else:
            behind_count += 1

        qa_pairs.append(
            {
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": left_right,
            }
        )
        qa_pairs.append(
            {
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": front_behind,
            }
        )
        qa_pairs.append(
            {
                "question": f"Where is {kart_name} relative to the ego car?",
                "answer": coarse,
            }
        )

    qa_pairs.extend(
        [
            {"question": "How many karts are to the left of the ego car?", "answer": str(left_count)},
            {"question": "How many karts are to the right of the ego car?", "answer": str(right_count)},
            {"question": "How many karts are in front of the ego car?", "answer": str(front_count)},
            {"question": "How many karts are behind the ego car?", "answer": str(behind_count)},
        ]
    )

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate(
    data_dir: str = "data/train",
    output_file: str | None = None,
    img_width: int = 150,
    img_height: int = 100,
):
    """
    Generate QA pairs for all views in a split directory and write a single json file.

    Args:
        data_dir: Directory containing *_info.json and corresponding *_im.jpg files.
        output_file: Output json path. Defaults to <data_dir>/balanced_qa_pairs.json.
        img_width: Width used for coordinate scaling.
        img_height: Height used for coordinate scaling.
    """
    split_dir = Path(data_dir)
    if output_file is None:
        output_path = split_dir / "balanced_qa_pairs.json"
    else:
        output_path = Path(output_file)

    info_files = sorted(split_dir.glob("*_info.json"))
    all_pairs: list[dict[str, str]] = []

    split_name = split_dir.name

    for info_path in info_files:
        with open(info_path) as f:
            info = json.load(f)

        num_views = len(info.get("detections", []))
        base_name = info_path.stem.replace("_info", "")

        for view_index in range(num_views):
            image_name = f"{base_name}_{view_index:02d}_im.jpg"
            image_path = split_dir / image_name

            if not image_path.exists():
                continue

            qa_pairs = generate_qa_pairs(str(info_path), view_index, img_width=img_width, img_height=img_height)
            image_file = f"{split_name}/{image_name}"

            for qa in qa_pairs:
                all_pairs.append(
                    {
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "image_file": image_file,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"Wrote {len(all_pairs)} QA pairs to {output_path}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate": generate})


if __name__ == "__main__":
    main()

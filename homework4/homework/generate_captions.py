from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    karts = extract_kart_objects(info_path, view_index, img_width=img_width, img_height=img_height)
    if not karts:
        return []

    ego = next((k for k in karts if k["is_center_kart"]), karts[0])
    ego_id = int(ego["instance_id"])
    ego_name = str(ego["kart_name"])

    captions = [
        f"{ego_name} is the ego car.",
        f"There are {len(karts)} karts in the scene.",
        f"The track is {extract_track_info(info_path)}.",
    ]

    for kart in karts:
        if int(kart["instance_id"]) == ego_id:
            continue

        kart_name = str(kart["kart_name"])
        dx = kart["center"][0] - ego["center"][0]
        dy = kart["center"][1] - ego["center"][1]

        if abs(dx) >= abs(dy):
            position = "left" if dx < 0 else "right"
            captions.append(f"{kart_name} is {position} of the ego car.")
        else:
            position = "in front" if dy < 0 else "behind"
            captions.append(f"{kart_name} is {position} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate(
    data_dir: str = "data/train",
    output_file: str | None = None,
    img_width: int = 150,
    img_height: int = 100,
):
    """
    Generate captions for all views in a split directory and write a single json file.

    Args:
        data_dir: Directory containing *_info.json and corresponding *_im.jpg files.
        output_file: Output json path. Defaults to <data_dir>/all_captions.json.
        img_width: Width used for coordinate scaling.
        img_height: Height used for coordinate scaling.
    """
    split_dir = Path(data_dir)
    if output_file is None:
        output_path = split_dir / "all_captions.json"
    else:
        output_path = Path(output_file)

    info_files = sorted(split_dir.glob("*_info.json"))
    all_captions: list[dict[str, str]] = []

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

            captions = generate_caption(str(info_path), view_index, img_width=img_width, img_height=img_height)
            image_file = f"{split_name}/{image_name}"

            for caption in captions:
                all_captions.append(
                    {
                        "caption": caption,
                        "image_file": image_file,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Wrote {len(all_captions)} captions to {output_path}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate})


if __name__ == "__main__":
    main()

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
    kart_names = ", ".join(str(k["kart_name"]) for k in karts)

    num_karts = len(karts)
    kart_word = "kart" if num_karts == 1 else "karts"
    kart_verb = "is" if num_karts == 1 else "are"

    captions = [
        f"{ego_name} is the ego car.",
        f"There {kart_verb} {num_karts} {kart_word} in the scene.",
        f"The {kart_word} in the scene {kart_verb} {kart_names}.",
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
            if position == "behind":
                captions.append(f"{kart_name} is {position} the ego car.")
            else:
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


def validate(
    data_dir: str = "data/valid",
    ground_truth_file: str | None = None,
    max_mismatches: int = 30,
    img_width: int = 150,
    img_height: int = 100,
):
    """
    Validate generated captions.

    Modes:
    - Sanity mode (default): checks for empty captions and common grammar issues.
    - Exact mode (when ground_truth_file is provided): compares generated captions to ground truth.

    Args:
        data_dir: Directory containing *_info.json and corresponding *_im.jpg files.
        ground_truth_file: Optional captions JSON for exact comparison.
        max_mismatches: Maximum number of mismatches to print in exact mode.
        img_width: Width used for coordinate scaling.
        img_height: Height used for coordinate scaling.
    """
    split_dir = Path(data_dir)
    info_files = sorted(split_dir.glob("*_info.json"))
    if not info_files:
        raise FileNotFoundError(f"No *_info.json files found in {split_dir}")

    split_name = split_dir.name
    generated_pairs: set[tuple[str, str]] = set()

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

            image_file = f"{split_name}/{image_name}"
            captions = generate_caption(str(info_path), view_index, img_width=img_width, img_height=img_height)
            for caption in captions:
                generated_pairs.add((image_file, caption))

    if not generated_pairs:
        raise RuntimeError("No captions were generated. Check that images exist for the split.")

    if ground_truth_file is None:
        captions_only = [caption for _, caption in generated_pairs]
        empty_count = sum(1 for c in captions_only if not c.strip())
        none_count = sum(1 for c in captions_only if "None" in c)
        bad_grammar = sum(
            1
            for c in captions_only
            if "There are 1 karts" in c
            or "There is 1 karts" in c
            or "There are 1 kart" in c
            or "behind of the ego car" in c
        )

        print("\nCaption sanity validation")
        print(f"Generated caption pairs: {len(generated_pairs)}")
        print(f"Empty captions: {empty_count}")
        print(f"Captions containing 'None': {none_count}")
        print(f"Captions with known grammar issues: {bad_grammar}")

        if empty_count == 0 and none_count == 0 and bad_grammar == 0:
            print("Result: PASS")
        else:
            print("Result: CHECK FAILED")
        return

    with open(ground_truth_file) as f:
        ground_truth = json.load(f)

    expected_pairs: set[tuple[str, str]] = set()
    for item in ground_truth:
        expected_pairs.add((item["image_file"], item["caption"]))

    matched_pairs = generated_pairs.intersection(expected_pairs)
    missing_pairs = sorted(expected_pairs.difference(generated_pairs))

    total = len(expected_pairs)
    matched = len(matched_pairs)

    print(f"\nCaption exact validation: {matched}/{total} match ({100 * matched / total:.1f}%)")

    if missing_pairs:
        print(f"\nMismatches ({len(missing_pairs)} total, showing up to {max_mismatches}):")
        for image_file, caption in missing_pairs[:max_mismatches]:
            print(f"  {image_file}")
            print(f"    expected caption: '{caption}'")
    else:
        print("\nAll match!")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate, "validate": validate})


if __name__ == "__main__":
    main()

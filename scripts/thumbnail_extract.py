from pathlib import Path
import argparse
import openslide
from PIL import Image

# Common WSI extensions
WSI_EXTENSIONS = {
    ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"
}


def save_thumbnails(
    input_dir: str,
    output_dir: str,
    thumbnail_size: int = 1024,
    recursive: bool = False,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if recursive:
        slide_paths = [
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in WSI_EXTENSIONS
        ]
    else:
        slide_paths = [
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in WSI_EXTENSIONS
        ]

    if not slide_paths:
        print(f"No WSIs found in: {input_path}")
        return

    print(f"Found {len(slide_paths)} WSI files.")

    for slide_path in slide_paths:
        try:
            slide = openslide.OpenSlide(str(slide_path))

            width, height = slide.dimensions

            # Preserve aspect ratio
            if width >= height:
                thumb_w = thumbnail_size
                thumb_h = int((height / width) * thumbnail_size)
            else:
                thumb_h = thumbnail_size
                thumb_w = int((width / height) * thumbnail_size)

            thumbnail = slide.get_thumbnail((thumb_w, thumb_h))

            # Ensure RGB before saving
            if thumbnail.mode != "RGB":
                thumbnail = thumbnail.convert("RGB")

            save_path = output_path / f"{slide_path.stem}_thumbnail.png"
            thumbnail.save(save_path)

            print(f"Saved: {save_path}")

        except Exception as e:
            print(f"Failed on {slide_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate thumbnails for all WSIs in a directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing WSI files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save thumbnails."
    )
    parser.add_argument(
        "--thumbnail_size",
        type=int,
        default=1024,
        help="Max size of the thumbnail's longer side."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories."
    )

    args = parser.parse_args()

    save_thumbnails(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        thumbnail_size=args.thumbnail_size,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
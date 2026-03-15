"""run.py - Run the Go board detection pipeline over images/*.jpg."""

import glob
import os
import cv2

from board import detect_board
from grid import detect_grid_lines
from visualize import annotate_original, annotate_grid

OUTPUT_DIR = "out"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = sorted(glob.glob("images/*.jpg"))

    for image_path in image_paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ERROR: could not read {image_path}")
            continue

        result = detect_board(img)
        h_lines, v_lines = detect_grid_lines(result.warped)

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{name}_original.jpg"),
            annotate_original(img, result.mask, result.corners),
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{name}_warped.jpg"),
            result.warped,
        )
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{name}_grid.jpg"),
            annotate_grid(result.warped, h_lines, v_lines),
        )

        print(f"{name}: done")


if __name__ == "__main__":
    main()

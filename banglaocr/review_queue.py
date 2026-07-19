"""
Prints the human-review queue: every OCR line where confidence was low or
the two engines disagreed. Use this to spot-check accuracy and manually
correct the worst lines rather than trusting a blanket auto-correction.
"""
import sys

import storage


def main(db_path: str):
    queue = storage.get_review_queue(db_path)
    if not queue:
        print("Review queue is empty.")
        return

    print(f"{len(queue)} lines need review:\n")
    for item in queue:
        print(
            f"[page {item['page_id']}] col{item['column_index']}/line{item['line_index']} "
            f"({item['engine']}, conf={item['confidence']:.1f}): {item['text']}"
        )


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "ocr.db"
    main(db)

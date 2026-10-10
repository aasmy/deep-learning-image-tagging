from pathlib import Path

DATA_DIR = Path("data/raw/NUS-WIDE")

database_images = set(
    (DATA_DIR / "database_img.txt").read_text(
        encoding="utf-8"
    ).splitlines()
)

test_images = set(
    (DATA_DIR / "test_img.txt").read_text(
        encoding="utf-8"
    ).splitlines()
)

overlap = database_images & test_images

print("Database images:", len(database_images))
print("Test images:", len(test_images))
print("Overlapping images:", len(overlap))
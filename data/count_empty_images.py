import os

def find_empty_png(root_dir, report_every=10000):
    count = 0
    empty_count = 0

    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".png"):
                path = os.path.join(root, name)
                count += 1

                # 进度输出
                if count % report_every == 0:
                    print(f"[Progress] Checked {count} PNG files, found {empty_count} empty")

                try:
                    if os.path.getsize(path) == 0:
                        print(f"[EMPTY] {path}")
                        empty_count += 1
                except OSError as e:
                    print(f"[ERROR] {path}: {e}")

    print(f"\nDone. Total checked: {count}, empty files: {empty_count}")

if __name__ == "__main__":
    root = "/root/projects/dm-cad/datasets/dataset_v0/cad_img"
    find_empty_png(root)

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

src_root = "/root/projects/dm-cad/datasets/dataset_v0/cad_img"
dst_root = "/root/projects/dm-cad/datasets/dataset_v2/cad_img"

MAX_WORKERS = 8
THREAD_PER_7Z = 8

TARGET_DIRS = {"0002", "0003", "0004", "0005"}

os.makedirs(dst_root, exist_ok=True)


def compress(name):
    src_path = os.path.join(src_root, name)

    if not os.path.isdir(src_path):
        return f"Skip(non-dir): {name}"

    dst_zip = os.path.join(dst_root, f"{name}.zip")

    # ✅ 覆盖逻辑：先删
    if os.path.exists(dst_zip):
        try:
            os.remove(dst_zip)
            status = "Overwrite"
        except Exception as e:
            return f"Error(remove): {name} -> {e}"
    else:
        status = "Create"

    cmd = [
        "7z", "a",
        f"-mmt={THREAD_PER_7Z}",
        dst_zip,
        src_path
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return f"{status}: {name}"
    except subprocess.CalledProcessError:
        return f"Error(compress): {name}"


def main():
    names = sorted(TARGET_DIRS)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(compress, name) for name in names]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()

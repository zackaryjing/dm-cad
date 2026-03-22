import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

src_root = "/root/projects/dm-cad/datasets/dataset_v0/cad_img"
dst_root = "/root/projects/dm-cad/datasets/dataset_v2/cad_img"

MAX_WORKERS = 24   # 你可以改成 16~32
THREAD_PER_7Z = 2  # -mmt=2

os.makedirs(dst_root, exist_ok=True)


def compress(name):
    src_path = os.path.join(src_root, name)

    # 只处理目录
    if not os.path.isdir(src_path):
        return f"Skip(non-dir): {name}"

    dst_zip = os.path.join(dst_root, f"{name}.zip")

    # 跳过已有
    if os.path.exists(dst_zip):
        return f"Skip(exists): {name}"

    cmd = [
        "7z", "a",
        f"-mmt={THREAD_PER_7Z}",
        dst_zip,
        src_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Done: {name}"
    except subprocess.CalledProcessError:
        return f"Error: {name}"


def main():
    names = sorted(os.listdir(src_root))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(compress, name) for name in names]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()

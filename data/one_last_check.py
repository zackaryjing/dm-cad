import os

TARGET_DIRS = [
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00045387_00005",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00045720_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00045667_00004",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00046024_00004",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00046042_00006",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00045643_00004",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00045474_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00049690_00002",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00046042_00003",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00046053_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00045790_00022",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00045619_00017",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00049702_00005",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0004/00045353_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0003/00035011_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0003/00035055_00003",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0003/00034990_00005",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0003/00035419_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0002/00026320_00013",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0002/00026351_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0002/00025910_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0002/00029046_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0002/00026030_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0002/00026356_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0002/00029161_00004",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0005/00051319_00003",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0005/00050954_00001",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0005/00050994_00004",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0005/00051322_00003",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0005/00051335_00002",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0005/00051283_00003",
    "/root/projects/dm-cad/datasets/dataset_v0/cad_img/0005/00051324_00003",
]


def check_one_dir(d):
    if not os.path.exists(d):
        return f"[MISSING DIR] {d}"

    files = [f for f in os.listdir(d) if f.endswith(".png")]

    if len(files) != 8:
        return f"[COUNT ERROR] {d} -> {len(files)} files"

    for f in files:
        p = os.path.join(d, f)
        try:
            if os.path.getsize(p) == 0:
                return f"[EMPTY] {p}"
        except OSError as e:
            return f"[ERROR] {p}: {e}"

    return f"[OK] {d}"


def main():
    ok = 0
    fail = 0

    for d in TARGET_DIRS:
        res = check_one_dir(d)
        print(res)

        if res.startswith("[OK]"):
            ok += 1
        else:
            fail += 1

    print("\nSummary:")
    print(f"OK: {ok}")
    print(f"Fail: {fail}")


if __name__ == "__main__":
    main()

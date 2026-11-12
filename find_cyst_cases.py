
import os
import nibabel as nib
import numpy as np

ORIGINAL_DATA_DIR = (
    "/mnt/d/Brian Lala/Research/"
    "2-step-segmentation-kidney-tumor-and-cyst/data/KiTS23"
)

MAX_CASES = 10


def main():
    cyst_cases = []

    case_dirs = sorted(
        name for name in os.listdir(ORIGINAL_DATA_DIR)
        if name.startswith("case_")
        and os.path.isdir(os.path.join(ORIGINAL_DATA_DIR, name))
    )

    print(f"Scanning {len(case_dirs)} KiTS23 cases for cyst annotations...")

    for case_id in case_dirs:
        seg_path = os.path.join(
            ORIGINAL_DATA_DIR, case_id, "segmentation.nii.gz"
        )

        if not os.path.exists(seg_path):
            continue

        seg = nib.load(seg_path).dataobj
        unique, counts = np.unique(np.asarray(seg), return_counts=True)

        distribution = dict(zip(unique.tolist(), counts.tolist()))

        cyst_voxels = distribution.get(3, 0)
        tumor_voxels = distribution.get(2, 0)

        if cyst_voxels > 0:
            cyst_cases.append({
                "case": case_id,
                "cyst_voxels": cyst_voxels,
                "tumor_voxels": tumor_voxels,
            })

            print(
                f"{case_id}: "
                f"cyst={cyst_voxels:,}, "
                f"tumor={tumor_voxels:,}"
            )

    print("\n===== SUMMARY =====")
    print(f"Cases containing cysts: {len(cyst_cases)}")

    print(f"\nFirst {MAX_CASES} candidates:")
    for item in cyst_cases[:MAX_CASES]:
        print(
            f"{item['case']} | "
            f"cyst={item['cyst_voxels']:,} | "
            f"tumor={item['tumor_voxels']:,}"
        )


if __name__ == "__main__":
    main()

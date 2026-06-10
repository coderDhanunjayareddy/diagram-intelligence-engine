import shutil
import os

artifact_dir = r"C:\Users\DHANUNJAYA SOMIREDDY\.gemini\antigravity\brain\526bf6ed-0c27-4132-89b3-98573c19bac3"
poc_dir = r"c:\Work\PPT Generation application\backend\storage\poc_outputs"

files_to_copy = [
    ("standard_digestive_system_(v2_validation)_comparison.png", "standard_digestive_system_v2_comparison.png"),
    ("high-res_3d_labeled_digestive_system_(upload)_comparison.png", "high_res_3d_labeled_digestive_system_comparison.png")
]

for src_name, dest_name in files_to_copy:
    src = os.path.join(poc_dir, src_name)
    dest = os.path.join(artifact_dir, dest_name)
    if os.path.exists(src):
        shutil.copy(src, dest)
        print(f"Copied {src_name} -> {dest_name}")
    else:
        print(f"File not found: {src}")

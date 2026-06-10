import os
import glob

downloads_dir = r"C:\Users\DHANUNJAYA SOMIREDDY\Downloads"
print("Listing files in Downloads matching pattern:")
for f in glob.glob(os.path.join(downloads_dir, "*")):
    basename = os.path.basename(f)
    if "diagram" in basename.lower() or "design" in basename.lower() or "stomach" in basename.lower() or "liver" in basename.lower() or basename.endswith((".png", ".jpg", ".pptx")):
        print(f" - {basename} ({os.path.getsize(f)} bytes)")

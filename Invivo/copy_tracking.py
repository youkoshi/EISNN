import os
import shutil

def copy_tracking_dirs(source_root, target_root):
    for root, dirs, files in os.walk(source_root):
        if 'Tracking' in dirs:
            rel_path = os.path.relpath(root, source_root)
            source_tracking = os.path.join(root, 'Tracking')
            target_dir = os.path.join(target_root, rel_path, 'Tracking')
            os.makedirs(target_dir, exist_ok=True)
            shutil.copytree(source_tracking, target_dir, dirs_exist_ok=True)
            print(f"Copied: {source_tracking} -> {target_dir}")


keyname = 'S6175'
source_directory = f"//tsclient/E/inVivo/{keyname}"  # 替换为你的源目录路径
target_directory = f"./{keyname}"  # 替换为你的目标目录路径

copy_tracking_dirs(source_directory, target_directory)
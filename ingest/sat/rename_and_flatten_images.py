import os
import shutil
from datetime import datetime,timedelta

# 输入输出路径
input_base = "output"
output_dir = "images"

            
os.makedirs(output_dir, exist_ok=True)

# 遍历所有天（001 ~ 365）和小时（00 ~ 23）
for day_str in os.listdir(input_base):
    day_path = os.path.join(input_base, day_str)
    if not os.path.isdir(day_path):
        continue

    for hour_str in os.listdir(day_path):
        hour_path = os.path.join(day_path, hour_str)
        if not os.path.isdir(hour_path):
            continue

        for filename in os.listdir(hour_path):
            if filename.endswith(".tif"):
                site = filename.replace(".tif", "")
                # 构造重命名后的文件名
                doy = int(day_str)
                hour = int(hour_str)
                date = datetime(2022, 1, 1) + timedelta(days=doy - 1)
                new_filename = f"{site}_{date.strftime('%Y%m%d')}_{hour:02d}.tif"
                src = os.path.join(hour_path, filename)
                dst = os.path.join(output_dir, new_filename)
                shutil.copyfile(src, dst)

print("✅ rename done")

import os
import csv

def make_csv(root):
    # 为 train 和 val 生成 CSV
    for split in ['train', 'val']:
        img_dir = os.path.join(root, split, 'images')
        lab_dir = os.path.join(root, split, 'labels')
        csv_path = os.path.join(root, f'{split}_list.csv')
        
        if not os.path.isdir(img_dir):
            continue
            
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 遍历图片，匹配对应的 npy 标签
            for img in sorted(os.listdir(img_dir)):
                if img.endswith('.png'):
                    stem = os.path.splitext(img)[0]
                    lab = stem + '.npy'
                    lab_path = os.path.join(lab_dir, lab)
                    
                    if os.path.exists(lab_path):
                        # 写入绝对路径，避免路径解析错误
                        writer.writerow([
                            os.path.abspath(os.path.join(img_dir, img)),
                            os.path.abspath(lab_path)
                        ])
        print(f"✅ 生成列表: {csv_path} (共 {len(os.listdir(img_dir))} 张图)")

# 为两个数据集生成
make_csv('./dataset_cellvitpp/consep')
make_csv('./dataset_cellvitpp/lizard')
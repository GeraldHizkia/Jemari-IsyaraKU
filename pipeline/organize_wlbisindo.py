"""
Script Pengorganisir Dataset WL-BISINDO (Fixed & Enhanced)
Memperbaiki bug 'range(10)' pada repositori orisinal agar mendukung seluruh 32 kelas
secara dinamis dari metadata SD (Signer-Dependent) maupun SI (Signer-Independent).

Penggunaan:
    python organize_wlbisindo.py --metadata ../reference-wlbisindo/data_structuring/SD_split_metadata.json --source /path/to/raw_videos --output ./dataset/SD
"""

import os
import json
import shutil
import argparse
from pathlib import Path

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def organize_dataset(metadata_path, source_dir, output_dir, copy_mode="copy"):
    print(f"[*] Membaca metadata dari: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Dapatkan seluruh label unik yang ada di metadata (dinamis, bukan range(10) statis)
    labels = sorted({int(item["label"]) for item in metadata})
    splits = sorted({instance["split"] for item in metadata for instance in item["instances"]})

    print(f"[+] Ditemukan {len(labels)} kelas unik: {min(labels)}..{max(labels)}")
    print(f"[+] Tipe split: {splits}")

    # Buat direktori tujuan secara aman
    for split in splits:
        for label in labels:
            ensure_dir(os.path.join(output_dir, split, str(label)))

    success_count = 0
    missing_count = 0

    for class_info in metadata:
        label = class_info['label']
        gloss = class_info.get('gloss', f'class_{label}')
        for instance in class_info['instances']:
            video_id = f"{instance['video_id']}.mp4"
            split = instance['split']

            src_path = os.path.join(source_dir, video_id)
            dest_path = os.path.join(output_dir, split, str(label), video_id)

            if os.path.exists(src_path):
                if copy_mode == "copy":
                    shutil.copy2(src_path, dest_path)
                elif copy_mode == "move":
                    shutil.move(src_path, dest_path)
                elif copy_mode == "link":
                    try:
                        os.link(src_path, dest_path)
                    except OSError:
                        shutil.copy2(src_path, dest_path)
                success_count += 1
            else:
                missing_count += 1

    print(f"\n[✓] Selesai:")
    print(f"    - Berhasil diproses: {success_count} file")
    print(f"    - File tidak ditemukan: {missing_count} file")
    print(f"    - Direktori output: {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Organize WL-BISINDO dataset dynamically for all 32 classes")
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path ke metadata JSON (SD_split_metadata.json atau SI_split_metadata.json)')
    parser.add_argument('--source', type=str, required=True,
                        help='Direktori file video mentah (.mp4) WL-BISINDO')
    parser.add_argument('--output', type=str, default='./dataset',
                        help='Direktori hasil pengorganisasian dataset')
    parser.add_argument('--mode', choices=['copy', 'move', 'link'], default='copy',
                        help='Metode pemindahan file: copy, move, atau link (hardlink hemat disk)')

    args = parser.parse_args()
    organize_dataset(args.metadata, args.source, args.output, args.mode)

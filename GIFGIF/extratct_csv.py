import requests
import gzip
import shutil
import os
from tqdm import tqdm

# Download the correct dataset
url = "https://s3-eu-west-1.amazonaws.com/lum-public/gifgif-dataset-20150121-v1.csv.gz"
output_gz = "gifgif-dataset-20150121-v1.csv.gz"
output_csv = "gifgif-dataset-20150121-v1.csv"

# Download with progress bar
print("Downloading...")
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(output_gz, 'wb') as f:
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_gz) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

print("Download complete!")
print(f"Downloaded file size: {os.path.getsize(output_gz) / (1024*1024):.2f} MB")

# Decompress with progress
print("\nDecompressing...")
with gzip.open(output_gz, 'rb') as f_in:
    with open(output_csv, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        
print("Done! CSV file created.")
print(f"Decompressed file size: {os.path.getsize(output_csv) / (1024*1024):.2f} MB")

# Quick preview
print("\nPreview of data:")
with open(output_csv, 'r', encoding='utf-8') as f:
    for i in range(5):
        print(f.readline().strip())
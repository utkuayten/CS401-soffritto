import pandas as pd
import glob

# 1️⃣ define mapping
rename_map = {
    'H3K27ac': 'H3K27ac',
    'H3K27me3': 'H3K27me3',
    'H3K36me3': 'H3K36me3',
    'H3K4me1': 'H3K4me1',
    'H3K4me3': 'H3K4me3',
    'H3K9me3': 'H3K9me3',
    'GC content': 'GC_content',
    'gene density': 'gene_density',
    '2-stage': '2-stage'
}

# 2️⃣ loop through all *_genomic.csv files in the current folder
for path in glob.glob("*_genomic.csv"):
    print(f"Processing {path} ...")
    df = pd.read_csv(path)
    df = df.rename(columns=rename_map)
    
    # optional: check result
    print("✅ Columns renamed:", df.columns[:15].to_list(), "...")
    
    df.to_csv(path, index=False)
    print(f"💾 Saved -> {path}\n")

print("🎉 All files processed successfully.")

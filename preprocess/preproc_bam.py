import pysam
import pandas as pd

bam_file = "preprocess/4DNFIRSSCVBK.bam"

data = []
with pysam.AlignmentFile(bam_file, "rb") as bam:
    for read in bam:
        data.append([
            read.query_name,
            read.reference_name,
            read.reference_start,
            read.reference_end,
            read.mapping_quality,
            read.cigarstring
        ])

# Create DataFrame
df = pd.DataFrame(data, columns=["Read Name", "Reference", "Start", "End", "Mapping Quality", "Cigar String"])

df.to_csv("preprocess/alignment_data.csv", index=False)
import HTSeq

bam_file = "4DNFIRSSCVBK.bam"

for read in HTSeq.BAM_Reader(bam_file):
    print(f"Read: {read.read.name}, Aligned to: {read.iv.chrom}:{read.iv.start}-{read.iv.end}")

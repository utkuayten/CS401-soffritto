import pyBigWig

# Open the BigWig file (replace with your file name)
bw = pyBigWig.open("/Users/ozgun/DataspellProjects/CS401-soffritto/newdata/ENCFF396RXV.bigWig")

# Print available chromosomes and their lengths
chroms = bw.chroms()
print("Chromosomes in the file:")
for chrom, length in chroms.items():
    print(f"{chrom}: {length}")

# Define a region to inspect (e.g., first 1,000,000 bases on chromosome 1)
chrom = "chr1"
start = 0
end = 1000000

# Get values for the specified region
values = bw.values(chrom, start, end)
print(f"\nValues for {chrom}:{start}-{end}:")

# Always close the BigWig file when done
bw.close()
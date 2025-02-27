import gzip
from Bio import SeqIO

file_path = "4DNFIGIS77SL.fastq.gz"

try:
    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        for record in SeqIO.parse(file, "fastq"):
            print(f"ID: {record.id}")
            print(f"Sequence: {record.seq}")
            print(f"Quality Scores: {record.letter_annotations['phred_quality']}")
            print("-" * 50)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except UnicodeDecodeError:
    print(f"Encoding error! Try changing the encoding to 'latin-1'.")
except Exception as e:
    print(f"An error occurred: {e}")


import sys
import os
import numpy as np

def ascii_convert(l: list):
    return [ord(c) for c in l]

def character_matrix(file):
    matrix = np.matrix([[-1] * 356] * 50)
    with open(file,"rt") as infile:
        raw_matrix = [ascii_convert(list(line)) for line in infile.readlines()]
        for i, l in enumerate(raw_matrix):
            for j, n in enumerate(l):
                matrix[i, j] = n
    return matrix

if __name__ == "__main__":
    path = sys.argv[1]
    # type = sys.argv[2] # Neutral/Readable/Unreadable
    scan_dir = os.scandir(path)
    for entry in scan_dir:
        if entry.is_file() and ".java" in entry.name:
            m = character_matrix(entry.path)
            np.savetxt(f"{entry.name}.matrix", m, delimiter=",", newline="\n", fmt="%s")
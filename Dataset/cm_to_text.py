
import sys
import os
import csv

def ascii_to_char(l: list):
    return [chr(int(c)) for c in l if  c != '']

if __name__ == "__main__":
    """
    Takes in a file and converts it from ASCII character matrix to text
    """
    path = sys.argv[1]
    scan_dir = os.scandir(path)
    for entry in scan_dir:
        if entry.is_file() and ".txt" in entry.name:
            out_text = ""
            print(entry.name)
            with open(entry.path, newline='') as csvf:
                reader = csv.reader(csvf, delimiter=',', quotechar='|')
                for row in reader:
                    # out_text.join(ascii_to_char(row))
                    print("".join(ascii_to_char(row)))
                exit()
            print(out_text)
            # m = character_matrix(entry.path)
            # np.savetxt(f"{entry.name}.matrix", m, delimiter=",", newline="\n", fmt="%s")
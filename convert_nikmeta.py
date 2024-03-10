import labrotation.file_handling as fh
import os

nikmeta_fpath = fh.open_file("Select Nikon metadata file")

with open(nikmeta_fpath, "r") as f:
    lines = f.readlines()
# change header
if "Time [m:s.ms]" in lines[0]:
    lines[0] = lines[0].replace("m:s.ms", "s")

# replace "," with "."
# replace m:s.ms format with s.ms format
for i in range(1, len(lines)):
    lines[i] = lines[i].replace(",", ".")
    line = lines[i].rstrip().split("\t")
    for j in range(len(line)):
        entry = line[j]
        if ":" in entry:  # assume then this is m:s.ms
            minutes = int(entry.split(":")[0])
            minutes = 60*minutes

            seconds = float(entry.split(":")[1])

            seconds = minutes + seconds
            seconds = str(seconds)
            line[j] = seconds
    lines[i] = "\t".join(line) + "\n"

output_fpath = os.path.splitext(nikmeta_fpath)[0] + "_modified.txt"
with open(output_fpath, "w") as f:
    f.writelines(lines)
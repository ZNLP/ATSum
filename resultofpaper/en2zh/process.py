import sys

with open(sys.argv[1], 'r') as file, open(sys.argv[1] + '.new', 'w') as file1:
    for line in file:
        line = line.strip()
        file1.write(''.join(line.split()) + '\n')

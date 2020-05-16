import sys

def merge(line):
    bpes = line.split()
    words = []
    for bpe in bpes:
        if bpe.startswith('▁' ):
            words.append(bpe[1:])
        elif words:
            words[-1] = words[-1] + bpe
        else:
            words.append(bpe)
    return ' '.join(words)


with open(sys.argv[1], 'r') as file1, open(sys.argv[2], 'w') as file2:
    for line in file1:
        line = merge(line.strip())
        file2.write(line + '\n')

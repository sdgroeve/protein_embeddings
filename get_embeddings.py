import sys
#import language_models as lm

device = 'cuda'

current_sequence_name = ''
current_sequence = ''

maxlen = 0
maxseq = ""
with open(sys.argv[1], 'r') as fasta_file:
    for line in fasta_file:
        line = line.strip()

        if line.startswith('>'):
            if current_sequence_name != '':    
                if len(current_sequence) > maxlen:
                    maxlen = len(current_sequence)
                    maxseq = current_sequence     
                #sequences[current_sequence_name] = current_sequence

            current_sequence_name = line[1:]
            current_sequence = ''
        else:
            current_sequence += line

    # Store the last sequence in the file
    #if current_sequence_name != '':
        #sequences[current_sequence_name] = current_sequence

print(maxlen)
print(maxseq)
d


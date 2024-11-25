import sys
import language_models as lm
from Bio import SeqIO
import torch

sequences = []
ids = []
with open(sys.argv[1]) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq)[:2000])

example_sequences = [
    'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
    'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE'
]

device = 'cuda'

#print(f'Running ESM-1b...')
#embedding, mask = lm.run_esm(example_sequences, 'ESM-1b', device=device)
#print(f'Embedding shape: {embedding.shape}')
#print(f'Mask shape: {mask.shape}')

#print(f'Running ESM-2-3B...')
#embedding, mask = lm.run_esm(example_sequences, 'ESM-2-3B', device=device)
#print(f'Embedding shape: {embedding.shape}')
#print(f'Mask shape: {mask.shape}')

#print(f'Running ESM-2-650M...')
#embedding, mask = lm.run_esm(example_sequences, 'ESM-2-650M', device=device)
#print(f'Embedding shape: {embedding.shape}')
#print(f'Mask shape: {mask.shape}')

print(f'Running ProtTransT5_XL_UniRef50...')
for i in range(len(sequences)):
    print(10*i)
    embedding, mask = lm.run_prottransxl(sequences[10*i:10*(i+1)], device=device)
    torch.save(embedding,"prottransT5xl.%i.pt"%i)

#print(f'Running Ankh_base...')
#embedding, mask = lm.run_ankh(example_sequences, 'base', device=device)
#print(f'Embedding shape: {embedding.shape}')
#print(f'Mask shape: {mask.shape}')

#print(f'Running Ankh_large...')
#embedding, mask = lm.run_ankh(example_sequences, 'large', device=device)
#print(f'Embedding shape: {embedding.shape}')
#print(f'Mask shape: {mask.shape}')


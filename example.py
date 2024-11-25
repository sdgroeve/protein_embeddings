import language_models as lm

example_sequences = [
    'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
    'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE'
]

p = ["A" for x in range(3000)]
p =''.join(p)
example_sequences.append(p)

for p in example_sequences:
    print(len(p))

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
embedding, mask = lm.run_prottransxl(example_sequences, device=device)
print(f'Embedding shape: {embedding.shape}')
print(f'Mask shape: {mask.shape}')

#print(f'Running Ankh_base...')
#embedding, mask = lm.run_ankh(example_sequences, 'base', device=device)
#print(f'Embedding shape: {embedding.shape}')
#print(f'Mask shape: {mask.shape}')

#print(f'Running Ankh_large...')
#embedding, mask = lm.run_ankh(example_sequences, 'large', device=device)
#print(f'Embedding shape: {embedding.shape}')
#print(f'Mask shape: {mask.shape}')


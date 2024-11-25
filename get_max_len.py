import language_models as lm

example_sequences = [
    'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
    'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE'
]

device = 'cuda'

s = ''.join(['A']*10)
seq = s
for i in range(50):
    seq += s
    print(len(seq))

    print(f'Running ESM-1b...')
    embedding, mask = lm.run_esm([seq], 'ESM-1b', device=device)
    print(f'Embedding shape: {embedding.shape}')
    print(f'Mask shape: {mask.shape}')

    print(f'Running ESM-2-3B...')
    embedding, mask = lm.run_esm([seq], 'ESM-2-3B', device=device)
    print(f'Embedding shape: {embedding.shape}')
    print(f'Mask shape: {mask.shape}')

    print(f'Running ESM-2-650M...')
    embedding, mask = lm.run_esm([seq], 'ESM-2-650M', device=device)
    print(f'Embedding shape: {embedding.shape}')
    print(f'Mask shape: {mask.shape}')

    print(f'Running ProtTransT5_XL_UniRef50...')
    embedding, mask = lm.run_prottransxl([seq], device=device)
    print(f'Embedding shape: {embedding.shape}')
    print(f'Mask shape: {mask.shape}')

    print(f'Running Ankh_base...')
    embedding, mask = lm.run_ankh([seq], 'base', device=device)
    print(f'Embedding shape: {embedding.shape}')
    print(f'Mask shape: {mask.shape}')

    print(f'Running Ankh_large...')
    embedding, mask = lm.run_ankh([seq], 'large', device=device)
    print(f'Embedding shape: {embedding.shape}')
    print(f'Mask shape: {mask.shape}')


import ankh
import torch
from transformers import T5Tokenizer, T5EncoderModel
import esm
import re

# PLMs to choose from
language_models = {
    'ESM-1b',
    'ESM-2-3B',
    'ESM-2-650M',
    'ProtTransT5_XL_UniRef50',
    'Ankh_base',
    'Ankh_large'
}

# size = 'base' or 'large'
def run_ankh(protein_sequences, size = 'large', device='cuda'):
    if size == 'large':
        model, tokenizer = ankh.load_large_model()
    elif size == 'base':
        model, tokenizer = ankh.load_base_model()
    else:
        raise ValueError(f'Size not supported: {size} (should be "small" or "large")')
    model.to(device)

    model.eval()
    protein_sequences = [list(seq) for seq in protein_sequences]
    outputs = tokenizer.batch_encode_plus(protein_sequences,
                                          add_special_tokens=True,
                                          padding=True,
                                          is_split_into_words=True,
                                          return_tensors="pt")
    with torch.no_grad():
        # embeddings = model(input_ids=outputs['input_ids'].to(device), attention_mask=outputs['attention_mask']).last_hidden_state.to(device)
        embeddings = model(input_ids=outputs['input_ids'].to(device)).last_hidden_state.to(device)
    return embeddings, outputs['attention_mask']

def run_prottransxl(protein_sequences, device='cuda'):
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model.to(device)
    model.half()

    protein_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein_sequences]
    ids = tokenizer.batch_encode_plus(protein_sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])

    # generate embeddings
    model.eval()
    with torch.no_grad():
        embedding_rpr = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))

    return embedding_rpr.last_hidden_state, attention_mask

# model_name = 'ESM-1b' or 'ESM-2-3B' or 'ESM-2-650M'
def run_esm(protein_sequences, model_name, device='cuda'):
    if model_name == 'ESM-2-650M':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        rep_layer = 33
    elif model_name == 'ESM-2-3B':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        rep_layer = 36
    elif model_name == 'ESM-1b':
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        rep_layer = 33
    else:
        raise ValueError(f'Model not supported: {model_name} (should be "ESM-1b" or "ESM-2-3B" or "ESM-2-650M")')
    model.to(device)

    protein_sequences = [
        (f'protein{i}', protein_sequences[i]) for i in range(len(protein_sequences))
    ]

    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(protein_sequences)

    model.eval()  # disables dropout for deterministic results
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[rep_layer], return_contacts=True)
    token_representations = results["representations"][rep_layer]
    mask = (batch_tokens != alphabet.padding_idx)

    # remove <cls> token representation
    token_representations = token_representations[:, 1:, :]
    mask = mask[:, 1:]
    return token_representations, mask
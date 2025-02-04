import torch
import torch.nn as nn
import torch.optim.adam
from tqdm import tqdm

import warnings

from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader, random_split

from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset1 import BilingualDataset, causal_mask
from model1 import build_transformer

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from config1 import get_weights_file_path, get_configuration


def get_or_build_tokenizer(config, language, dataset):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='<unk>'))
        trainer = WordLevelTrainer(special_tokens=['<sos>', '<eos>', '<unk>', '<pad>'], min_frquency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]   


def get_dataset(config):
    ds_train_raw = load_dataset('Helsinki-NLP/opus-100', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    ds_val_raw = load_dataset('Helsinki-NLP/opus-100', f'{config["lang_src"]}-{config["lang_tgt"]}', split='validation')
    # ds_test__raw = load_dataset('Helsinki-NLP/opus-100', f'{config["lang_src"]}-{config["lang_tgt"]}', split='test')
    # build tokenizers
    tokenizer_enc = get_or_build_tokenizer(config, config['lang_src'], ds_train_raw)
    tokenizer_dec = get_or_build_tokenizer(config, config['lang_tgt'], ds_train_raw)

    train_ds = BilingualDataset(ds_train_raw, tokenizer_enc, tokenizer_dec, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(ds_val_raw, tokenizer_enc, tokenizer_dec, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_enc = 0
    max_len_dec = 0

    for item in ds_val_raw:
        src_ids = tokenizer_enc.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_dec.encode(item['translation'][config['lang_tgt']]).ids
        max_len_enc = max(max_len_enc, len(src_ids))
        max_len_dec = max(max_len_dec, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_enc}')
    print(f'Max length of target sentence: {max_len_dec}')  

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_enc, tokenizer_dec

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model



def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_enc, tokenizer_dec = get_dataset(config)
    model = get_model(config, tokenizer_enc.get_vocab_size(), tokenizer_dec.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    # to resueme the training if the model crashes 

    preload = config['preload']
    if preload:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1 
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer_enc.token_to_id('<pad>'), label_smoothing=0.1)

    for epoch in range(initial_epoch, config['epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            print(f"Encoder output shape: {encoder_output.shape}")

            decoder_output = model.decode(decoder_input, decoder_mask, encoder_input, encoder_mask)
            print(f"Decoder output shape: {decoder_output.shape}")

            proj_output = model.project(decoder_output)
            print(f"Projection layer input shape: {decoder_output.shape}")
            print(f"Projection layer output shape: {proj_output.shape}")

            label = batch['label'].to(device)

            loss = loss_func(proj_output.view(-1, tokenizer_dec.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'global_step': global_step
            }, model_filename
        )

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_enc, tokenizer_dec = get_dataset(config)
    model = get_model(config, tokenizer_enc.get_vocab_size(), tokenizer_dec.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    # to resueme the training if the model crashes 

    preload = config['preload']
    if preload:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1 
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer_enc.token_to_id('<pad>'), label_smoothing=0.1)

    for epoch in range(initial_epoch, config['epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, decoder_mask, encoder_input, encoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_func(proj_output.view(-1, tokenizer_dec.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'global_step': global_step
            }, model_filename
        )  

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_configuration()
    train_model(config)
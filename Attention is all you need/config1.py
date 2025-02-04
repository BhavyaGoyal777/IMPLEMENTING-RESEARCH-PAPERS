from pathlib import Path

def get_configuration():
    return {
        'batch_size': 8,
        'epochs': 20,
        'lang_src': 'en',
        'lang_tgt': 'hi',
        'seq_len': 350,
        'lr': 1e-4,
        'model_folder': 'weights',
        'preload': None,
        'model_basename': 'TMODEL__',
        'tokenizer_file': 'tokenizer{0}.json',
        'experiment_name': 'runs/TMODEL',
        'd_model': 512
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)
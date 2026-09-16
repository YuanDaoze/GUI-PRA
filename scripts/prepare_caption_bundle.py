"""Build a self-contained Florence + OmniParser caption bundle in a new directory."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--florence-base', required=True, help='Downloaded Florence-2-base model/code/tokenizer directory.')
    p.add_argument('--icon-caption', required=True, help='Downloaded OmniParser-v2.0/icon_caption directory.')
    p.add_argument('--output', required=True, help='New output directory; existing directories are refused.')
    a = p.parse_args()
    base, caption, out = map(lambda value: Path(value).resolve(), [a.florence_base, a.icon_caption, a.output])
    required = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json']
    for name in required:
        if not (base / name).is_file():
            raise FileNotFoundError(base / name)
    if not (caption / 'config.json').is_file():
        raise FileNotFoundError(caption / 'config.json')
    weights = list(caption.glob('*.safetensors')) + list(caption.glob('pytorch_model*.bin'))
    if not weights:
        raise ValueError('No caption weights found')
    out.mkdir(parents=True, exist_ok=False)
    for source in base.iterdir():
        if source.is_file() and (source.suffix in {'.json', '.py', '.txt', '.model', '.tiktoken'} or source.name in {'LICENSE', 'README.md'}):
            if source.name not in {'model.safetensors.index.json', 'pytorch_model.bin.index.json'}:
                shutil.copy2(source, out / source.name)
    for source in caption.iterdir():
        if source.is_file() and (source.suffix in {'.safetensors', '.bin', '.json'} or source.name == 'LICENSE'):
            shutil.copy2(source, out / ('OMNIPARSER_LICENSE' if source.name == 'LICENSE' else source.name))
    config_file = out / 'config.json'
    config = json.loads(config_file.read_text())
    base_config = json.loads((base / 'config.json').read_text())
    config['auto_map'] = base_config['auto_map']
    # The corresponding Python implementations were copied from the selected base.
    for path in [config_file, out / 'preprocessor_config.json', out / 'tokenizer_config.json']:
        value = config if path == config_file else json.loads(path.read_text())
        if 'auto_map' in value:
            def local_reference(item):
                if isinstance(item, str):
                    return item.split('--')[-1]
                if isinstance(item, list):
                    return [local_reference(x) for x in item]
                return item
            value['auto_map'] = {key: local_reference(item) for key, item in value['auto_map'].items()}
        path.write_text(json.dumps(value, indent=2) + '\n')
    print(f'Caption bundle created: {out}. Source directories were read only.')


if __name__ == '__main__':
    main()

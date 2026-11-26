"""Architecture-dependent collection entry point. No guessed MLX hooks."""
import argparse, importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--adapter',type=Path); a=p.parse_args()
    if not (ROOT/'results'/'architecture.json').exists(): p.error('Run 11B first.')
    if a.adapter is None: p.error('Implement a local architecture-specific adapter after inspecting 11B. See README.md. No weights changed.')
    spec=importlib.util.spec_from_file_location('local_adapter',a.adapter); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.collect(ROOT)

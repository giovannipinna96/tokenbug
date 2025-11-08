"""
Script per creare il dataset Hugging Face dal JSON processato - VERSIONE OTTIMIZZATA PER MEMORIA.

Questa versione usa generatori per evitare di accumulare tutti i dati in memoria,
risolvendo il problema dell'OOM killer su sistemi con limiti di memoria (SLURM cgroup).

Per ogni linea di codice crea tre entry con diversi contesti:
- context_before: riga corrente + max 3 righe precedenti
- context_after: riga corrente + max 3 righe successive
- context_full: riga corrente + tutto il blocco di codice

Score: 0 se la linea è buggata, 1 se è corretta.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from datasets import Dataset, DatasetDict, Features, Value
import gc


def get_context_before(lines: List[str], line_idx: int, context_size: int = 3) -> str:
    """
    Ottiene il contesto precedente (max context_size righe) + riga corrente.

    Args:
        lines: Lista di tutte le righe del codice
        line_idx: Indice della riga corrente (0-based)
        context_size: Numero massimo di righe precedenti da includere

    Returns:
        Stringa con il contesto precedente e la riga corrente
    """
    start_idx = max(0, line_idx - context_size)
    context_lines = lines[start_idx:line_idx + 1]
    return '\n'.join(context_lines)


def get_context_after(lines: List[str], line_idx: int, context_size: int = 3) -> str:
    """
    Ottiene la riga corrente + contesto successivo (max context_size righe).

    Args:
        lines: Lista di tutte le righe del codice
        line_idx: Indice della riga corrente (0-based)
        context_size: Numero massimo di righe successive da includere

    Returns:
        Stringa con la riga corrente e il contesto successivo
    """
    end_idx = min(len(lines), line_idx + context_size + 1)
    context_lines = lines[line_idx:end_idx]
    return '\n'.join(context_lines)


def get_context_full(lines: List[str], line_idx: int) -> str:
    """
    Ottiene tutto il blocco di codice.

    Args:
        lines: Lista di tutte le righe del codice
        line_idx: Indice della riga corrente (non usato, ma mantenuto per consistenza)

    Returns:
        Stringa con tutto il codice
    """
    return '\n'.join(lines)


def process_code_snippet(
    code: str,
    buggy_lines_indices: List[int],
    dataset_type: str,
    split: str,
    metadata: Dict[str, Any],
    remove_full_code: bool = True
) -> Iterator[Dict[str, Any]]:
    """
    Processa uno snippet di codice e genera entry per il dataset HF.
    VERSIONE GENERATOR per efficienza di memoria.

    Per ogni linea crea 3 entry (context_before, context_after, context_full).

    Args:
        code: Codice sorgente
        buggy_lines_indices: Lista di indici delle linee errate (0-based)
        dataset_type: 'buggy' o 'stable'
        split: 'train', 'validation' o 'test'
        metadata: Metadati aggiuntivi (function_name, filename, etc.)
        remove_full_code: Se True, non include il campo full_code

    Yields:
        Dizionari per il dataset
    """
    if not code or not code.strip():
        return

    lines = code.split('\n')

    for line_idx, line in enumerate(lines):
        # Determina se la linea è buggata
        is_buggy = line_idx in buggy_lines_indices
        score = 0 if is_buggy else 1

        # Crea le 3 entry con diversi contesti
        contexts = [
            ('before', get_context_before(lines, line_idx)),
            ('after', get_context_after(lines, line_idx)),
            ('full', get_context_full(lines, line_idx))
        ]

        for context_type, context in contexts:
            entry = {
                'current_line': line,
                'line_index': line_idx,
                'context': context,
                'context_type': context_type,
                'score': score,
                'split': split,
                'dataset_type': dataset_type,
                'function_name': metadata.get('function_name', ''),
                'filename': metadata.get('filename', ''),
                'traceback_type': metadata.get('traceback_type', ''),
            }

            # Aggiungi full_code solo se richiesto
            if not remove_full_code:
                entry['full_code'] = code

            yield entry


def generate_entries(
    json_path: Path,
    target_split: str,
    remove_full_code: bool = True,
    max_entries: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    Generatore che produce entry per un singolo split.

    Args:
        json_path: Percorso al file JSON
        target_split: 'train', 'validation', o 'test'
        remove_full_code: Se True, rimuove il campo full_code
        max_entries: Numero massimo di entry da processare dal JSON

    Yields:
        Entry per il dataset
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_entries and max_entries < len(data):
        data = data[:max_entries]

    processed_count = 0
    yielded_count = 0

    for idx, entry in enumerate(data):
        if idx % 1000 == 0:
            print(f"  [{target_split}] Processing entry {idx}/{len(data)} ({100*idx/len(data):.1f}%) - Yielded {yielded_count} rows")

        dataset_type = entry['dataset_type']
        split = entry['split']

        # Salta se non è lo split che stiamo processando
        if split != target_split:
            continue

        processed_count += 1

        # Determina quale codice usare
        if dataset_type == 'buggy':
            code = entry['buggy_code']
            buggy_lines_indices = entry['buggy_lines_indices']
        else:
            code = entry['correct_code']
            buggy_lines_indices = []

        if not code:
            continue

        # Metadati
        metadata = {
            'function_name': entry.get('function_name', ''),
            'filename': entry.get('filename', ''),
            'traceback_type': entry.get('traceback_type', ''),
        }

        # Genera entry per questo snippet (usando il generator)
        for snippet_entry in process_code_snippet(
            code=code,
            buggy_lines_indices=buggy_lines_indices,
            dataset_type=dataset_type,
            split=split,
            metadata=metadata,
            remove_full_code=remove_full_code
        ):
            yielded_count += 1
            yield snippet_entry

    print(f"  [{target_split}] ✓ Completed: processed {processed_count} JSON entries, yielded {yielded_count} dataset rows")


def create_hf_dataset(
    json_path: Path,
    output_dir: Path,
    remove_full_code: bool = True,
    max_entries: Optional[int] = None
):
    """
    Crea il dataset Hugging Face dal JSON processato.
    VERSIONE OTTIMIZZATA: usa generatori per ridurre uso di memoria.

    Args:
        json_path: Percorso al file JSON con i dati processati
        output_dir: Directory dove salvare il dataset Arrow
        remove_full_code: Se True, rimuove il campo full_code per risparmiare memoria (default: True)
        max_entries: Numero massimo di entry da processare (None = tutte). Utile per testing.
    """
    print("=" * 80)
    print("Creating HuggingFace Dataset - MEMORY OPTIMIZED VERSION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input JSON: {json_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  File size: {json_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Remove full_code field: {remove_full_code}")
    if max_entries:
        print(f"  Max entries to process: {max_entries}")

    # Definisci lo schema del dataset
    feature_dict = {
        'current_line': Value('string'),
        'line_index': Value('int32'),
        'context': Value('string'),
        'context_type': Value('string'),
        'score': Value('int32'),
        'split': Value('string'),
        'dataset_type': Value('string'),
        'function_name': Value('string'),
        'filename': Value('string'),
        'traceback_type': Value('string'),
    }

    if not remove_full_code:
        feature_dict['full_code'] = Value('string')

    features = Features(feature_dict)

    # Crea output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Processa ogni split separatamente usando generatori
    datasets = {}

    for split_name in ['train', 'validation', 'test']:
        print(f"\n{'=' * 80}")
        print(f"Processing '{split_name}' split...")
        print("=" * 80)

        # Crea il dataset direttamente dal generatore (molto efficiente per memoria)
        dataset = Dataset.from_generator(
            lambda: generate_entries(json_path, split_name, remove_full_code, max_entries),
            features=features,
        )

        if len(dataset) > 0:
            datasets[split_name] = dataset
            print(f"✓ '{split_name}' dataset created: {len(dataset):,} rows")

            # Statistiche
            buggy_count = sum(1 for x in dataset if x['score'] == 0)
            correct_count = sum(1 for x in dataset if x['score'] == 1)
            print(f"  Buggy lines (score=0): {buggy_count:,}")
            print(f"  Correct lines (score=1): {correct_count:,}")

            # Conta per context_type
            context_types = {}
            for x in dataset:
                ct = x['context_type']
                context_types[ct] = context_types.get(ct, 0) + 1
            print(f"  Context types: {context_types}")
        else:
            print(f"⚠ '{split_name}' split is empty, skipping...")

        # Forza garbage collection dopo ogni split
        gc.collect()

    if not datasets:
        raise ValueError("No splits were created! Check your input data.")

    # Crea DatasetDict
    print(f"\n{'=' * 80}")
    print("Creating DatasetDict and saving...")
    print("=" * 80)
    dataset_dict = DatasetDict(datasets)

    # Salva in formato Arrow
    print(f"\nSaving dataset to {output_dir}...")
    dataset_dict.save_to_disk(str(output_dir))

    print("\n" + "=" * 80)
    print("✓ Dataset created successfully!")
    print("=" * 80)
    print(f"\nFinal dataset statistics:")
    for split_name, dataset in dataset_dict.items():
        print(f"  {split_name}: {len(dataset):,} rows")

    return dataset_dict


def main():
    """Funzione principale."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Crea dataset HuggingFace da JSON processato (versione ottimizzata memoria)'
    )
    parser.add_argument('--keep-full-code', action='store_true',
                        help='Mantieni il campo full_code (usa più memoria)')
    parser.add_argument('--max-entries', type=int, default=None,
                        help='Numero massimo di entry da processare (per testing)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent / 'data'
    json_path = base_dir / 'processed_dataset.json'
    output_dir = base_dir / 'hf_dataset'

    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        print("Please run process_pytracebugs.py first.")
        return

    remove_full_code = not args.keep_full_code

    try:
        dataset_dict = create_hf_dataset(
            json_path,
            output_dir,
            remove_full_code=remove_full_code,
            max_entries=args.max_entries
        )

        print(f"\n✓ Dataset saved to: {output_dir}")
        print("\nTo load the dataset:")
        print(f"  from datasets import load_from_disk")
        print(f"  dataset = load_from_disk('{output_dir}')")
    except Exception as e:
        print(f"\n✗ Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

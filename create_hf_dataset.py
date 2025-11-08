"""
Script per creare il dataset Hugging Face dal JSON processato.

Per ogni linea di codice crea tre entry con diversi contesti:
- context_before: riga corrente + max 3 righe precedenti
- context_after: riga corrente + max 3 righe successive
- context_full: riga corrente + tutto il blocco di codice

Score: 0 se la linea è buggata, 1 se è corretta.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import Dataset, DatasetDict, Features, Value


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
    metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Processa uno snippet di codice e genera entry per il dataset HF.

    Per ogni linea crea 3 entry (context_before, context_after, context_full).

    Args:
        code: Codice sorgente
        buggy_lines_indices: Lista di indici delle linee errate (0-based)
        dataset_type: 'buggy' o 'stable'
        split: 'train', 'validation' o 'test'
        metadata: Metadati aggiuntivi (function_name, filename, etc.)

    Returns:
        Lista di dizionari per il dataset
    """
    if not code or not code.strip():
        return []

    lines = code.split('\n')
    entries = []

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
                'full_code': code,
                'split': split,
                'dataset_type': dataset_type,
                'function_name': metadata.get('function_name', ''),
                'filename': metadata.get('filename', ''),
                'traceback_type': metadata.get('traceback_type', ''),
            }
            entries.append(entry)

    return entries


def create_hf_dataset(json_path: Path, output_dir: Path, remove_full_code: bool = True, max_entries: Optional[int] = None):
    """
    Crea il dataset Hugging Face dal JSON processato.

    Args:
        json_path: Percorso al file JSON con i dati processati
        output_dir: Directory dove salvare il dataset Arrow
        remove_full_code: Se True, rimuove il campo full_code per risparmiare memoria (default: True)
        max_entries: Numero massimo di entry da processare (None = tutte). Utile per testing.
    """
    print(f"Loading processed data from {json_path}")
    print(f"File size: {json_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Remove full_code field: {remove_full_code}")
    if max_entries:
        print(f"Max entries to process: {max_entries}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Limita il numero di entry se richiesto
    if max_entries and max_entries < len(data):
        print(f"Limiting to first {max_entries} entries (out of {len(data)})")
        data = data[:max_entries]

    print(f"✓ Loaded {len(data)} entries")
    print(f"Starting processing...")

    # Raccoglie le entry per split
    train_entries = []
    validation_entries = []
    test_entries = []

    total_rows = 0

    for idx, entry in enumerate(data):
        if idx % 100 == 0:  # Print more frequently
            print(f"Processing entry {idx}/{len(data)} ({100*idx/len(data):.1f}%) - Total rows so far: {total_rows}")

        dataset_type = entry['dataset_type']
        split = entry['split']

        # Determina quale codice usare
        if dataset_type == 'buggy':
            # Per buggy: usa il codice buggato
            code = entry['buggy_code']
            buggy_lines_indices = entry['buggy_lines_indices']
        else:
            # Per stable: usa il codice corretto (non ha bug)
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

        # Genera entry per questo snippet
        snippet_entries = process_code_snippet(
            code=code,
            buggy_lines_indices=buggy_lines_indices,
            dataset_type=dataset_type,
            split=split,
            metadata=metadata
        )

        # Rimuovi full_code se richiesto per risparmiare memoria
        if remove_full_code:
            for se in snippet_entries:
                se.pop('full_code', None)

        total_rows += len(snippet_entries)

        # Aggiungi allo split appropriato
        if split == 'train':
            train_entries.extend(snippet_entries)
        elif split == 'validation':
            validation_entries.extend(snippet_entries)
        elif split == 'test':
            test_entries.extend(snippet_entries)

    print("\nCreating Hugging Face datasets...")
    print(f"  Train entries: {len(train_entries)}")
    print(f"  Validation entries: {len(validation_entries)}")
    print(f"  Test entries: {len(test_entries)}")

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

    # Aggiungi full_code solo se non è stato rimosso
    if not remove_full_code:
        feature_dict['full_code'] = Value('string')

    features = Features(feature_dict)
    print("CREATE DATASET")
    # Crea i dataset
    datasets = {}

    if train_entries:
        print(f"Creating train dataset from {len(train_entries)} entries...")
        print(f"Estimated memory usage: {len(train_entries) * 500 / (1024*1024):.2f} MB")
        datasets['train'] = Dataset.from_list(train_entries, features=features)
        print(f"✓ Train dataset created")
        # Free memory
        train_entries = None

    if validation_entries:
        print(f"Creating validation dataset from {len(validation_entries)} entries...")
        print(f"Estimated memory usage: {len(validation_entries) * 500 / (1024*1024):.2f} MB")
        datasets['validation'] = Dataset.from_list(validation_entries, features=features)
        print(f"✓ Validation dataset created")
        validation_entries = None

    if test_entries:
        print(f"Creating test dataset from {len(test_entries)} entries...")
        print(f"Estimated memory usage: {len(test_entries) * 500 / (1024*1024):.2f} MB")
        datasets['test'] = Dataset.from_list(test_entries, features=features)
        print(f"✓ Test dataset created")
        test_entries = None

    # Crea DatasetDict
    print("CREATE DATADICT")
    dataset_dict = DatasetDict(datasets)

    # Salva in formato Arrow
    print(f"\nSaving dataset to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    print("✓ Dataset created successfully!")
    print(f"\nDataset statistics:")
    for split_name, dataset in dataset_dict.items():
        print(f"  {split_name}:")
        print(f"    Total rows: {len(dataset)}")
        print(f"    Buggy lines (score=0): {sum(1 for x in dataset if x['score'] == 0)}")
        print(f"    Correct lines (score=1): {sum(1 for x in dataset if x['score'] == 1)}")

        # Conta per context_type
        context_types = {}
        for x in dataset:
            ct = x['context_type']
            context_types[ct] = context_types.get(ct, 0) + 1
        print(f"    Context types: {context_types}")

    return dataset_dict


def main():
    """Funzione principale."""
    import argparse

    parser = argparse.ArgumentParser(description='Crea dataset HuggingFace da JSON processato')
    parser.add_argument('--keep-full-code', action='store_true',
                        help='Mantieni il campo full_code (usa più memoria)')
    parser.add_argument('--max-entries', type=int, default=None,
                        help='Numero massimo di entry da processare (per testing)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent / 'data'
    json_path = base_dir / 'processed_dataset.json'
    output_dir = base_dir / 'hf_dataset'

    print(f"BASE_DIR: {base_dir}")
    print(f"JSON_PATH: {json_path}")
    print(f"OUTPUT_DIR: {output_dir}")

    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        print("Please run process_pytracebugs.py first.")
        return

    remove_full_code = not args.keep_full_code
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


if __name__ == '__main__':
    main()

"""
Script per elaborare il dataset PyTraceBugs da pickle a JSON.

Carica i file pickle da buggy_dataset e stable_dataset e crea un JSON
con codice corretto, codice buggato, linee errate e indici delle linee errate.
"""

import pickle
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


def parse_bug_lines(bug_lines_str: str) -> List[int]:
    """
    Estrae gli indici delle linee (0-based) dalla stringa 'bug lines'.

    Args:
        bug_lines_str: Stringa che descrive le linee con bug (es. "1-3, 5", "line 10")

    Returns:
        Lista di indici delle linee (0-based)
    """
    if pd.isna(bug_lines_str) or not bug_lines_str:
        return []

    indices = []

    # Normalizza la stringa
    bug_lines_str = str(bug_lines_str).lower().strip()

    # Rimuovi parole comuni
    bug_lines_str = bug_lines_str.replace('line', '').replace('lines', '')

    # Pattern per range (es. "1-3")
    range_pattern = r'(\d+)\s*-\s*(\d+)'
    for match in re.finditer(range_pattern, bug_lines_str):
        start = int(match.group(1))
        end = int(match.group(2))
        # Converti da 1-based a 0-based
        indices.extend(range(start - 1, end))

    # Rimuovi i range già processati
    bug_lines_str = re.sub(range_pattern, '', bug_lines_str)

    # Pattern per numeri singoli
    single_pattern = r'\d+'
    for match in re.finditer(single_pattern, bug_lines_str):
        line_num = int(match.group())
        # Converti da 1-based a 0-based
        indices.append(line_num - 1)

    return sorted(list(set(indices)))


def extract_buggy_lines_text(code: str, indices: List[int]) -> List[str]:
    """
    Estrae il testo delle linee errate dal codice.

    Args:
        code: Codice sorgente completo
        indices: Lista di indici (0-based) delle linee errate

    Returns:
        Lista di stringhe contenenti le linee errate
    """
    if not code or not indices:
        return []

    lines = code.split('\n')
    buggy_lines = []

    for idx in indices:
        if 0 <= idx < len(lines):
            buggy_lines.append(lines[idx])

    return buggy_lines


def process_buggy_pickle(pickle_path: Path, split_name: str) -> List[Dict[str, Any]]:
    """
    Processa un file pickle del buggy_dataset.

    Args:
        pickle_path: Percorso al file pickle
        split_name: Nome dello split (train, validation, test)

    Returns:
        Lista di dizionari con i dati processati
    """
    print(f"Processing buggy dataset: {pickle_path}")

    # Usa pd.read_pickle per gestire problemi di compatibilità
    df = pd.read_pickle(pickle_path)

    processed_data = []

    for idx, row in df.iterrows():
        buggy_code = row.get('before_merge', '')
        correct_code = row.get('after_merge', '')

        # Estrai indici delle linee errate
        if split_name == 'test' and 'bug lines' in row:
            # Test set ha il campo 'bug lines'
            bug_lines_indices = parse_bug_lines(row.get('bug lines', ''))
        else:
            # Train/val set: dobbiamo inferire dalle differenze
            # Per semplicità, usiamo un approccio di diff
            bug_lines_indices = infer_buggy_lines_from_diff(buggy_code, correct_code)

        buggy_lines_text = extract_buggy_lines_text(buggy_code, bug_lines_indices)

        entry = {
            'correct_code': correct_code,
            'buggy_code': buggy_code,
            'buggy_lines_text': buggy_lines_text,
            'buggy_lines_indices': bug_lines_indices,
            'dataset_type': 'buggy',
            'split': split_name,
            'url': row.get('url', ''),
            'function_name': row.get('function_name', '') or row.get('bug function_name', ''),
            'filename': row.get('filename', '') or row.get('bug filename', ''),
            'traceback_type': row.get('traceback_type', ''),
        }

        processed_data.append(entry)

    print(f"  Processed {len(processed_data)} buggy snippets from {split_name}")
    return processed_data


def infer_buggy_lines_from_diff(buggy_code: str, correct_code: str) -> List[int]:
    """
    Inferisce le linee errate confrontando il codice buggato con quello corretto.

    Args:
        buggy_code: Codice con bug
        correct_code: Codice corretto

    Returns:
        Lista di indici (0-based) delle linee che differiscono
    """
    import difflib

    buggy_lines = buggy_code.split('\n')
    correct_lines = correct_code.split('\n')

    # Usa difflib per trovare le differenze
    diff = difflib.unified_diff(buggy_lines, correct_lines, lineterm='')

    buggy_indices = []
    current_line = 0

    for line in diff:
        if line.startswith('@@'):
            # Estrai il numero di linea dal marker @@
            match = re.search(r'-(\d+)', line)
            if match:
                current_line = int(match.group(1)) - 1
        elif line.startswith('-') and not line.startswith('---'):
            # Linea rimossa (presente nel buggy, non nel correct)
            buggy_indices.append(current_line)
            current_line += 1
        elif not line.startswith('+'):
            current_line += 1

    return sorted(list(set(buggy_indices)))


def process_stable_pickle(pickle_path: Path, split_name: str) -> List[Dict[str, Any]]:
    """
    Processa un file pickle dello stable_dataset.

    Args:
        pickle_path: Percorso al file pickle
        split_name: Nome dello split (train, validation, test)

    Returns:
        Lista di dizionari con i dati processati
    """
    print(f"Processing stable dataset: {pickle_path}")

    # Usa pd.read_pickle per gestire problemi di compatibilità
    df = pd.read_pickle(pickle_path)

    processed_data = []

    for idx, row in df.iterrows():
        correct_code = row.get('before_merge', '')  # Nel stable dataset, before_merge è già corretto

        entry = {
            'correct_code': correct_code,
            'buggy_code': None,  # Nessun codice buggato
            'buggy_lines_text': [],
            'buggy_lines_indices': [],
            'dataset_type': 'stable',
            'split': split_name,
            'url': '',
            'function_name': row.get('function_name', ''),
            'filename': row.get('filename', ''),
            'traceback_type': '',
            'repo_name': row.get('repo_name', ''),
            'commit': row.get('commit', ''),
        }

        processed_data.append(entry)

    print(f"  Processed {len(processed_data)} stable snippets from {split_name}")
    return processed_data


def main():
    """Funzione principale per processare tutti i dataset."""

    base_dir = Path(__file__).parent / 'data'

    # Percorsi ai dataset
    buggy_dir = base_dir / 'buggy_dataset'
    stable_dir = base_dir / 'stable_dataset'
    output_path = base_dir / 'processed_dataset.json'

    all_data = []

    # Processa buggy dataset
    if buggy_dir.exists():
        # Mapping dei nomi dei file reali
        split_files = {
            'train': 'bugfixes_train.pickle',
            'validation': 'bugfixes_valid.pickle',
            'test': 'bugfixes_test.pickle'
        }
        for split_name, filename in split_files.items():
            pickle_file = buggy_dir / filename
            if pickle_file.exists():
                try:
                    data = process_buggy_pickle(pickle_file, split_name)
                    all_data.extend(data)
                except Exception as e:
                    print(f"Error processing {pickle_file}: {e}")
            else:
                print(f"Warning: {pickle_file} not found")
    else:
        print(f"Warning: {buggy_dir} not found")

    # Processa stable dataset
    if stable_dir.exists():
        for split_name in ['train', 'validation', 'test']:
            pickle_file = stable_dir / f'{split_name}.pickle'
            if pickle_file.exists():
                try:
                    data = process_stable_pickle(pickle_file, split_name)
                    all_data.extend(data)
                except Exception as e:
                    print(f"Error processing {pickle_file}: {e}")
            else:
                print(f"Warning: {pickle_file} not found")
    else:
        print(f"Warning: {stable_dir} not found")

    # Salva il JSON
    print(f"\nSaving processed data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Successfully processed {len(all_data)} total entries")
    print(f"  - Buggy: {sum(1 for d in all_data if d['dataset_type'] == 'buggy')}")
    print(f"  - Stable: {sum(1 for d in all_data if d['dataset_type'] == 'stable')}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()

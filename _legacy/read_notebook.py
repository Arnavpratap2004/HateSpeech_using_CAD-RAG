import json

with open('cad_rag.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
for i, cell in enumerate(cells):
    cell_type = cell.get('cell_type', 'unknown')
    source = ''.join(cell.get('source', []))
    print(f"=== Cell {i+1} ({cell_type}) ===")
    print(source)
    print()

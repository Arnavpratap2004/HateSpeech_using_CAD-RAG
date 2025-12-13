import json

with open('cad_rag.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']

# Write all code cells to a single Python file for execution
with open('cad_rag_script.py', 'w', encoding='utf-8') as out:
    out.write("# Auto-extracted from cad_rag.ipynb\n\n")
    for i, cell in enumerate(code_cells):
        source = ''.join(cell.get('source', []))
        out.write(f"# === Cell {i+1} ===\n")
        out.write(source)
        out.write("\n\n")

print(f"Extracted {len(code_cells)} code cells to cad_rag_script.py")

# Also print the code for review
for i, cell in enumerate(code_cells):
    source = ''.join(cell.get('source', []))
    print(f"\n{'='*60}")
    print(f"CELL {i+1}")
    print('='*60)
    print(source)

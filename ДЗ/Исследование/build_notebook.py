"""Собрать ReadMe.ipynb из отчёта с PNG/GIF внутри Markdown-ячеек.

python build_notebook.py — только стандартная библиотека, расчёты не запускает.
"""
import base64
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def main():
    report = (ROOT / 'Отчёт.md').read_text(encoding='utf-8')
    anchors = re.findall(r'\]\(#([^)]+)\)', report)
    cells = []
    embedded = 0
    # shortcut: разбиение рассчитано на заголовки этого отчёта вне блоков кода;
    # при появлении таких заголовков внутри примеров нужен Markdown-парсер.
    sections = re.split(r'(?=^## \d+\. |^### )', report, flags=re.MULTILINE)
    for index, section in enumerate(sections):
        if not section.strip():
            continue
        attachments = {}

        def embed(match):
            nonlocal embedded
            alt, relative = match.groups()
            path = ROOT / relative
            mime = {'.png': 'image/png', '.gif': 'image/gif'}[path.suffix.lower()]
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            attachments[path.name] = {mime: encoded}
            assert base64.b64decode(encoded, validate=True) == path.read_bytes()
            embedded += 1
            return f'![{alt}](attachment:{path.name})'

        source = re.sub(r'!\[([^\]]*)\]\((results/[^)]+)\)', embed, section)
        heading = re.match(r'## (\d+)\. ', source)
        if heading:
            anchor = anchors[int(heading.group(1)) - 1]
            source = f'<a id="{anchor}"></a>\n\n' + source
        cell = {
            'cell_type': 'markdown',
            'id': hashlib.sha256(f'{index}:{section}'.encode('utf-8')).hexdigest()[:12],
            'metadata': {},
            'source': source.splitlines(keepends=True),
        }
        if attachments:
            cell['attachments'] = attachments
        cells.append(cell)
    assert ''.join(sections) == report
    assert embedded == 5, f'Ожидались четыре PNG и один GIF, получено {embedded}'
    assert len({c['id'] for c in cells}) == len(cells)
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12.7'},
            'title': 'От локальных правил к дендритам',
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    destination = ROOT / 'ReadMe.ipynb'
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    saved = json.loads(destination.read_text(encoding='utf-8'))
    assert saved == notebook
    assert all(c['cell_type'] == 'markdown' for c in saved['cells'])
    print(f'{destination}: {len(cells)} Markdown-ячеек, {embedded} встроенных изображений, включая GIF.')


if __name__ == '__main__':
    main()

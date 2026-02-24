# Configuration file for the Sphinx documentation builder.

import shutil
import glob
import os

# -- Copy notes/*.md into docs/source/notes/ --------------------------------
# The source of truth is notes/ at the repo root. We copy them here so Sphinx
# can include them in the toctree. The docs/source/notes/ directory is in
# .gitignore so these copies are never committed.

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_notes_src = os.path.join(_repo_root, 'notes')
_notes_dst = os.path.join(os.path.dirname(__file__), 'notes')

os.makedirs(_notes_dst, exist_ok=True)
for _f in glob.glob(os.path.join(_notes_src, '*.md')):
    if not _f.endswith('~'):
        shutil.copy2(_f, _notes_dst)

# Copy configs/ so that relative links from notes (e.g. ../configs/...) resolve.
# For each .yml/.yaml file, also generate a rendered .md page so that links
# display the YAML in the browser instead of triggering a download.
_configs_src = os.path.join(_repo_root, 'configs')
_configs_dst = os.path.join(os.path.dirname(__file__), 'configs')
if os.path.isdir(_configs_src):
    if os.path.exists(_configs_dst):
        shutil.rmtree(_configs_dst)
    shutil.copytree(_configs_src, _configs_dst)
    for _root, _dirs, _files in os.walk(_configs_dst):
        for _fname in _files:
            if _fname.endswith(('.yml', '.yaml')):
                _yml_path = os.path.join(_root, _fname)
                with open(_yml_path) as _fin:
                    _yml_content = _fin.read()
                with open(_yml_path + '.md', 'w') as _fout:
                    _fout.write('---\norphan: true\n---\n\n')
                    _fout.write(f'# {_fname}\n\n```yaml\n{_yml_content}```\n')

# Rewrite .yml/.yaml links in the copied notes to point to the rendered .md pages.
import re
for _f in glob.glob(os.path.join(_notes_dst, '*.md')):
    with open(_f) as _fin:
        _text = _fin.read()
    _new_text = re.sub(r'\(([^)]*\.ya?ml)\)', lambda m: f'({m.group(1)}.md)', _text)
    if _new_text != _text:
        with open(_f, 'w') as _fout:
            _fout.write(_new_text)

# -- Generate CLI subcommand summary table -----------------------------------
# Import the argparse parser and extract the one-line help for each subcommand.

import sys as _sys
_sys.path.insert(0, _repo_root)
from pirate_frb.__main__ import get_parser as _get_parser

_parser = _get_parser()
_lines = ['| Subcommand | Description |', '|---|---|']
for _action in _parser._subparsers._actions:
    if not hasattr(_action, '_choices_actions'):
        continue
    for _choice in _action._choices_actions:
        _lines.append(f'| `{_choice.dest}` | {_choice.help or ""} |')

# Write the summary table and per-subcommand argparse directives to _cli_generated.md.
_cli_gen_path = os.path.join(os.path.dirname(__file__), '_cli_generated.md')
with open(_cli_gen_path, 'w') as _fout:
    # Summary table
    _fout.write('## Subcommands\n\n')
    _fout.write('\n'.join(_lines) + '\n\n')
    # Per-subcommand detailed docs
    _fout.write('## Detailed usage\n\n')
    for _action in _parser._subparsers._actions:
        if not hasattr(_action, '_choices_actions'):
            continue
        for _choice in _action._choices_actions:
            _name = _choice.dest
            _fout.write(f'### {_name}\n\n')
            _fout.write('```{eval-rst}\n')
            _fout.write('.. argparse::\n')
            _fout.write('   :module: pirate_frb.__main__\n')
            _fout.write('   :func: get_parser\n')
            _fout.write('   :prog: pirate_frb\n')
            _fout.write(f'   :path: {_name}\n')
            _fout.write('```\n\n')

# -- Project information -----------------------------------------------------

project = 'pirate_frb'
copyright = '2026, Kendrick Smith'
author = 'Kendrick Smith'
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinxarg.ext',
]

templates_path = ['_templates']
exclude_patterns = ['_cli_generated.md', '_cli_summary.md']

source_suffix = {
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#007bff",
        "color-brand-content": "#0056b3",
    },
}

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
]

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

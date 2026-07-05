# Configuration file for the Sphinx documentation builder.

import shutil
import glob
import os
import tomllib

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

# Copy any compiled LaTeX PDFs (notes/*.pdf, produced by 'make tex') into the
# _static dir, so the docs can link to them as ordinary served assets that open
# INLINE in the browser. (A {download} link would emit an <a download> that forces
# a save-to-disk instead of viewing.) Creating _static here also satisfies
# html_static_path (otherwise Sphinx warns it does not exist). These PDFs are
# gitignored and rebuilt before each Sphinx run by 'make docs' / docs/deploy.sh.
_static_dst = os.path.join(os.path.dirname(__file__), '_static')
os.makedirs(_static_dst, exist_ok=True)
for _f in glob.glob(os.path.join(_notes_src, '*.pdf')):
    shutil.copy2(_f, _static_dst)

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

# Copy grpc/*.proto into docs/source/grpc/ and generate rendered .md pages.
_grpc_src = os.path.join(_repo_root, 'grpc')
_grpc_dst = os.path.join(os.path.dirname(__file__), 'grpc')
os.makedirs(_grpc_dst, exist_ok=True)

_proto_files = []
for _f in sorted(glob.glob(os.path.join(_grpc_src, '*.proto'))):
    _fname = os.path.basename(_f)
    shutil.copy2(_f, _grpc_dst)
    _proto_files.append(_fname)
    _proto_path = os.path.join(_grpc_dst, _fname)
    with open(_proto_path) as _fin:
        _proto_content = _fin.read()
    with open(_proto_path + '.md', 'w') as _fout:
        _fout.write(f'---\norphan: true\n---\n\n# {_fname}\n\n```protobuf\n{_proto_content}\n```\n')

# Generate a page listing all proto files (no toctree, to keep them out of the sidebar).
# The dict gives each proto a one-line description and also sets the display order. Any
# proto file not listed here is appended afterward (alphabetically) with no description,
# so a newly-added proto still shows up.
_grpc_descriptions = {
    'frb_search.proto':  'RPC server embedded in pirate (e.g. write_file callbacks)',
    'frb_grouper.proto': 'RPC intercommunication between pirate and grouper',
    'frb_sifter.proto':  'RPC intercommunication between grouper and sifter',
}
_grpc_ordered = ([_f for _f in _grpc_descriptions if _f in _proto_files]
                 + [_f for _f in _proto_files if _f not in _grpc_descriptions])

_grpc_gen_path = os.path.join(os.path.dirname(__file__), '_grpc_generated.md')
with open(_grpc_gen_path, 'w') as _fout:
    _fout.write('# gRPC Protocol Definitions\n\n')
    for _fname in _grpc_ordered:
        _desc = _grpc_descriptions.get(_fname)
        _suffix = f' - {_desc}' if _desc else ''
        _fout.write(f'- [{_fname}](grpc/{_fname}.md){_suffix}\n')

# Rewrite .yml/.yaml/.proto links in the copied notes to point to the rendered .md pages.
import re
for _f in glob.glob(os.path.join(_notes_dst, '*.md')):
    with open(_f) as _fin:
        _text = _fin.read()
    _new_text = re.sub(r'\(([^)]*\.ya?ml)\)', lambda m: f'({m.group(1)}.md)', _text)
    _new_text = re.sub(r'\(([^)]*\.proto)\)', lambda m: f'({m.group(1)}.md)', _new_text)
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
        _help = re.sub(r'(\w+_\w+)', r'`\1`', _choice.help or "")
        _lines.append(f'| `{_choice.dest}` | {_help} |')

# Write the summary table and per-subcommand help to _cli_generated.md.
_cli_gen_path = os.path.join(os.path.dirname(__file__), '_cli_generated.md')
with open(_cli_gen_path, 'w') as _fout:
    # Summary table
    _fout.write('## Subcommands\n\n')
    _fout.write('\n'.join(_lines) + '\n\n')
    # Per-subcommand detailed docs (captured from argparse format_help)
    _fout.write('## Detailed usage\n\n')
    for _action in _parser._subparsers._actions:
        if not hasattr(_action, '_choices_actions'):
            continue
        for _choice in _action._choices_actions:
            _name = _choice.dest
            _subparser = _action.choices[_name]
            _subparser.prog = f'pirate_frb {_name}'
            _fout.write(f'### {_name}\n\n')
            _fout.write(f'```text\n{_subparser.format_help()}```\n\n')

# -- Project information -----------------------------------------------------

project = 'pirate_frb'
copyright = '2026, Kendrick Smith'
author = 'Kendrick Smith'

# Read the version from pyproject.toml (the single source of truth) so the docs
# never drift from the package version. pyproject.toml is at the repo root (see
# _repo_root above); tomllib is stdlib on Python >= 3.11.
with open(os.path.join(_repo_root, 'pyproject.toml'), 'rb') as _f:
    release = tomllib.load(_f)['project']['version']
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    # napoleon parses NumPy/Google-style docstring sections (Parameters, Returns,
    # Yields, ...) into field lists. Without it, the "Parameters"/"Returns"
    # underlines are read as rST section headings and render in a huge font.
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
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

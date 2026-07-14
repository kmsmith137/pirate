"""Sphinx extension: auto-cross-link doc mentions of documented objects.

Design rationale, in brief: at
'doctree-resolved' time -- after autodoc has populated the python domain and
every generated page (configs/*, cli/*, grpc/*, notes/*) exists -- we walk
each page's doctree and turn a plain-text or inline-code MENTION of a
documented thing into a hyperlink, without editing any source file. Docstrings,
CLI help-text, and notes/*.md all stay pristine.

What becomes a link:
  - documented python classes / methods (the autoclass pages under classes/),
    matched as a bare name, a dotted path, or `Class.method` / `Class::member`;
  - config files (configs/....yml), gRPC protos (foo.proto), notes pages
    (notes/foo.md), and CLI subcommands (`pirate_frb <name>`), each keyed on
    the generated page's docname;
  - curated aliases from autolink_overrides.yml (e.g. RPC method names -> their
    .proto page, "tex notes" -> the compiled PDF).

Per-document policy: on notes/* pages we
create ONLY the Sphinx-only links (classes, cli). A real-file mention (a
config/proto/sibling-note PATH) is left for a handwritten markdown link in the
notes source -- which also works on GitHub -- and is reported as a
'handwrite-in-source' candidate instead of being auto-linked.

Every build writes docs/build/autolink_report.json: what got linked, and what
looked linkable but was skipped (unknown-class, missing-config-page, denied,
handwrite-in-source, unknown-cli). That report drives the /review-docs command.
"""

import json
import os
import re

from docutils import nodes
from sphinx import addnodes
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Rebuilt each 'builder-inited'; populated lazily on the first doctree.
_REGISTRY = None
_REPORT = None
_OVERRIDES = None
# refuris already present on the current page (set per doctree). Used to avoid
# nagging 'handwrite-in-source' when the page already links that target once.
_EXISTING_REFURIS = set()

# Subtrees we never descend into: code, math, existing links, autodoc
# signatures, headings. (literal_block is handled explicitly first, so the
# cli-help exception can fire before this blanket skip.)
_SKIP = (
    nodes.doctest_block, nodes.math, nodes.math_block, nodes.raw,
    nodes.comment, nodes.reference, nodes.target, nodes.title,
    addnodes.desc_signature, addnodes.pending_xref,
)

# CamelCase tokens (>=2 humps) -- used only to flag undocumented class-like
# names in the report (a wishlist of pages someone might want to add).
_CAMEL_RE = re.compile(r'\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b')

# Advisory-only noise: CamelCase that is never a pirate doc page, so keep it out
# of the 'unknown-class' wishlist. Python/library names, plus protobuf message
# and RPC-request/response types (matched by suffix below).
_ADVISORY_IGNORE = {
    'RuntimeError', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'KeyboardInterrupt', 'NotImplementedError', 'FileNotFoundError',
    'RpcError', 'NumPy', 'GitHub',
}
_ADVISORY_IGNORE_SUFFIXES = ('Request', 'Response', 'Message', 'Messages')


# -- overrides ---------------------------------------------------------------

def _load_overrides(path):
    try:
        import yaml
    except Exception:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_alias_target(val):
    # 'static:_static/foo.pdf' | 'docname#anchor' | 'docname'
    if val.startswith('static:'):
        return ('static', val[len('static:'):], None)
    if '#' in val:
        dn, anchor = val.split('#', 1)
        return ('doc', dn, anchor)
    return ('doc', val, None)


# -- registry ----------------------------------------------------------------

def _entry_fields(entry):
    # Sphinx >=4 ObjectEntry(docname, node_id, objtype, aliased); older = tuple.
    if hasattr(entry, 'docname'):
        return entry.docname, entry.node_id, entry.objtype
    return entry[0], entry[1], entry[2]


def _build_registry(app):
    env = app.env
    class_by_name = {}    # 'XEngineMetadata' and full dotted -> (docname, id)
    member_by_name = {}   # 'FrbGrouper.get_output' -> (docname, id)
    fullnames = {}        # exact fullname -> (docname, id)

    py = env.domains['py'].objects if 'py' in env.domains else {}
    for name, entry in py.items():
        docname, node_id, objtype = _entry_fields(entry)
        fullnames[name] = (docname, node_id)
        parts = name.split('.')
        if objtype == 'class':
            class_by_name[parts[-1]] = (docname, node_id)
            class_by_name[name] = (docname, node_id)
        elif objtype in ('method', 'attribute', 'property', 'function') and len(parts) >= 2:
            member_by_name['%s.%s' % (parts[-2], parts[-1])] = (docname, node_id)

    page = {}             # keyword string -> target docname
    cli_names = set()
    for docname in env.found_docs:
        if docname.startswith('configs/') and (docname.endswith('.yml') or docname.endswith('.yaml')):
            page[docname] = docname                       # 'configs/foo.yml'
        elif docname.startswith('grpc/') and docname.endswith('.proto'):
            page[docname] = docname                       # 'grpc/foo.proto'
            page[docname.split('/', 1)[1]] = docname      # 'foo.proto'
        elif docname.startswith('notes/'):
            page[docname + '.md'] = docname               # 'notes/foo.md'
        elif docname.startswith('cli/'):
            cli_names.add(docname.split('/', 1)[1])

    overrides = _OVERRIDES or {}
    alias_targets = {k: _parse_alias_target(v) for k, v in (overrides.get('aliases') or {}).items()}
    denied_global = set()
    denied_page = {}
    for d in (overrides.get('deny') or []):
        kw, pg = d.get('keyword'), d.get('page')
        if pg:
            denied_page.setdefault(pg, set()).add(kw)
        else:
            denied_global.add(kw)
    policy = overrides.get('policy') or {}

    class_bare = sorted((n for n in class_by_name if '.' not in n), key=len, reverse=True)
    class_alt = '|'.join(re.escape(n) for n in class_bare)
    alias_alt = '|'.join(re.escape(k) for k in sorted(alias_targets, key=len, reverse=True))

    parts = []
    if alias_alt:
        parts.append(r'(?<!\w)(?P<alias>' + alias_alt + r')(?!\w)')
    parts.append(r'(?<!\w)(?P<cli>(?:python -m pirate_frb|pirate_frb)\s+(?P<cliname>[a-zA-Z_]\w*))')
    parts.append(r'(?<!\w)(?P<pyfull>pirate_frb(?:\.\w+)+)')
    if class_alt:
        parts.append(r'(?P<member>(?P<mcls>' + class_alt + r')(?:\.|::)(?P<mname>\w+))(?:\(\))?')
    parts.append(r'(?P<config>configs/[\w./-]+\.ya?ml)')
    parts.append(r'(?P<proto>\b[\w./-]*\w+\.proto\b)')
    parts.append(r'(?P<notesmd>\bnotes/[\w./-]+\.md\b)')
    if class_alt:
        # Optional trailing 's' links a plural (AssembledFrames -> AssembledFrame)
        # while the span('cls') excludes it, so only the class name is linked.
        parts.append(r'(?<!\w)(?P<cls>(?:' + class_alt + r'))s?(?![\w])')

    return {
        'builder': app.builder,
        'class_by_name': class_by_name,
        'member_by_name': member_by_name,
        'fullnames': fullnames,
        'page': page,
        'cli_names': cli_names,
        'alias_targets': alias_targets,
        'denied_global': denied_global,
        'denied_page': denied_page,
        'has_alias': bool(alias_alt),
        'has_class': bool(class_alt),
        'link_cli_help': policy.get('link_cli_help_blocks', True),
        'master': re.compile('|'.join(parts)),
    }


# -- matching ----------------------------------------------------------------

def _scan(text, docname, allow_realfile, advisory, reg, denied_here, report):
    """Find link spans in `text`. Returns [(start, end, target, keyword)].

    Records skip/advisory candidates into `report` as a side effect. `target`
    is ('doc', docname, node_id|None) or ('static', path, None).
    """
    spans = []
    for m in reg['master'].finditer(text):
        target = keyword = None
        if reg['has_alias'] and m.group('alias') is not None:
            keyword = m.group('alias')
            target = reg['alias_targets'][keyword]      # aliases bypass allow_realfile
            s, e = m.span('alias')
        elif m.group('cli') is not None:
            name = m.group('cliname')
            s, e = m.span('cli')
            if name in reg['cli_names']:
                target, keyword = ('doc', 'cli/' + name, None), 'pirate_frb ' + name
            else:
                _skip(report, docname, m.group('cli'), 'unknown-cli')
                continue
        elif m.group('pyfull') is not None:
            tok = m.group('pyfull')
            s, e = m.span('pyfull')
            if tok in reg['fullnames']:
                dn, nid = reg['fullnames'][tok]
            elif tok.split('.')[-1] in reg['class_by_name']:
                dn, nid = reg['class_by_name'][tok.split('.')[-1]]
            else:
                continue                                 # e.g. pirate_frb.cuda_generator
            target, keyword = ('doc', dn, nid), tok
        elif reg['has_class'] and m.group('member') is not None:
            mcls, mname = m.group('mcls'), m.group('mname')
            s, e = m.span('member')
            mk = '%s.%s' % (mcls, mname)
            if mk in reg['member_by_name']:
                dn, nid = reg['member_by_name'][mk]
                keyword = mk
            else:
                dn, nid = reg['class_by_name'][mcls]
                keyword = mcls
            target = ('doc', dn, nid)
        elif m.group('config') is not None:
            tok = m.group('config')
            s, e = m.span('config')
            if not allow_realfile:
                _skip_realfile(report, docname, tok, reg, missing='missing-config-page')
                continue
            if tok not in reg['page']:
                _skip(report, docname, tok, 'missing-config-page')
                continue
            target, keyword = ('doc', reg['page'][tok], None), tok
        elif m.group('proto') is not None:
            tok = m.group('proto')
            s, e = m.span('proto')
            if not allow_realfile:
                _skip_realfile(report, docname, tok, reg)
                continue
            if tok not in reg['page']:
                continue
            target, keyword = ('doc', reg['page'][tok], None), tok
        elif m.group('notesmd') is not None:
            tok = m.group('notesmd')
            s, e = m.span('notesmd')
            if not allow_realfile:
                _skip_realfile(report, docname, tok, reg)
                continue
            if tok not in reg['page']:
                continue
            target, keyword = ('doc', reg['page'][tok], None), tok
        elif reg['has_class'] and m.group('cls') is not None:
            tok = m.group('cls')
            s, e = m.span('cls')
            dn, nid = reg['class_by_name'][tok]
            target, keyword = ('doc', dn, nid), tok
        else:
            continue

        if keyword in reg['denied_global'] or keyword in denied_here:
            _skip(report, docname, keyword, 'denied')
            continue
        if target[0] == 'doc' and target[1] == docname:
            continue                                     # self-link suppression
        spans.append((s, e, target, keyword))

    if advisory:
        covered = [(s, e) for s, e, _, _ in spans]
        for cm in _CAMEL_RE.finditer(text):
            tok = cm.group(0)
            if tok in reg['class_by_name'] or tok in _ADVISORY_IGNORE:
                continue
            if tok.endswith(_ADVISORY_IGNORE_SUFFIXES):
                continue
            if any(s <= cm.start() < e for s, e in covered):
                continue
            _skip(report, docname, tok, 'unknown-class')

    spans.sort()
    filtered, last_end = [], -1
    for sp in spans:
        if sp[0] >= last_end:
            filtered.append(sp)
            last_end = sp[1]
    return filtered


def _make_ref(target, docname, reg, inner):
    ref = nodes.reference('', '', internal=True)
    if target[0] == 'doc':
        uri = reg['builder'].get_relative_uri(docname, target[1])
        if target[2]:
            uri += '#' + target[2]
    else:
        uri = ('../' * docname.count('/')) + target[1]
    ref['refuri'] = uri
    ref.append(inner)
    return ref


def _link_text(text, docname, reg, spans, report):
    """Split a run of text into Text + reference nodes at `spans`."""
    out, pos = [], 0
    for s, e, target, kw in spans:
        if s > pos:
            out.append(nodes.Text(text[pos:s]))
        out.append(_make_ref(target, docname, reg, nodes.Text(text[s:e])))
        _linked(report, docname, kw, target)
        pos = e
    if pos < len(text):
        out.append(nodes.Text(text[pos:]))
    return out


# -- doctree walk ------------------------------------------------------------

def _process_text(node, docname, allow_realfile, advisory, reg, denied_here, report):
    text = node.astext()
    spans = _scan(text, docname, allow_realfile, advisory, reg, denied_here, report)
    if not spans:
        return
    parent = node.parent
    idx = parent.index(node)
    parent[idx:idx + 1] = _link_text(text, docname, reg, spans, report)


def _process_literal(node, docname, allow_realfile, reg, denied_here, report):
    # An inline code span: only link if a single keyword covers the whole span
    # (a trailing '()' is ignored), so we never mangle a compound code token.
    core = node.astext().strip()
    probe = core[:-2] if core.endswith('()') else core
    spans = _scan(probe, docname, allow_realfile, False, reg, denied_here, report)
    full = [sp for sp in spans if sp[0] == 0 and sp[1] == len(probe)]
    if len(full) != 1:
        return
    _, _, target, kw = full[0]
    parent = node.parent
    parent[parent.index(node)] = _make_ref(target, docname, reg, node.deepcopy())
    _linked(report, docname, kw, target)


def _process_cli_help(node, docname, reg, denied_here, report):
    text = node.astext()
    spans = _scan(text, docname, True, False, reg, denied_here, report)
    if not spans:
        return
    node.children = []
    node.extend(_link_text(text, docname, reg, spans, report))
    # Sphinx's HTML translator highlights a literal_block (discarding our child
    # nodes) only when node.rawsource == node.astext(). Breaking that equality
    # routes it through the parsed-literal path, which renders children (our
    # links) inside <pre>. The 'cli-help' class is preserved either way.
    node.rawsource = ''


def _walk(node, docname, allow_realfile, advisory, reg, denied_here, report):
    if isinstance(node, nodes.literal_block):
        if reg['link_cli_help'] and 'cli-help' in node.get('classes', []):
            _process_cli_help(node, docname, reg, denied_here, report)
        return
    if isinstance(node, _SKIP):
        return
    if isinstance(node, nodes.literal):
        _process_literal(node, docname, allow_realfile, reg, denied_here, report)
        return
    for child in list(node.children):
        if isinstance(child, nodes.Text):
            _process_text(child, docname, allow_realfile, advisory, reg, denied_here, report)
        else:
            _walk(child, docname, allow_realfile, advisory, reg, denied_here, report)


# -- report ------------------------------------------------------------------

def _target_str(target):
    if target[0] == 'static':
        return 'static:' + target[1]
    return target[1] + ('#' + target[2] if target[2] else '')


def _linked(report, page, keyword, target):
    report['linked'].append({'page': page, 'keyword': keyword, 'target': _target_str(target)})


def _skip(report, page, text, reason):
    report['skipped'].append({'page': page, 'text': text, 'reason': reason})


def _skip_realfile(report, docname, tok, reg, missing=None):
    """Report a real-file mention on a notes page as needing a handwritten link
    -- unless the page already links that target once (first-mention convention)
    or the target has no page at all (`missing`)."""
    target_doc = reg['page'].get(tok)
    if target_doc is None:
        if missing:
            _skip(report, docname, tok, missing)
        return
    if reg['builder'].get_relative_uri(docname, target_doc) in _EXISTING_REFURIS:
        return
    _skip(report, docname, tok, 'handwrite-in-source')


# -- event handlers ----------------------------------------------------------

def _on_builder_inited(app):
    global _REGISTRY, _REPORT, _OVERRIDES
    _REGISTRY = None
    _REPORT = {'linked': [], 'skipped': []}
    _OVERRIDES = _load_overrides(
        os.path.join(os.path.dirname(__file__), os.pardir, 'autolink_overrides.yml'))


def _on_doctree_resolved(app, doctree, docname):
    global _REGISTRY, _EXISTING_REFURIS
    if app.builder.format != 'html':
        return
    if _REGISTRY is None:
        _REGISTRY = _build_registry(app)
    reg = _REGISTRY
    _EXISTING_REFURIS = {r['refuri'] for r in doctree.findall(nodes.reference) if r.get('refuri')}
    allow_realfile = not docname.startswith('notes/')
    advisory = not docname.startswith(('cli/', 'configs/', 'grpc/'))
    denied_here = reg['denied_page'].get(docname, set())
    _walk(doctree, docname, allow_realfile, advisory, reg, denied_here, _REPORT)


def _on_build_finished(app, exception):
    if _REPORT is None or app.builder.format != 'html':
        return
    seen, skipped = set(), []
    for item in _REPORT['skipped']:
        key = (item['page'], item['text'], item['reason'])
        if key not in seen:
            seen.add(key)
            skipped.append(item)
    skipped.sort(key=lambda d: (d['reason'], d['page'], d['text']))
    out = {'linked': _REPORT['linked'], 'skipped': skipped}
    path = os.path.normpath(os.path.join(app.outdir, os.pardir, 'autolink_report.json'))
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info('autolink: %d links created, %d skipped candidates -> %s',
                len(out['linked']), len(skipped), path)


def setup(app):
    app.connect('builder-inited', _on_builder_inited)
    app.connect('doctree-resolved', _on_doctree_resolved)
    app.connect('build-finished', _on_build_finished)
    return {'version': '0.1', 'parallel_read_safe': True, 'parallel_write_safe': False}

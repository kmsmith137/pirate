# Writing sphinx-friendly docstrings

The HTML docs are built from docstrings by sphinx `autoclass`/`autodoc`, with the
`sphinx.ext.napoleon` extension enabled (see `docs/source/conf.py`). Docstrings are
rendered as reStructuredText, and napoleon additionally parses NumPy/Google-style
sections. A few conventions keep the rendered output clean. (See also
`notes/pybind11.md` for the pybind-specific version of these rules.)

## Simple properties / attributes: document them in the class docstring

Prefer documenting "simple" members -- typical read-only pybind11 properties
(`def_readonly`) and plain data attributes -- with a one-line bullet in the class
docstring, and leave the binding itself bare (no per-property docstring). For example,
in C++:

```
py::class_<Foo>(m, "Foo",
    "One-line summary of Foo.\n"
    "\n"
    "Attributes (read-only):\n"
    "\n"
    "- ``nbatches`` (int) -- beam-batches per time chunk.\n"
    "- ``grouper_ip_addr`` (str) -- the grouper's listen address (``ip:port``).\n")
    .def_readonly("nbatches", &Foo::nbatches)
    .def_readonly("grouper_ip_addr", &Foo::grouper_ip_addr);
```

Why this, rather than a per-property docstring:

- It renders compactly (one line per member), instead of the multi-line
  name / type / description block that a per-property docstring produces.
- The members are not registered as separate sphinx objects, so they stay out of
  the right-hand "on this page" sidebar (a long auto-generated property list there
  is mostly noise).
- It sidesteps a napoleon gotcha: napoleon reads a property docstring as
  "type: description" and splits it at the first colon, which mangles any docstring
  containing a colon (see "the colon gotcha" below).

Conventions for the bullets:

- Use ` -- ` (space-dash-dash-space) between the name and the description, not a
  colon. (` -- ` renders as an en-dash; a colon is easier to misparse.)
- Wrap the member name -- and any token containing a colon, `*`, or `_` -- in
  double backticks, so it renders as monospace and is inert in rST.
- Do not start the lead-in line with a bare section keyword followed by a colon
  (`Attributes:`, `Parameters:`, ...): napoleon would treat it as a section header
  and re-parse the bullets. `Attributes (read-only):` is safe (the `(` defeats the
  match).

This applies to attributes set from Python too -- e.g. in `__init__`, or attached by a
context manager's `__enter__` -- not just pybind11 properties. Document the key public
ones in the same bullet style; omit attributes that exist only for the class's internal
use (conventionally underscore-prefixed).

Methods are different: leave them as normal `:members:`-documented methods, each with
its own docstring -- they belong in the sidebar.

**The colon gotcha.** If you do write a per-member docstring anyway (e.g. a pybind11
`def_readonly("x", &C::x, "...")`, so `help(obj.x)` shows text), keep a bare colon out
of its first line. napoleon reads a member/attribute docstring as `type: description`
and splits it at the first colon, so e.g. `"listen address ('ip:port')"` renders as a
mangled "Type" field in the `autoclass` output. Wrap the colon-bearing (or `*`/`_`-
bearing) token in a double-backtick inline literal, which prevents the split and renders
it as monospace:

```
Bad:  "The grouper's listen address ('ip:port'), set at construction."
Good: "The grouper's listen address (``ip:port``), set at construction."
```

(This only affects member/attribute docstrings, not method docstrings.)

## Method / function docstrings

For methods with parameters or return values, use NumPy-style sections:

```
Summary line.

Extended description (free prose) goes here, ABOVE the sections.

Parameters
----------
x : int
    Description of x.

Returns
-------
SomeType
    Description.
```

- Put all free prose in the summary / extended description, above the first section.
  Do NOT put prose paragraphs between sections (e.g. between Parameters and Returns):
  napoleon parses the base-indented lines after a parameter list as more parameters,
  shredding the prose.
- Recognized section names include Parameters, Returns, Yields, Raises, Notes; each
  is underlined with dashes (NumPy style).

## General rST hygiene

- Keep bullet lists (and paragraphs) flush with the surrounding text. Indenting a
  block more than its surroundings makes rST render it as a block quote -- a grey,
  extra-indented box. Autodoc strips the docstring's common leading indent first, so
  what matters is indentation *relative to* the other body lines, not the raw column.
- Inline literal (monospace): wrap the text in double backticks.
- A bare `*` or `_` can be read as rST emphasis/target markup; wrap such tokens in
  double backticks if they are meant literally.
- A literal block is introduced by `::` at the end of a line, followed by an indented
  block (used e.g. for the `Usage::` examples in some class docstrings).

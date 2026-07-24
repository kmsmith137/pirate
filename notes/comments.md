# Writing comments and docstrings

Guidelines for comment/docstring CONTENT, in any language (C++, Python, CUDA,
shell). For the sphinx/rST mechanics of Python docstrings, see
[notes/docstrings.md](docstrings.md).

Comments serve two different audiences, and mixing them up is the most common way
comments go wrong. "High-level" comments (docstrings, .hpp declaration comments)
are for a human CALLER deciding how to use something. "Low-level" comments (interleaved
with code) are for a future developer -- human or LLM -- who wants to change the
implementation, or understand it in more detail.

## High-level comments (docstrings, declaration comments)

Tell the caller, as expediently as possible:

- what the function/class does;
- key "need-to-know" facts (example: "caller must hold the lock");
- counterintuitive behavior;
- footguns.

High-level comments are usually short. Humans read prose slowly -- lead with what
changes the caller's decision, and omit what doesn't.

- In C++, where declarations are strongly typed and no docstring is expected, it's
  okay to omit the high-level comment entirely, if the behavior is transparent
  from reading the function signature. (Python public API is different: the sphinx
  docs are built from docstrings.)
- High-level comments should rarely contain implementation details. If an
  implementation detail has consequences for the caller, describe the
  CONSEQUENCES, not the mechanism. It's okay to refer the reader to the code for
  more detail.
- Every rule has exceptions: a situation may arise where explaining implementation
  details is the most expedient way to explain something.

Example, from FrbGrouper -- documenting a constructor arg with a known footgun.
Too much (the caller doesn't need the mechanism):

```
restore_cuda_device (bool): controls what __exit__ does with the CUDA device that
__enter__ selected. There is an unavoidable tradeoff -- exact state restoration
vs. a possible out-of-memory error -- because in the CUDA runtime API "device N
is current" and "device N has a context" are the same fact:
[12 more lines: cudaSetDevice() semantics, context sizes, per-case bullets]
```

Right altitude (risk named in one clause, concrete recommendation, mechanism left
to a comment in `__enter__`):

```
restore_cuda_device (bool): if True, the context manager switches back to the
caller's cuda device on exit. This is "cleaner" but has a hidden footgun: it can
allocate GPU memory, which can lead to confusing out-of-memory errors on shutdown
paths. In contexts such as the grouper, where the intent is to use one gpu
throughout, False is recommended.
```

## Low-level comments (interleaved with code)

The audience is a future developer changing the implementation, or reading it in
depth. Being concise is still preferable, but here it's less important than
explaining things clearly and comprehensively. Interleaving comments with code
blocks is often good style.

The most valuable low-level comments say things that are NOT obvious from
"locally" reading the code:

- Non-local coupling. Example: "logic in the code below must be kept in sync with
  logic in some_other_function(). If anything changes here, then carefully
  revisit some_other_function()."
- Subtle nuances, counterintuitive behavior, footguns.
- Rationale behind design decisions: "design decision X was made here, instead of
  the apparently-simpler design decision Y, because of subtle nuance Z."
  Rationale is the one thing that cannot be recovered by re-reading the code.
- Block summaries, when substantially shorter than the code itself (e.g. a
  one-paragraph summary preceding a 50-100 line code block).

What to avoid:

- Don't reference old versions of the code ("previously this used ..."): the
  comment will be read in a future when the old code is forgotten. Describe the
  code as it is. If the rationale involves a rejected alternative, describe the
  alternative on its own terms (the X-instead-of-Y form above), not as history.
- Bugfixes: a comment is justified if the bug was subtle (the comment prevents
  the bug from being quietly reintroduced); routine bugfixes usually don't need
  comments.
- Superficial comments that restate what a line of code plainly does.

## Keeping the two levels straight

- State each fact at ONE altitude, and point to it from elsewhere ("see the class
  docstring", "see notes/foo.md") rather than restating it. Duplicated
  explanations drift out of sync.
- When trimming prose, relocate rather than delete, and cut mechanism before
  rationale: "how" can be recovered from the code, "why" cannot.
- Comments must stay up to date when code changes; a stale comment is worse than
  no comment.

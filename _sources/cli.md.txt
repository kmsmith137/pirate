# CLI reference

Many `pirate` features are accessed via the command-line interface:
```
pirate_frb SUBCOMMAND [ARGS...]
pirate_frb GROUP SUBCOMMAND [ARGS...]
```
where the list of subcommands, and documentation for each subcommand, are given below.
Most subcommands are nested in a group (`run`, `rpc`, `show`, `varmap`, `dev`), whose row
in the table below links to a page listing that group's subcommands; `test`, `time` and
`time_dedisperser` are typed directly. Each subcommand's page embeds its `--help` output,
captured from the argparse parser when the docs are built. Note that
`python -m pirate_frb ...` is equivalent to `pirate_frb ...`.

```{include} _cli_generated.md
```

# AGENTS.md

## Project Context

`textual-view` is a new research project. Its research questions, methods,
evaluation tasks, and target venue remain open and will be established through
discussion. Do not infer a technology stack or research conclusions from the
project name.

Do not carry over the source project's research direction, system design,
experimental results, or historical decisions.

This file records the current direction and working rules. Design details agreed
with the user belong in the relevant documents. Flag conflicts between those
documents and this file rather than silently overwriting either. Distinguish
proposals, agreed decisions, and experimental findings.

## Working Principles for Agents

1. Preserve reproducibility: record inputs, configurations, code versions, random
   seeds, and artifact provenance.
2. Prefer small, inspectable changes. Avoid large architectural changes that do
   not support the research.
3. Do not silently change experimental configurations, evaluation criteria, or
   baseline behavior.
4. Preserve evidence, tool interactions, and execution traces. Distinguish
   observed facts, authors' claims, and agent judgments.
5. Avoid introducing nondeterminism unless an experiment requires it.
6. Separate experimental mechanisms from baselines. Keep materials, models,
   budgets, and other comparison conditions consistent.
7. Report negative results honestly. Do not change tasks or metrics after the
   fact to obtain favorable results.
8. Confirm hypotheses, data splits, baselines, metrics, and budgets before formal
   experiments, then freeze the configurations.

## Working Rhythm

These conventions follow the source project's working practices. Complete work
the user has already explicitly authorized without asking for confirmation again.

1. **Discuss experiment code first.** Do not create formal experiment directories,
   add analysis scripts, or launch full-scale experiments without the user's
   agreement. Temporary exploration to answer a question is fine; report the
   findings in the reply.
2. **Update `docs/` only when asked.** Do not add design, discussion, or experiment
   records as incidental cleanup, which may conflict with the user's edits.
   Initialization only establishes the category directories.
3. **Focus on one clear task at a time.** Complete the authorized scope without
   expanding the research or starting the next phase on your own.
4. **Work on `main`.** Commit only when asked. Do not create feature branches for
   commits on your own.

## Repository Layout

| Path | Purpose |
|---|---|
| `docs/discussions/` | Research discussions and comparisons of approaches |
| `docs/designs/` | Method, system, and evaluation designs |
| `docs/experiments/` | Experimental protocols, results, and limitations |
| `references/` | External material metadata and local copies; see its README |
| `.agents/skills/` | Single source of truth for project skills |
| `packages/arxiv2md/` | Standalone HTML-to-Markdown tool used for paper close reading |

Create `experiments/`, `system/`, `eval/`, or `paper/` when needed; these names do
not imply existing implementations. Keep external data, models, caches, and run
outputs local according to `.gitignore`.

## References and Evidence

- `references/refs.bib` is the single source of paper metadata. Do not track PDFs,
  external repository clones, or datasets in Git.
- Use `<year>-<topic>` citekeys without author surnames. Use the same key for a
  paper's PDF, close-reading directory, and literature note.
- Add a paper to `refs.bib` only when it supports an identifiable research
  question, decision, or experiment; explain its purpose in the `note` field.
  Do not reuse the source project's literature rationale or add its papers by
  default.
- Keep agent-generated close readings in `references/papers/<citekey>/` until
  human review determines whether to turn them into notes in `docs/literature/`.
  Do not put unreviewed conclusions in formal documents or BibTeX notes.
- Read full papers and long literature notes only when the task requires them.
  Record the repository URL and commit when citing code as evidence.

## Agent Skills

- `paper-close-read`: close reading of a single paper from arXiv, a local PDF, or
  existing text. Produces source materials and a narrative analysis of the paper's
  motivation, approach, and contribution, with relevant figures and evidence.
  Read `.agents/skills/paper-close-read/SKILL.md` when using it.

Load skills as needed for the task. Do not migrate sessions, credentials, global
agent settings, or caches.

## Environment and Notebooks

There is currently no root Python workspace, shared `.venv`, or lockfile. Do not
assume the source project's `make setup`, `make verify`, or evaluation harness
exists here.

- `packages/arxiv2md/` is a standalone uv project. The close-reading wrapper
  explicitly locates and runs it. Install uv before use; the first run may
  download dependencies. The default cache is `.arxiv2md_cache/` at the root.
- The local PDF helper declares its PyMuPDF dependency through PEP 723 and runs
  via `uv run`.
- If a root uv workspace is introduced, use a shared lockfile and environment,
  and sync from the root. Syncing from a member directory can remove shared
  developer tools. Declare development and experiment runtime dependencies
  separately.
- Put reproducible workflows in scripts and build entry points. Use notebooks
  to read artifacts, explore, and plot; they must not be the only way to produce
  reported numbers. Every notebook must run from top to bottom on a fresh kernel.
- Clear cell outputs before committing notebooks for readable diffs. No
  nbstripout Git filter is currently installed. Keep experiment logs in artifacts
  with provenance rather than relying on notebook outputs.
- Export and commit frozen notebook HTML snapshots only when the user asks.

## Math in Markdown

Use `$...$` for inline formulas and mathematical variables. Use `$$` on separate
lines for display formulas. Do not put formulas that need rendering in code
blocks or inline code. Keep code, pseudocode, and API identifiers in code format.

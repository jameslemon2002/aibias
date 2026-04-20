# AIBias Open Release

This repository is the public code-and-data package for the study
`Large language models perceive cities through a culturally uneven baseline`.

It is a curated release rather than a dump of the full working analysis tree.
Only materials needed for source-data inspection and figure regeneration are included.

## Included

- prompt templates used in the main and robustness analyses
- panel-level and upstream source data for the four main-text figures
- supplementary source tables used by the curated supplement
- release-grade scripts to regenerate the supplementary figures and robustness figures

## Excluded

- manuscript LaTeX, journal submission files, and prose drafts
- API keys, provider tokens, and request headers tied to private local environments
- raw street-view imagery and other provider-restricted visual assets
- full model response dumps and intermediate caches that depend on private or licensed inputs
- legacy exploratory scripts not needed for the released figure set

## Repository Layout

- `prompts/`: prompt-family JSON files used in the released analyses
- `source_data/main_figures/`: source-data bundle for main-text figures
- `source_data/supplement/`: supplementary source tables and robustness summaries
- `scripts/`: public figure-generation scripts
- `figures/supplement/`: regenerated supplementary figure outputs
- `docs/`: scope and release notes

## What This Release Supports

This release is designed to support:

- source-data review for all main-text figures
- regeneration of the curated supplementary figures
- regeneration of the robustness supplementary figures
- inspection of the exact prompt wording used in the public analyses

The main-text figure source-data bundle is organized by manuscript figure and panel.
It contains the panel-level CSV, TSV and JSON tables used to assemble the manuscript figures.

## Quick Start

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

Regenerate supplementary figures:

```bash
python scripts/make_curated_supplement_figures.py
python scripts/plot_robust100_supplement_figures.py
```

Outputs are written to:

- `figures/supplement/`

## Data Note

Street-view imagery is not redistributed in this repository.
The release provides derived source tables and figure-level source data instead.
This keeps the public package lightweight and avoids redistributing provider-restricted image assets.

## Maintenance Principle

This repository is intended to remain stable and maintainable.
Only files directly tied to the current manuscript figures and supplement are included.


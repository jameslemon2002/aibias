# Main Figure Source Data

This folder collects the source data and intermediate figure outputs used for the main-text figures in the manuscript.

The manuscript figure numbering is `Fig. 1` to `Fig. 4`. In the analysis output folders, the corresponding figure groups are offset by one: manuscript `Fig. 1` draws from output `fig2`, manuscript `Fig. 2` from output `fig3`, manuscript `Fig. 3` from output `fig4`, and manuscript `Fig. 4` from output `fig5`.

## Folder Structure

- `assembled_main_pdf/`: the final assembled figure PDFs used in the manuscript.
- `Fig1/`: source data and panel renders for manuscript Fig. 1.
- `Fig2/`: source data and panel renders for manuscript Fig. 2.
- `Fig3/`: source data and panel renders for manuscript Fig. 3.
- `Fig4/`: source data and panel renders for manuscript Fig. 4.
- `manifest.csv`: machine-readable list of all copied files.
- `figure_data_map_existing.md`: the earlier figure-data mapping used as traceability documentation.

Each figure folder separates panel-level figure outputs from upstream analysis tables where possible. Panel-level folders contain the files directly used to render individual panels, usually including PDF/PNG renders and source CSV files. `upstream_analysis/` folders contain analysis summaries or joined data tables from which the panel-level files were produced.

## Figure Mapping

### Fig. 1

- `a_neutral_distance/`: semantic distance between regional prompts and the neutral prompt.
- `b_delta_distribution/`: bootstrap ENA-vs-other neutral-distance gaps.
- `c_local_pca/`: local PCA projection of prompt centroids in semantic space.
- `d_ipi_maps/`: sentiment-based IPI maps.
- `e_ipi_forest/`: region-level sentiment-based IPI estimates.

### Fig. 2

- `a_human_alignment_distance/`: semantic distance between model outputs and Geograph human text.
- `b_centroid_dispersion/`: human and model text dispersion around centroids.
- `c_distinct2/`: lexical diversity measured by DISTINCT-2.
- `d_sentiment_forest/`: human and model sentiment comparison.
- `e_sentiment_gap_map/`: spatial distribution of neutral model-minus-human sentiment gaps.
- `f_g_nation_ipi/`: nation-level human and LLM IPI comparison for England, Scotland and Wales.

### Fig. 3

- `a_shift_heatmap/`: standardized identity-conditioned shifts across six structured dimensions.
- `b_shift_by_region/`: mean absolute standardized shift by prompted region.
- `c_shift_by_dimension/`: mean absolute standardized shift by perceptual dimension.

### Fig. 4

- `a_place_pulse_vs_llm/`: Place Pulse baseline scores compared with neutral LLM structured scores.
- `b_pairwise_replication_by_axis/`: divergence and strict replication rates by subgroup axis.
- `c_margin_stratification/`: pairwise replication stratified by human qscore margin.

## Notes

- The folder is intended as a source-data bundle for the main-text figures, not as a full computational reproduction archive.
- Large model outputs and embeddings are not fully duplicated here unless they were directly required by a panel-level or upstream analysis table.
- The canonical file inventory is `manifest.csv`.

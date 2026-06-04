import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    import splink.comparison_library as cl

    return (
        DuckDBAPI,
        Linker,
        Path,
        SettingsCreator,
        block_on,
        cl,
        mo,
        pa,
        pl,
        pq,
    )


@app.cell
def _(Path):
    INPUT_PARQUET = Path("data/classified_grants.parquet")
    OUTPUT_PARQUET = Path("data/entity_clusters.parquet")
    MODEL_JSON_DIR = Path("data/splink_models")
    return INPUT_PARQUET, MODEL_JSON_DIR, OUTPUT_PARQUET


@app.cell
def _(INPUT_PARQUET, mo, pl):
    """Load classified grants and create synthetic unique_id."""
    raw_df = pl.read_parquet(INPUT_PARQUET)
    df = raw_df.with_row_index("unique_id")

    total_rows = df.height
    n_unique_ref = df["ref_number"].n_unique()
    dup_ref = total_rows - n_unique_ref

    mo.md(f"""
    # Entity Resolution with Splink

    ## Data loaded

    - **Total rows:** {total_rows:,}
    - **Unique ref_numbers:** {n_unique_ref:,}
    - **Duplicate ref_numbers:** {dup_ref:,}
    - **Synthetic `unique_id`:** created as row index
    """)
    return (df,)


@app.cell
def _(df, mo, pl):
    """Clean empty strings to nulls so Splink handles missing data correctly."""
    text_cols = [
        "recipient_legal_name_en",
        "recipient_legal_name_fr",
        "recipient_operating_name_en",
        "recipient_operating_name_fr",
        "research_organization_name_en",
        "research_organization_name_fr",
        "recipient_business_number",
        "recipient_postal_code",
        "recipient_city",
    ]
    exprs = []
    for col in text_cols:
        if col in df.columns:
            exprs.append(
                pl.when(pl.col(col).str.strip_chars().eq(""))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
    df_clean = df.with_columns(exprs) if exprs else df

    null_summary = []
    for col in text_cols:
        if col in df_clean.columns:
            nc = df_clean[col].null_count()
            null_summary.append(
                {
                    "column": col,
                    "null_count": nc,
                    "null_pct": round(nc / df_clean.height * 100, 1),
                }
            )
    null_summary_df = pl.DataFrame(null_summary).sort("null_count", descending=True)

    mo.md("### Field completeness (after converting empty strings -> null)")
    mo.ui.table(null_summary_df)
    return (df_clean,)


@app.cell
def _(df_clean, mo, pl):
    """Show recipient_type distribution."""
    type_dist = (
        df_clean.group_by("recipient_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    mo.md(
        "### Recipient type distribution (each type is a **separate deduplication shard** to avoid WSL memory crashes)"
    )
    mo.ui.table(type_dist)
    return


@app.cell
def _(block_on, cl):
    """Define blocking rules and comparisons."""
    blocking_rules = [
        block_on("recipient_business_number"),
        block_on("recipient_legal_name_en", "recipient_postal_code"),
        block_on("recipient_legal_name_en", "recipient_city"),
        block_on(
            "recipient_postal_code",
            "substr(recipient_legal_name_en, 1, 4)",
        ),
        block_on(
            "recipient_city",
            "substr(recipient_legal_name_en, 1, 3)",
        ),
        block_on("research_organization_name_en"),
    ]

    comparisons = [
        cl.NameComparison("recipient_legal_name_en"),
        cl.NameComparison("recipient_legal_name_fr"),
        cl.NameComparison("recipient_operating_name_en"),
        cl.ExactMatch("recipient_business_number").configure(
            term_frequency_adjustments=True
        ),
        cl.ExactMatch("recipient_postal_code").configure(
            term_frequency_adjustments=True
        ),
        cl.ExactMatch("recipient_city"),
        cl.NameComparison("research_organization_name_en"),
    ]
    return blocking_rules, comparisons


@app.cell
def _(
    DuckDBAPI,
    Linker,
    MODEL_JSON_DIR,
    OUTPUT_PARQUET,
    SettingsCreator,
    block_on,
    blocking_rules,
    comparisons,
    df_clean,
    mo,
    pa,
    pl,
    pq,
):
    """Process each recipient_type independently.

    Each shard is written incrementally to a single Parquet file via
    pyarrow.parquet.ParquetWriter, so at most one shard's DataFrame is
    in memory at a time.  
    """
    TYPES = ["F", "N", "P", "A", "S", "O", "G", "I"]

    cluster_id_offset = 0
    per_type_results = []
    total_rows_written = 0
    writer = None

    for rt in TYPES:
        subset = df_clean.filter(pl.col("recipient_type") == rt)
        n = subset.height

        if n < 2:
            per_type_results.append(
                {"type": rt, "rows": n, "entities": n, "status": "skipped (too few)"}
            )
            continue

        db_api = DuckDBAPI()
        settings = SettingsCreator(
            link_type="dedupe_only",
            blocking_rules_to_generate_predictions=blocking_rules,
            comparisons=comparisons,
            retain_matching_columns=False,
            retain_intermediate_calculation_columns=False,
        )
        linker = Linker(subset, settings, db_api=db_api)

        # Training
        linker.training.estimate_probability_two_random_records_match(
            [
                block_on("recipient_business_number"),
                block_on("recipient_legal_name_en", "recipient_postal_code"),
            ],
            recall=0.7,
        )
        linker.training.estimate_u_using_random_sampling(max_pairs=500_000)
        linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("recipient_business_number")
        )
        linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("recipient_legal_name_en", "recipient_postal_code")
        )

        # Predict & cluster
        pairwise = linker.inference.predict(threshold_match_probability=0.5)
        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
            pairwise, threshold_match_probability=0.95
        )
        cluster_df = clusters.as_pandas_dataframe()
        n_entities = int(cluster_df["cluster_id"].nunique())

        # Save model JSON while linker is still alive
        MODEL_JSON_DIR.mkdir(parents=True, exist_ok=True)
        linker.misc.save_model_to_json(
            str(MODEL_JSON_DIR / f"model_{rt}.json"), overwrite=True
        )

        # CRITICAL: delete linker before it leaks into cell output
        del linker
        del db_api
        del settings
        del pairwise
        del clusters

        # Offset cluster IDs and write shard directly to parquet
        cluster_df["cluster_id"] = cluster_df["cluster_id"] + cluster_id_offset
        cluster_id_offset = int(cluster_df["cluster_id"].max()) + 1

        table = pa.Table.from_pandas(cluster_df, preserve_index=False)
        del cluster_df

        if writer is None:
            OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(OUTPUT_PARQUET), table.schema)
        writer.write_table(table)
        total_rows_written += table.num_rows
        del table

        per_type_results.append(
            {"type": rt, "rows": n, "entities": n_entities, "status": "ok"}
        )

    if writer is not None:
        writer.close()
    del writer

    total_entities = cluster_id_offset
    per_type_df = pl.DataFrame(per_type_results)

    mo.md("### Per-type deduplication results")
    mo.ui.table(per_type_df)

    mo.md(f"""
    ### Aggregate result

    - **Total rows processed:** {df_clean.height:,}
    - **Total rows written:** {total_rows_written:,}
    - **Total unique entities:** {total_entities:,}
    - **Reduction ratio:** {(1 - total_entities / df_clean.height) * 100:.1f}%
    """)
    return (total_rows_written,)


@app.cell
def _(OUTPUT_PARQUET, mo, total_rows_written):
    """Confirm entity clusters were saved to disk."""
    if total_rows_written > 0:
        size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
        mo.md(f"""
        ### Output saved

        - **Entity clusters:** `{OUTPUT_PARQUET}` ({total_rows_written:,} rows, {size_mb:.1f} MB)
        """)
    else:
        mo.md("No clusters generated.")
    return


@app.cell
def _(OUTPUT_PARQUET, mo, pl):
    """Validation: cluster size distribution."""
    if OUTPUT_PARQUET.exists():
        fc = pl.read_parquet(OUTPUT_PARQUET)

        size_dist = (
            fc.group_by("cluster_id")
            .agg(pl.len().alias("cluster_size"))
            .group_by("cluster_size")
            .agg(pl.len().alias("num_clusters"))
            .sort("cluster_size")
        )

        largest = (
            fc.group_by("cluster_id")
            .agg(pl.len().alias("cluster_size"))
            .sort("cluster_size", descending=True)
            .head(20)
        )

        mo.vstack([
            mo.md("### Cluster size distribution"),
            mo.ui.table(size_dist.head(20)),
            mo.md("### Largest 20 clusters"),
            mo.ui.table(largest),
        ])
    else:
        mo.md("No data to validate.")
    return


@app.cell
def _(OUTPUT_PARQUET, df_clean, mo, pl):
    """Validation: inspect sample clusters to spot-check link quality."""
    if OUTPUT_PARQUET.exists():
        fc2 = pl.read_parquet(OUTPUT_PARQUET)

        top_cluster_ids = (
            fc2.group_by("cluster_id")
            .agg(pl.len().alias("cluster_size"))
            .sort("cluster_size", descending=True)
            .head(10)
            .select("cluster_id")
            .to_series()
            .to_list()
        )

        sample = (
            fc2.filter(pl.col("cluster_id").is_in(top_cluster_ids))
            .join(
                df_clean.select(
                    [
                        "unique_id",
                        "recipient_legal_name_en",
                        "recipient_type",
                        "recipient_business_number",
                        "recipient_postal_code",
                    ]
                ),
                on="unique_id",
                how="left",
            )
            .sort("cluster_id")
        )

        mo.vstack([
            mo.md("### Sample high-size clusters (spot-check for over-linking)"),
            mo.ui.table(sample.head(50)),
        ])
    else:
        mo.md("No data to sample.")
    return


if __name__ == "__main__":
    app.run()

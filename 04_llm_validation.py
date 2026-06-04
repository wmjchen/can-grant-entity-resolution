import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    return Path, mo, pl


@app.cell
def _(Path):
    INPUT_PARQUET = Path("data/classified_grants.parquet")
    return INPUT_PARQUET,


@app.cell
def _(INPUT_PARQUET, mo, pl):
    final_df = pl.read_parquet(INPUT_PARQUET)

    total = final_df.height
    null_count = final_df["recipient_type"].null_count()
    filled_count = total - null_count

    mo.md(f"""
    ## Loaded classified grants

    - **Total rows:** {total:,}
    - **Classified:** {filled_count:,} ({filled_count/total*100:.1f}%)
    - **Still null:** {null_count:,} ({null_count/total*100:.1f}%)
    - **Source:** `{INPUT_PARQUET}`
    """)
    return final_df,


@app.cell
def _(final_df, mo, pl):
    source_breakdown = (
        final_df
        .group_by("recipient_type_source")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    type_breakdown = (
        final_df
        .group_by("recipient_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    still_null = final_df["recipient_type"].null_count()

    mo.md(f"""
    ### Summary

    - **Still null:** {still_null:,} rows
    """)
    mo.ui.table(source_breakdown)
    mo.ui.table(type_breakdown)
    return


if __name__ == "__main__":
    app.run()

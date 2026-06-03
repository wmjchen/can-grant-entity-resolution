import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl 

    return Path, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Deterministic normalization todo list:

    - Need to strip whitespace and periods (e.g. Inc. to become Inc) from names
    - Normalize case (lowercase for comparison)
    - Bilingual names need to be split
    - Business numbers need to be pure 9 digit strings (business numbers can have leading 0s). Lots of BNs have suffixes for programs (e.g. GST numbers)
    - Placeholder business numbers need to be nulled (e.g. 999999999)
    - Postal codes need to be normalized in format H0H0H0
    - ref_number need to be stripped of whitespace

    ##Edge cases:

    - Universities often receive grants on behalf of individual researchers. Should legal recipient be individual or research_organization_name? For this pipeline, I would want recipient to be the research_organization because I'm looking for aggregates for my visualization.
    """)
    return


@app.cell
def _(Path, pl):
    INPUT_CSV = Path("data/grants.csv")
    OUTPUT_PARQUET = Path("data/normalized_grants.parquet")

    raw_df = pl.read_csv(
        INPUT_CSV,
        encoding="utf8-lossy",
        infer_schema_length=10_000,
    )
    return OUTPUT_PARQUET, raw_df


@app.cell
def _(pl, raw_df):
    # Strip whitespace and periods, split bilingual names
    # Caveat: No | means that it will be treated as an English name.

    name_cols = [
        "recipient_legal_name",
        "recipient_operating_name",
        "research_organization_name",
    ]

    exprs = []

    for col in name_cols:
        raw_col = pl.col(col).cast(pl.Utf8)

        parts = raw_col.str.split_exact("|", 1)

        en_part = parts.struct.field("field_0")
        fr_part = parts.struct.field("field_1")

        exprs.extend([
            raw_col.alias(f"{col}_raw"),

            en_part
            .str.strip_chars()
            .str.replace_all(r"\.", "")
            .str.to_lowercase()
            .alias(f"{col}_en"),

            fr_part
            .str.strip_chars()
            .str.replace_all(r"\.", "")
            .str.to_lowercase()
            .alias(f"{col}_fr"),
        ])

    normalized_name_df = raw_df.with_columns(exprs)
    return (normalized_name_df,)


@app.cell
def _(mo, normalized_name_df, pl):
    # 

    placeholders = [
        "000000000",
        "999999999",
        "99999999",
        "123456789",
        "00000000",
        "N/A",
        "n/a",
        "xxxxxxx",
        "Non actif",
    ]

    digit_placeholders = [
        "000000000",
        "999999999",
        "99999999",
        "123456789",
        "00000000",
    ]

    normalized_bn_df = normalized_name_df.rename(
        {"recipient_business_number": "bn_raw"}
    )

    bn_raw = pl.col("bn_raw").cast(pl.Utf8).str.strip_chars()
    bn_stripped = bn_raw.str.replace_all(r"[^\d]", "")

    bn_digits = (
        pl.when(bn_stripped.str.len_chars() >= 9)
        .then(bn_stripped.str.slice(0, 9))
        .otherwise(None)
    )

    bn_is_placeholder = bn_raw.is_in(placeholders) | bn_digits.is_in(digit_placeholders)

    normalized_bn_df = normalized_bn_df.with_columns(
        pl.when(bn_is_placeholder)
        .then(None)
        .otherwise(bn_digits)
        .alias("recipient_business_number")
    )

    total = normalized_bn_df.height
    bn_null = normalized_bn_df["recipient_business_number"].null_count()
    bn_valid = total - bn_null

    summary_bn_df = mo.md(
        f"""
    Summary:

    - **Valid business numbers:** {bn_valid:,} ({bn_valid / total * 100:.1f}%)
    - **Null (missing or placeholder):** {bn_null:,} ({bn_null / total * 100:.1f}%)
    """
    )

    normalized_bn_df, summary_bn_df
    return (normalized_bn_df,)


@app.cell
def _(mo, normalized_bn_df, pl):
    # Normalize Canadian postal codes. 
    # Caveat: Nothing is done for foreign postal codes, and there aren't many.

    country_normalized = (
        pl.col("recipient_country")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
    )

    is_canada = country_normalized.is_in(["CA", "CANADA"])

    normalized_pc_df = normalized_bn_df.rename(
        {"recipient_postal_code": "postal_code_raw"}
    )

    pc_raw = pl.col("postal_code_raw").cast(pl.Utf8)
    pc_clean = pc_raw.str.strip_chars().str.replace_all(r"\s+", "").str.to_uppercase()
    pc_valid = pc_clean.str.contains(r"^[A-Z]\d[A-Z]\d[A-Z]\d$")

    pc_normalized = (
        pl.when(is_canada)
        .then(pl.when(pc_valid).then(pc_clean).otherwise(pl.lit(None)))
        .otherwise(pc_raw)
    )

    normalized_pc_df = normalized_pc_df.with_columns(
        pc_normalized.alias("recipient_postal_code")
    )

    ca_count = normalized_pc_df.filter(is_canada).height
    ca_pc_valid = normalized_pc_df.filter(
        is_canada & pl.col("recipient_postal_code").is_not_null()
    ).height
    non_ca_count = normalized_pc_df.filter(~is_canada).height

    summary_pc_df = mo.md(
        f"""
    ## Postal code summary

    - **Canadian records:** {ca_count:,}
    - **Canadian with valid postal code:** {ca_pc_valid:,} ({ca_pc_valid / ca_count * 100:.1f}% of Canadian)
    - **Non-Canadian records:** {non_ca_count:,} (postal codes unchanged)
    """
    )

    normalized_pc_df, summary_pc_df
    return (normalized_pc_df,)


@app.cell
def _(mo, normalized_pc_df, pl):
    # Strip whitespace from reference numbers

    normalized_ref_df = normalized_pc_df.rename({"ref_number": "ref_number_raw"})

    ref_raw = pl.col("ref_number_raw").cast(pl.Utf8)
    ref_normalized = ref_raw.str.replace_all(r"\s+", "")

    normalized_ref_df = normalized_ref_df.with_columns(ref_normalized.alias("ref_number"))

    had_whitespace = normalized_ref_df.filter(
        pl.col("ref_number_raw").cast(pl.Utf8).str.contains(r"\s")
        & pl.col("ref_number_raw").is_not_null()
    ).height

    summary_ref_df = mo.md(
        "## Reference number normalization\n\n"
        f"**Records with whitespace stripped:** {had_whitespace:,}"
    )

    normalized_ref_df,summary_ref_df

    return (normalized_ref_df,)


@app.cell
def _(OUTPUT_PARQUET, mo, normalized_ref_df):
    # Export the dataframe as a parquet
    normalized_ref_df.write_parquet(OUTPUT_PARQUET)

    mo.vstack([
        mo.md(
            f"Wrote **{normalized_ref_df.height:,}** rows with "
            f"**{len(normalized_ref_df.columns)}** columns to "
            f"`{OUTPUT_PARQUET}`"
        ),
        mo.ui.table(normalized_ref_df.head(20)),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, json, mo, pl


@app.cell
def _(Path):
    INPUT_CSV = Path("data/grants.csv")
    OUTPUT_DIR = Path("data")
    OUTPUT_PATH = OUTPUT_DIR / "profile_report.json"
    return INPUT_CSV, OUTPUT_DIR, OUTPUT_PATH


@app.cell
def _(mo):
    mo.md("""
    # Grants data profile

    First pass of the Federal Government of Canada grants and contributions dataset found at:

    https://open.canada.ca/data/en/dataset/432527ab-7aac-45b5-81d6-7597107a7013
    """)
    return


@app.cell
def _(INPUT_CSV, mo):
    mo.md(f"""
    **Input file:** `{INPUT_CSV}`
    """)
    return


@app.cell
def _(INPUT_CSV, pl):
    # Read with utf8-lossy because the file had mixed character encoding for some reason?
    # Schema length increased to accurately guess column types in polars
    df = pl.read_csv(
        INPUT_CSV,
        encoding="utf8-lossy",
        infer_schema_length=10_000,
    )

    row_count = df.height
    column_count = len(df.columns)

    row_count, column_count
    return column_count, df, row_count


@app.cell
def _(column_count, mo, row_count):
    mo.md(f"""
    Shape: {row_count:,} rows and {column_count:,} columns
    """)
    return


@app.cell
def _(df):
    key_cols = [
        "recipient_business_number",
        "recipient_legal_name",
        "recipient_operating_name",
        "research_organization_name",
        "recipient_city",
        "recipient_province",
        "recipient_postal_code",
        "recipient_type",
        "owner_org",
    ]

    available_key_cols = [col for col in key_cols if col in df.columns]
    missing_key_cols = [col for col in key_cols if col not in df.columns]

    available_key_cols, missing_key_cols
    return (available_key_cols,)


@app.cell
def _(pl):
    def count_empty_strings(frame: pl.DataFrame, column_name: str) -> int:
        """Count blank strings without counting null values twice."""
        return (
            frame.filter(pl.col(column_name).is_not_null())
            .select(pl.col(column_name).cast(pl.Utf8).str.strip_chars().eq("").sum())
            .item()
        )


    def profile_completeness(frame: pl.DataFrame, columns: list[str]) -> dict:
        total_rows = frame.height
        completeness = {}

        for column_name in columns:
            null_count = frame[column_name].null_count()
            empty_count = count_empty_strings(frame, column_name)
            filled_count = total_rows - null_count - empty_count

            completeness[column_name] = {
                "total": total_rows,
                "filled": filled_count,
                "null": null_count,
                "empty": empty_count,
                "fill_rate": round(filled_count / total_rows * 100, 1),
            }

        return completeness


    def profile_recipient_types(frame: pl.DataFrame) -> dict:
        total_rows = frame.height

        if "recipient_type" not in frame.columns:
            return {}

        type_counts = (
            frame.group_by("recipient_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

        result = {}
        for row in type_counts.iter_rows(named=True):
            recipient_type = row["recipient_type"] or "(empty)"
            count = row["count"]
            result[recipient_type] = {
                "count": count,
                "pct": round(count / total_rows * 100, 1),
            }

        return result


    def profile_bn_formats(frame: pl.DataFrame) -> dict:
        if "recipient_business_number" not in frame.columns:
            return {}

        bn_clean = frame["recipient_business_number"].cast(pl.Utf8).str.strip_chars()
        frame_with_bn = frame.with_columns(bn_clean.alias("bn_clean"))

        total_rows = frame_with_bn.height
        missing_bn = frame_with_bn.filter(
            pl.col("bn_clean").is_null() | pl.col("bn_clean").eq("")
        ).height

        has_bn = frame_with_bn.filter(
            pl.col("bn_clean").is_not_null() & pl.col("bn_clean").ne("")
        )

        has_rt_suffix = has_bn.filter(pl.col("bn_clean").str.contains(r"(?i)RT")).height
        has_rp_suffix = has_bn.filter(pl.col("bn_clean").str.contains(r"(?i)RP")).height
        pure_digits = has_bn.filter(pl.col("bn_clean").str.contains(r"^\d+$")).height
        placeholders = has_bn.filter(
            pl.col("bn_clean").is_in(
                ["00000000", "000000000", "99999999", "999999999", "n/a", "N/A"]
            )
        ).height

        other_format = (
            total_rows
            - missing_bn
            - pure_digits
            - has_rt_suffix
            - has_rp_suffix
            - placeholders
        )

        return {
            "total_rows": total_rows,
            "missing_bn": missing_bn,
            "has_bn": total_rows - missing_bn,
            "pure_9_digit": pure_digits,
            "has_rt_suffix": has_rt_suffix,
            "has_rp_suffix": has_rp_suffix,
            "placeholder": placeholders,
            "other_format": other_format,
        }


    def profile_bilingual_names(frame: pl.DataFrame) -> dict:
        columns_to_check = [
            "recipient_legal_name",
            "recipient_operating_name",
            "research_organization_name",
            "recipient_city",
        ]

        result = {}
        for column_name in columns_to_check:
            if column_name not in frame.columns:
                continue

            pipe_count = frame.filter(
                pl.col(column_name).is_not_null()
                & pl.col(column_name).cast(pl.Utf8).str.contains(r"\|")
            ).height

            result[column_name] = {
                "with_pipe": pipe_count,
                "pct": round(pipe_count / frame.height * 100, 1),
            }

        return result


    def profile_owner_orgs(frame: pl.DataFrame) -> dict:
        if "owner_org" not in frame.columns:
            return {}

        org_counts = (
            frame.group_by("owner_org")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(20)
        )

        return {
            row["owner_org"] or "(empty)": row["count"]
            for row in org_counts.iter_rows(named=True)
        }


    def profile_unique_names(frame: pl.DataFrame) -> dict:
        if "recipient_legal_name" not in frame.columns:
            return {}

        legal_names = frame["recipient_legal_name"].cast(pl.Utf8).str.strip_chars()
        unique_exact = legal_names.drop_nulls().n_unique()
        unique_lower = legal_names.drop_nulls().str.to_lowercase().n_unique()

        return {
            "unique_legal_names_exact": unique_exact,
            "unique_legal_names_lower": unique_lower,
            "case_collisions": unique_exact - unique_lower,
        }

    return (
        profile_bilingual_names,
        profile_bn_formats,
        profile_completeness,
        profile_owner_orgs,
        profile_recipient_types,
        profile_unique_names,
    )


@app.cell
def _(
    available_key_cols,
    df,
    profile_bilingual_names,
    profile_bn_formats,
    profile_completeness,
    profile_owner_orgs,
    profile_recipient_types,
    profile_unique_names,
):
    completeness = profile_completeness(df, available_key_cols)
    recipient_types = profile_recipient_types(df)
    business_number_formats = profile_bn_formats(df)
    bilingual_names = profile_bilingual_names(df)
    top_owner_orgs = profile_owner_orgs(df)
    unique_names = profile_unique_names(df)

    report = {
        "total_rows": df.height,
        "total_columns": len(df.columns),
        "completeness": completeness,
        "recipient_types": recipient_types,
        "business_number_formats": business_number_formats,
        "bilingual_names": bilingual_names,
        "top_owner_orgs": top_owner_orgs,
        "unique_names": unique_names,
    }

    report
    return (
        bilingual_names,
        business_number_formats,
        completeness,
        recipient_types,
        report,
        top_owner_orgs,
        unique_names,
    )


@app.cell
def _(completeness, pl):
    completeness_df = pl.DataFrame(
        [
            {
                "column": column_name,
                **stats,
            }
            for column_name, stats in completeness.items()
        ]
    ).sort("fill_rate")

    completeness_df
    return (completeness_df,)


@app.cell
def _(completeness_df, mo):
    mo.md("## Completeness by key column")
    mo.ui.table(completeness_df)
    return


@app.cell
def _(pl, recipient_types):
    recipient_type_df = pl.DataFrame(
        [
            {
                "recipient_type": recipient_type,
                **stats,
            }
            for recipient_type, stats in recipient_types.items()
        ]
    ).sort("count", descending=True)

    recipient_type_df
    return


@app.cell
def _(business_number_formats, mo):
    has_bn = business_number_formats.get("has_bn", 0)
    total_rows = business_number_formats.get("total_rows", 0)
    bn_rate = round(has_bn / total_rows * 100, 1) if total_rows else 0

    mo.md(
        f"""
        ## Business number coverage

        Business number is present on **{has_bn:,} / {total_rows:,} rows** (**{bn_rate}%**).
        """
    )
    return


@app.cell
def _(business_number_formats, pl):
    business_number_df = pl.DataFrame(
        [
            {"metric": metric, "value": value}
            for metric, value in business_number_formats.items()
        ]
    )

    business_number_df
    return (business_number_df,)


@app.cell
def _(business_number_df, mo):
    mo.ui.table(business_number_df)
    return


@app.cell
def _(bilingual_names, pl):
    bilingual_df = pl.DataFrame(
        [
            {
                "column": column_name,
                **stats,
            }
            for column_name, stats in bilingual_names.items()
        ]
    ).sort("with_pipe", descending=True)

    bilingual_df
    return (bilingual_df,)


@app.cell
def _(bilingual_df, mo):
    mo.md("## Bilingual-looking values")
    mo.ui.table(bilingual_df)
    return


@app.cell
def _(pl, top_owner_orgs):
    owner_org_df = pl.DataFrame(
        [
            {"owner_org": owner_org, "count": count}
            for owner_org, count in top_owner_orgs.items()
        ]
    ).sort("count", descending=True)

    owner_org_df
    return (owner_org_df,)


@app.cell
def _(mo, owner_org_df):
    mo.md("## Top owner organizations")
    mo.ui.table(owner_org_df)
    return


@app.cell
def _(mo, unique_names):
    mo.md(f"""
    ## Unique recipient legal names

    - Exact unique names: **{unique_names.get('unique_legal_names_exact', 0):,}**
    - Lowercase unique names: **{unique_names.get('unique_legal_names_lower', 0):,}**
    - Possible case-only collisions: **{unique_names.get('case_collisions', 0):,}**
    """)
    return


@app.cell
def _(OUTPUT_DIR, OUTPUT_PATH, json, mo, report):
    # Report from the current run
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w") as output_file:
        json.dump(report, output_file, indent=2)

    mo.md(f"Report written to `{OUTPUT_PATH}`")
    return


@app.cell
def _(df, mo):
    mo.md("## Quick sample")
    mo.ui.table(df.head(20))
    return


@app.cell
def _(df, pl):
    # Confirm if ref_number is truly unique. Could be used as a unique reference for Splink.abs

    def profile_unique_ref_numbers(frame: pl.DataFrame) -> dict:
        if "ref_number" not in frame.columns:
            return {}

        total_rows = frame.height

        ref_numbers = (
            frame["ref_number"]
            .cast(pl.Utf8)
            .str.strip_chars()
        )

        non_empty = ref_numbers.filter(
            ref_numbers.is_not_null() & ref_numbers.ne("")
        )

        unique_count = non_empty.n_unique()
        duplicate_count = non_empty.len() - unique_count

        return {
            "total_rows": total_rows,
            "non_empty_ref_numbers": non_empty.len(),
            "unique_ref_numbers": unique_count,
            "duplicate_ref_numbers": duplicate_count,
            "every_non_empty_ref_number_is_unique?": duplicate_count == 0,
        }

    profile_unique_ref_numbers(df)
    return


if __name__ == "__main__":
    app.run()

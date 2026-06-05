import polars as pl
from pathlib import Path

INPUT = Path("data/entity_clusters.parquet")
OUTPUT = Path("data/canonical_names.parquet")

SUFFIX_MAP = {
    "P": " (Individuals)",
    "A": " (Indigenous)",
    "S": " (Academic)",
    "F": " (Festivals)",
    "N": " (Non-profits)",
    "O": " (Other)",
    "G": " (Government)",
    "I": " (International)",
}


def main():
    df = pl.read_parquet(INPUT)

    # Choose best name per cluster: prefer research_organization_name, else recipient_legal_name
    canonical = (
        df.group_by("cluster_id")
        .agg(
            # Most frequent non-null research_organization_name
            research_name=pl.col("research_organization_name")
            .drop_nulls()
            .mode()
            .first(),
            # Fallback to most frequent recipient_legal_name
            legal_name=pl.col("recipient_legal_name").mode().first(),
            recipient_type=pl.col("recipient_type").first(),
            total_funding=pl.col("agreement_value").sum(),
            grant_count=pl.len(),
        )
        .with_columns(
            pl.when(pl.col("research_name").is_not_null())
            .then(pl.col("research_name"))
            .otherwise(pl.col("legal_name"))
            .alias("base_name")
        )
        .with_columns(
            (
                pl.col("base_name")
                + pl.col("recipient_type").replace_strict(SUFFIX_MAP, default="")
            ).alias("canonical_name")
        )
        .select(
            "cluster_id",
            "canonical_name",
            "recipient_type",
            "total_funding",
            "grant_count",
        )
        .sort("cluster_id")
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_parquet(OUTPUT)

    print(f"Written {len(canonical):,} canonical names to {OUTPUT}")
    print(canonical.head(10))


if __name__ == "__main__":
    main()

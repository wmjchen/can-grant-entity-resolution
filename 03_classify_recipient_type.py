import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import json
    import asyncio
    import os
    import time
    from pathlib import Path

    from dotenv import load_dotenv
    import marimo as mo
    import polars as pl

    # Load .env from the same directory as this notebook so it works
    # regardless of where the server / runner was started.
    _script_dir = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
    _dotenv_path = _script_dir / ".env"
    if _dotenv_path.exists():
        load_dotenv(dotenv_path=_dotenv_path, override=True)
    else:
        load_dotenv()
    return Path, asyncio, json, mo, os, pl, time


@app.cell
def _(Path, os):
    INPUT_PARQUET = Path("data/normalized_grants.parquet")
    OUTPUT_PARQUET = Path("data/classified_grants.parquet")
    LLM_CACHE_PATH = Path("data/.llm_results_cache.parquet")

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepinfra.com/v1/openai")
    OPENAI_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

    BATCH_SIZE = 1
    MAX_CONCURRENT = 200
    MAX_RETRIES = 2
    return (
        BATCH_SIZE,
        INPUT_PARQUET,
        LLM_CACHE_PATH,
        MAX_CONCURRENT,
        MAX_RETRIES,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        OPENAI_MODEL,
        OUTPUT_PARQUET,
    )


@app.cell
def _(Path, mo):
    _llm_cache_path = Path("data/.llm_results_cache.parquet")
    cache_exists = _llm_cache_path.exists()
    reuse_llm_cache = mo.ui.checkbox(
        label="Reuse existing LLM cache",
        value=cache_exists,
    )
    mo.md(f"""
    ## LLM Cache Control
    
    Cache file: `{_llm_cache_path}` ({'exists' if cache_exists else 'not found'})
    """)
    reuse_llm_cache
    return reuse_llm_cache


@app.cell
def _(INPUT_PARQUET, mo, pl):
    df = pl.read_parquet(INPUT_PARQUET)

    total = df.height
    null_count = df["recipient_type"].null_count()
    filled_count = total - null_count

    mo.md(f"""
    ## Loaded data

    - **Total rows:** {total:,}
    - **Already classified:** {filled_count:,} ({filled_count/total*100:.1f}%)
    - **Null recipient_type:** {null_count:,} ({null_count/total*100:.1f}%)
    """)
    return df, total


@app.cell
def _(df, mo, pl, total):
    def classify_deterministic(frame: pl.DataFrame) -> pl.DataFrame:
        name_col = pl.col("recipient_legal_name_en")
        research_col = pl.col("research_organization_name_en")
        bn_col = pl.col("bn_raw")

        single_comma = name_col.str.count_matches(",", literal=True).eq(1)
        comma_after = name_col.str.extract(r"^[^,]*, ?(.*)$", 1).fill_null("")
        suffix_short = comma_after.str.len_chars().le(18)
        comma_before = name_col.str.extract(r"^([^,]*),", 1).fill_null("")
        prefix_few_words = comma_before.str.strip_chars().str.split(" ").list.len().le(2)
        total_words_ok = name_col.str.split(" ").list.len().is_between(2, 4)
        person_pattern = single_comma & suffix_short & prefix_few_words & total_words_ok

        rules = (
            pl.when(
                research_col.is_not_null() & (research_col != "")
            ).then(pl.lit("S"))
            .when(name_col.str.contains(r"(?i)universit|college|c[eé]gep|polytechnic|school district"))
            .then(pl.lit("S"))
            .when(bn_col.str.contains(r"RR\d{1,4}"))
            .then(pl.lit("N"))
            .when(name_col.str.contains(r"(?i)\binc\b|\bincorporated\b|\bltd\b|\bllc\b|\bllp\b|\bcorp\b|corporation|company|limited|\blt[ée]e\b|\benr\b|senc\b"))
            .then(pl.lit("F"))
            .when(name_col.str.contains(r"(?i)society|association|foundation|non-?profit|charitable|charity|federation|council|coalition|network|committee|league|guild|chamber of commerce|\bagency\b|coop[ée]rati[fv]|\bgroup\b|\bgroupe\b|conservancy|conservation authority|\bfestival\b|alliance"))
            .then(pl.lit("N"))
            .when(name_col.str.contains(r"(?i)first nation|indigenous|aboriginal|inuit|m[eé]tis|band council"))
            .then(pl.lit("A"))
            .when(name_col.str.contains(r"(?i)government|municipal|city of|town of|village of|province of|regional municipality|county of|\bcounty\b|\bministry\b|minist[èe]re\b|\bdepartment\b"))
            .then(pl.lit("G"))
            .when(name_col.str.contains(r"(?i)hospital|h[ôo]pital"))
            .then(pl.lit("N"))
            .when(name_col.str.contains(r"(?i)church|parish|paroisse|mosque|synagogue|temple"))
            .then(pl.lit("N"))
            .when(name_col.str.contains(r"(?i)museum|musee|musée|library|biblioth"))
            .then(pl.lit("N"))
            .when(name_col.str.contains(r"(?i)media|newspaper|journal|press|radio|broadcast"))
            .then(pl.lit("F"))
            .when(person_pattern)
            .then(pl.lit("P"))
            .otherwise(None)
        )

        return frame.with_columns(
            pl.when(frame["recipient_type"].is_not_null())
            .then(frame["recipient_type"])
            .otherwise(rules)
            .alias("recipient_type_new"),
            pl.when(frame["recipient_type"].is_not_null())
            .then(pl.lit("original"))
            .when(rules.is_not_null())
            .then(pl.lit("deterministic"))
            .otherwise(None)
            .alias("recipient_type_source"),
            pl.when(frame["recipient_type"].is_not_null())
            .then(pl.lit(1.0))
            .when(rules.is_not_null())
            .then(pl.lit(1.0))
            .otherwise(None)
            .alias("recipient_type_confidence"),
        )

    classified_df = classify_deterministic(df)

    det_classified = classified_df.filter(
        (pl.col("recipient_type_source") == "deterministic")
    ).height
    remaining_for_llm = classified_df.filter(
        pl.col("recipient_type_source").is_null()
    ).height

    mo.md(f"""
    ## Deterministic classification

    - **Classified by rules:** {det_classified:,} rows
    - **Remaining for LLM:** {remaining_for_llm:,} rows
    - **Coverage:** {(total - remaining_for_llm) / total * 100:.1f}%
    """)
    return (classified_df,)


@app.cell
def _(classified_df, mo, pl):
    det_breakdown = (
        classified_df
        .filter(pl.col("recipient_type_source") == "deterministic")
        .group_by("recipient_type_new")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    mo.md("### Breakdown so far (from rules-based inference)")
    mo.ui.table(det_breakdown)
    return


@app.cell
def _(classified_df, pl):
    remaining = classified_df.filter(pl.col("recipient_type_source").is_null())

    ctx = (
        remaining
        .group_by("recipient_legal_name_en")
        .agg([
            pl.col("owner_org").drop_nulls().unique().alias("orgs"),
            pl.col("prog_name_en").drop_nulls().unique().alias("progs"),
            pl.col("naics_identifier").drop_nulls().unique().alias("naics"),
            pl.col("recipient_operating_name_en").drop_nulls().unique().alias("op_names"),
            pl.col("recipient_business_number").drop_nulls().is_not_null().any().alias("has_bn"),
            pl.len().alias("n"),
        ])
        .sort("n", descending=True)
    )

    unique_names = ctx.height
    return (ctx,)


@app.cell
def _(BATCH_SIZE, ctx):
    def build_prompt_entry(row: dict) -> str:
        lines = [f"name: {row['recipient_legal_name_en']}"]
        if row["op_names"]:
            lines.append(f"operating_as: {', '.join(row['op_names'][:2])}")
        lines.append(f"funded_by: {', '.join(row['orgs'][:3])}")
        if row["progs"]:
            lines.append(f"programs: {', '.join(row['progs'][:2])}")
        if row["naics"]:
            lines.append(f"naics: {', '.join(row['naics'][:2])}")
        if row["has_bn"]:
            lines.append("has_business_number: yes")
        return "\n".join(lines)

    SYSTEM_PROMPT = """Classify each Canadian grant recipient into exactly one type:
    - F = For-profit business (corporation, company, sole proprietorship)
    - N = Non-profit (society, association, foundation, charity, NGO)
    - P = Person/individual
    - A = Aboriginal/Indigenous organization (First Nation, Métis, Inuit)
    - G = Government or intergovernmental organization
    - S = School, university, college, educational institution
    - O = Other (if none of the above fit)
    - I = International organization

    Return a JSON array of objects with "type" (single letter) and "confidence" (0.0-1.0)."""

    names_list = ctx.to_dicts()
    batches = [names_list[i:i+BATCH_SIZE] for i in range(0, len(names_list), BATCH_SIZE)]
    return SYSTEM_PROMPT, batches, build_prompt_entry


@app.cell
async def _(
    LLM_CACHE_PATH,
    MAX_CONCURRENT,
    MAX_RETRIES,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    asyncio,
    batches,
    build_prompt_entry,
    json,
    mo,
    pl,
    reuse_llm_cache,
    time,
):
    # Token bucket approached used because DeepInfra does not support traditional batch inference. This could be replaced with a batch inference endpoint if needed.
    # Inference run cost $0.7 USD using DeepSeek Flash V4 on DeepInfra

    import aiohttp
    import re

    if reuse_llm_cache.value and LLM_CACHE_PATH.exists():
        cache_df = pl.read_parquet(LLM_CACHE_PATH)
        llm_results = [
            json.loads(row["llm_result_json"]) if row["llm_result_json"] is not None else None
            for row in cache_df.iter_rows(named=True)
        ]
        batch_debug = [
            json.loads(row["batch_debug_json"]) if row["batch_debug_json"] is not None else None
            for row in cache_df.iter_rows(named=True)
        ]
        
        success = sum(1 for r in llm_results if r is not None)
        failed = len(llm_results) - success
        
        mo.md(f"""
        ### LLM results loaded from cache
        
        - **Cache file:** `{LLM_CACHE_PATH}`
        - **Successful batches:** {success:,}
        - **Failed batches:** {failed:,}
        - **API calls skipped:** Yes (reusing cache)
        """)
    else:
        chat_url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"

        def _parse_model_json(text: str):
            """Parse JSON from model response, stripping markdown code fences if present."""
            text = text.strip()
            if text.startswith("```"):
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
                if match:
                    text = match.group(1).strip()
            return json.loads(text)

        async def classify_batch(session, batch, semaphore, results, debug_logs, idx):
            async with semaphore:
                user_content = "\n---\n".join([build_prompt_entry(row) for row in batch])

                payload = {
                    "model": OPENAI_MODEL,
                    "prompt_cache_key": f"{OPENAI_MODEL}-classify-recipient-type",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200,
                }

                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }

                last_error = None
                last_status = None
                last_body = None

                for attempt in range(MAX_RETRIES + 1):
                    try:
                        async with session.post(chat_url, json=payload, headers=headers, timeout=30) as resp:
                            last_status = resp.status
                            body_text = await resp.text()
                            last_body = body_text[:1000]

                            if resp.status == 200:
                                data = json.loads(body_text)
                                content = data["choices"][0]["message"]["content"]
                                parsed = _parse_model_json(content)
                                results[idx] = parsed
                                debug_logs[idx] = {
                                    "status": "success",
                                    "attempts": attempt + 1,
                                }
                                return
                            elif resp.status == 429:
                                await asyncio.sleep(1 * (attempt + 1))
                            else:
                                results[idx] = None
                                debug_logs[idx] = {
                                    "status": "http_error",
                                    "status_code": resp.status,
                                    "body": last_body,
                                    "attempts": attempt + 1,
                                }
                                return
                    except Exception as e:
                        last_error = repr(e)
                        if attempt == MAX_RETRIES:
                            results[idx] = None
                            debug_logs[idx] = {
                                "status": "exception",
                                "error": last_error,
                                "last_status": last_status,
                                "last_body": last_body,
                                "attempts": attempt + 1,
                            }
                        else:
                            await asyncio.sleep(0.5 * (attempt + 1))

        async def run_all_batches():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)
            results = [None] * len(batches)
            debug_logs = [None] * len(batches)

            connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, limit_per_host=MAX_CONCURRENT)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [
                    classify_batch(session, batch, semaphore, results, debug_logs, idx)
                    for idx, batch in enumerate(batches)
                ]

                for coro in asyncio.as_completed(tasks):
                    await coro

            return results, debug_logs

        start_time = time.time()
        llm_results, batch_debug = await run_all_batches()
        elapsed = time.time() - start_time

        cache_df = pl.DataFrame({
            "batch_idx": list(range(len(llm_results))),
            "recipient_legal_name_en": [
                batch[0]["recipient_legal_name_en"] if batch else None
                for batch in batches
            ],
            "llm_result_json": [
                json.dumps(r, ensure_ascii=False) if r is not None else None
                for r in llm_results
            ],
            "batch_debug_json": [
                json.dumps(log, ensure_ascii=False) if log is not None else None
                for log in batch_debug
            ],
        })
        LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache_df.write_parquet(LLM_CACHE_PATH)

        success = sum(1 for r in llm_results if r is not None)
        failed = len(llm_results) - success

        failure_logs = [log for log in batch_debug if log and log.get("status") != "success"]
        failure_preview = ""
        if failure_logs:
            sample = failure_logs[:5]
            failure_preview = f"\n\n**Failure sample ({len(failure_logs)} total):**\n```json\n{json.dumps(sample, indent=2)}\n```"

        mo.md(f"""
        ### LLM classification complete

        - **Successful batches:** {success:,}
        - **Failed batches:** {failed:,}
        - **Time elapsed:** {elapsed:.1f}s
        - **Cache written:** `{LLM_CACHE_PATH}`
        {failure_preview}
        """)
    
    return batch_debug, llm_results


@app.cell
def _(LLM_CACHE_PATH, OPENAI_BASE_URL, OPENAI_MODEL, Path, json, mo, pl):
    from datetime import datetime, timezone

    backup_dir = Path("backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup_path = backup_dir / f"llm_results_{timestamp}.parquet"
    latest_backup_path = backup_dir / "llm_results_latest.parquet"

    backup_df = pl.read_parquet(LLM_CACHE_PATH)

    backup_created_at = datetime.now(timezone.utc).isoformat()
    backup_df = backup_df.with_columns(
        pl.col("llm_result_json").is_not_null().alias("success"),
        pl.lit(backup_created_at).alias("backup_created_at"),
        pl.lit(OPENAI_MODEL).alias("model"),
        pl.lit(OPENAI_BASE_URL).alias("openai_base_url"),
    )

    backup_df.write_parquet(backup_path)
    backup_df.write_parquet(latest_backup_path)

    successful_batches = backup_df["success"].sum()
    failed_batches = (~backup_df["success"]).sum()

    mo.md(f"""
    ### Backup written

    - **Timestamped backup:** `{backup_path}`
    - **Latest backup:** `{latest_backup_path}`
    - **Rows:** `{backup_df.height:,}`
    - **Successful batches:** `{successful_batches:,}`
    - **Failed batches:** `{failed_batches:,}`
    """)
    return


@app.cell
def _(LLM_CACHE_PATH, json, mo, pl):
    _cache_df = pl.read_parquet(LLM_CACHE_PATH)
    _rows = []
    for _row in _cache_df.iter_rows(named=True):
        if _row["llm_result_json"] is not None:
            _parsed = json.loads(_row["llm_result_json"])
            if isinstance(_parsed, dict):
                _parsed = [_parsed]
            for _item in _parsed:
                _rows.append({
                    "recipient_legal_name_en": _row["recipient_legal_name_en"],
                    "llm_type": _item.get("type"),
                    "llm_confidence": _item.get("confidence"),
                })
    llm_classifications = pl.DataFrame(_rows) if _rows else pl.DataFrame(
        schema={"recipient_legal_name_en": pl.Utf8, "llm_type": pl.Utf8, "llm_confidence": pl.Float64}
    )

    if llm_classifications.height > 0:
        _type_dist = (
            llm_classifications
            .group_by("llm_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

        _low_conf = llm_classifications.filter(pl.col("llm_confidence") < 0.7).sort("llm_confidence")

        mo.md("### LLM type distribution")
        mo.ui.table(_type_dist)

        mo.md("### Low confidence classifications (< 0.7)")
        mo.ui.table(_low_conf.head(20))
    return llm_classifications


@app.cell
def _(classified_df, llm_classifications, mo, pl, OUTPUT_PARQUET):
    final_df = classified_df.join(
        llm_classifications.select(["recipient_legal_name_en", "llm_type", "llm_confidence"]),
        on="recipient_legal_name_en",
        how="left",
    )

    final_df = final_df.with_columns(
        pl.when(pl.col("recipient_type_new").is_not_null())
        .then(pl.col("recipient_type_new"))
        .when(pl.col("llm_type").is_not_null())
        .then(pl.col("llm_type"))
        .otherwise(None)
        .alias("recipient_type"),
        pl.when(pl.col("recipient_type_source").is_not_null())
        .then(pl.col("recipient_type_source"))
        .when(pl.col("llm_type").is_not_null())
        .then(pl.lit("llm"))
        .otherwise(None)
        .alias("recipient_type_source"),
        pl.when(pl.col("recipient_type_confidence").is_not_null())
        .then(pl.col("recipient_type_confidence"))
        .when(pl.col("llm_confidence").is_not_null())
        .then(pl.col("llm_confidence"))
        .otherwise(None)
        .alias("recipient_type_confidence"),
    ).drop(["recipient_type_new", "llm_type", "llm_confidence"])

    original_count = final_df.filter(pl.col("recipient_type_source") == "original").height
    deterministic_count = final_df.filter(pl.col("recipient_type_source") == "deterministic").height
    llm_count = final_df.filter(pl.col("recipient_type_source") == "llm").height
    still_null = final_df["recipient_type"].null_count()

    final_df.write_parquet(OUTPUT_PARQUET)

    mo.md(f"""
    ## Final output saved

    - **Original classifications:** {original_count:,}
    - **Deterministic classifications:** {deterministic_count:,}
    - **LLM classifications:** {llm_count:,}
    - **Still null:** {still_null:,}
    - **Total coverage:** {(final_df.height - still_null) / final_df.height * 100:.1f}%
    - **Output file:** `{OUTPUT_PARQUET}`
    """)
    return final_df


if __name__ == "__main__":
    app.run()

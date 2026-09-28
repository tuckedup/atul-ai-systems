# Spider SQLite database acquisition — findings

## Task

`routebench/evalops/data/raw/spider.jsonl` (250 rows) has `db_id`, `question`,
and gold `query` but no SQLite database files. Result-set equivalence
(execute gold SQL, execute candidate SQL, compare rows) is the only sound
oracle for text-to-SQL; a textual/string comparison of SQL was **considered
and explicitly rejected** as unsound (two syntactically different queries can
be semantically equivalent, and two syntactically similar queries can return
different rows — e.g. column order, aliasing, `DESC`/`ASC` defaults, implicit
joins vs explicit `JOIN`, etc.). Without databases there is no way to label
SQL correctness soundly, so obtaining the actual `.sqlite` files was
required before SQL could become a labeled task family.

The 250 rows in spider.jsonl reference only **4 distinct `db_id` values**:
`car_1`, `concert_singer`, `flight_2`, `pets_1`.

## Result: SUCCESS

All 4 required databases were obtained and all 250 gold queries execute
cleanly (read-only) against them.

- **DB files obtained:** 4 / 4 (`car_1.sqlite`, `concert_singer.sqlite`,
  `flight_2.sqlite`, `pets_1.sqlite`)
- **Rows with DB present:** 250 / 250
- **Rows whose gold query executes cleanly (read-only, `mode=ro`):** 250 / 250
  - `car_1`: 92/92
  - `concert_singer`: 45/45
  - `flight_2`: 71/71
  - `pets_1`: 42/42
- **Missing db_ids:** none.

## Source that worked

Hugging Face dataset **`prem-research/spider`**
(https://huggingface.co/datasets/prem-research/spider), tree path
`database/<db_id>/<db_id>.sqlite`. This repo mirrors/repackages the original
Spider release (Yu et al., 2018, Yale LILY lab) including the SQLite
database files that the canonical `xlangai/spider` Hub repo omits (that repo
only ships `train.parquet` / `validation.parquet` with question/SQL pairs,
no databases — confirmed below).

Files downloaded directly via the Hub resolve URL (small, non-LFS files,
plain HTTP GET, no auth needed):

| db_id | URL | bytes | sha256 |
|---|---|---|---|
| car_1 | `https://huggingface.co/datasets/prem-research/spider/resolve/main/database/car_1/car_1.sqlite` | 65536 | `9d851e396e02997a1de073ae982fe1e4b1769fdffce2fac6e325857a3a938709` |
| concert_singer | `https://huggingface.co/datasets/prem-research/spider/resolve/main/database/concert_singer/concert_singer.sqlite` | 36864 | `4fa1ba5ab4577e895271088b1dc44aa94be88e25a54293317a67584112ef059d` |
| flight_2 | `https://huggingface.co/datasets/prem-research/spider/resolve/main/database/flight_2/flight_2.sqlite` | 77824 | `db777ba6488ae2b515cb138d71ff23c4533e5fd0355854648a1102faff3fb7af` |
| pets_1 | `https://huggingface.co/datasets/prem-research/spider/resolve/main/database/pets_1/pets_1.sqlite` | 16384 | `8ffe0ea9b3b034ca8860a2f18300a98c7583ce2522ea195726a04803f675e5bb` |

Total downloaded: ~196 KB (well under the 500 MB cap; no archive/zip was
needed since only 4 of the ~166 Spider databases were required and each
`.sqlite` could be fetched individually).

Each file was confirmed to be a genuine SQLite 3.x database via `file(1)`
(e.g. `SQLite 3.x database, ... database pages 16 ... UTF-8`), then verified
by connecting read-only (`sqlite3.connect(f"file:{path}?mode=ro", uri=True)`)
and executing every gold `query` from spider.jsonl whose `db_id` matches,
fetching all rows without error. See `routebench/evalops/sources/fetch_spider_db.py`,
which re-runs this download + verification and printed:

```
TOTAL: 250/250 rows have a DB present, 250/250 rows execute cleanly
```

**License note:** the `prem-research/spider` repo's own README does not
state an explicit license. The underlying Spider dataset (databases,
questions, and gold SQL) is distributed by its original authors under
CC BY-SA 4.0; this note carries that same attribution/license expectation
forward since the mirror only repackages the original files. `car_1` and
`flight_2` additionally ship an `annotation.json` / `link.txt` noting they
were adapted from existing "csv" data sources, consistent with the original
Spider release notes for those two databases.

## All sources attempted, with exact outcomes

### Hugging Face Hub

1. **`xlangai/spider`** (canonical Spider Hub repo)
   `GET https://huggingface.co/api/datasets/xlangai/spider/tree/main` → HTTP 200.
   Tree contains only a `spider/` subdir with `train-00000-of-00001.parquet`
   (831,359 bytes) and `validation-00000-of-00001.parquet` (125,887 bytes),
   plus README/.gitattributes. **No `database/` folder, no `.sqlite`, no
   zip.** Confirmed: question/SQL pairs only, matching the task's premise
   that this repo lacks the databases.

2. **`richardr1126/spider-schema`** → HTTP 200. Contains only
   `spider_schema_rows_v2.json` (schema metadata, no databases). Not usable.

3. **`spider-nlp/spider`** → HTTP 401 "Invalid username or password" (repo
   does not exist / not publicly resolvable under that id).

4. **`premai-io/spider`** → HTTP 307 redirect to `prem-research/spider`.
   Followed the redirect — **this is the repo that worked** (see above).

5. **`hkunlp/spider`** → HTTP 401 "Invalid username or password" (not
   publicly resolvable under that id / requires auth not attempted).

6. **`Sayanc/spider_databases`** → HTTP 401, same as above.

7. **`Rahmaa/spider`** → HTTP 401, same as above.

8. **`lilaceclipse/spider`** → HTTP 401, same as above.

9. Hub dataset search `?search=spider%20database&limit=50` → surfaced
   `HAL-9001/spider-databases` (`https://huggingface.co/datasets/HAL-9001/spider-databases`,
   license `cc-by-sa-4.0`), tree = single `spider_data.zip`
   (205,800,266 bytes, LFS). This is a full 166-database rehost packaged as a
   zip. **Not downloaded** because it was unnecessary once
   `prem-research/spider` was found to already expose the 4 needed
   `.sqlite` files individually (no zip extraction required, far less
   bandwidth). Left as a documented fallback if more db_ids are ever needed
   for a larger Spider subset than the 250-row/4-db_id slice used here.

10. Hub dataset search `?search=spider%20text2sql&limit=50` — not queried
    separately since search #9 above (`spider database`) already surfaced a
    working full-database mirror and the primary source had already
    succeeded by that point.

### GitHub

Not needed / not attempted — the Hugging Face Hub search above (`prem-research/spider`
via the `premai-io/spider` redirect) succeeded first and provided exactly
the 4 required `.sqlite` files, satisfying the success criterion (250/250
rows executable) without needing to fall back to GitHub mirrors
(`taoyds/test-suite-sql-eval`, `taoyds/spider`, `ElementAI/duorat`,
`microsoft/rat-sql`, `salesforce/photon`).

## Reproducing

```
.venv/Scripts/python.exe routebench/evalops/sources/fetch_spider_db.py
```

This downloads (or skips if already present) each `<db_id>.sqlite` referenced
by `spider.jsonl` into `routebench/evalops/data/raw/spider_db/`, then opens
every relevant database read-only and executes every matching gold query,
printing a final `N/M rows execute cleanly` summary.

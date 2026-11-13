"""Build the four tables required by the existing notebook from public data.

Run with ../.venv/bin/python build_tidy_finance.py. Existing raw downloads are
reused. A populated database is never overwritten without --replace.
Compatibility choices are documented in data_sources/数据来源与兼容说明.txt.
"""

import argparse
import hashlib
import io
import json
import re
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_sources"
SOURCES = {
    "F-F_Research_Data_Factors_CSV.zip": [
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    ],
    "10_Industry_Portfolios_CSV.zip": [
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Industry_Portfolios_CSV.zip"
    ],
    "q5_factors_monthly_2022.csv": [
        "https://global-q.org/uploads/1/2/2/6/122679606/q5_factors_monthly_2022.csv",
        "https://global-q.weebly.com/uploads/1/2/2/6/122679606/q5_factors_monthly_2022.csv",
    ],
    "goyal_welch_monthly.csv": [
        "https://docs.google.com/spreadsheets/d/1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG/gviz/tq?tqx=out:csv&sheet=Monthly"
    ],
}


def download_missing():
    RAW.mkdir(exist_ok=True)
    for name, urls in SOURCES.items():
        path = RAW / name
        if path.is_file():
            continue
        errors = []
        for url in urls:
            try:
                response = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0", "Referer": "https://global-q.org/factors.html"},
                    timeout=60,
                )
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError("The download returned HTML instead of data")
                path.write_bytes(response.content)
                break
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError("\n".join(errors))


def french_monthly(filename):
    # First monthly block: FF three factors / value-weighted industry returns.
    with zipfile.ZipFile(RAW / filename) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        lines = archive.read(member).decode("utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(r"^\s*\d{6}\s*,", line))
    stop = start
    while stop < len(lines) and re.match(r"^\s*\d{6}\s*,", lines[stop]):
        stop += 1
    frame = pd.read_csv(io.StringIO("\n".join(["date" + lines[start - 1].strip(), *lines[start:stop]])))
    frame.columns = [name.strip().lower() for name in frame.columns]
    frame = frame.rename(columns={"mkt-rf": "mkt_excess"})
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m")
    values = frame.columns.drop("date")
    frame[values] = frame[values].replace([-99.99, -999.0], np.nan) / 100.0
    return frame


def build_tables():
    ff = french_monthly("F-F_Research_Data_Factors_CSV.zip")
    industries = french_monthly("10_Industry_Portfolios_CSV.zip")
    q = pd.read_csv(RAW / "q5_factors_monthly_2022.csv")
    q["date"] = pd.to_datetime(dict(year=q["year"], month=q["month"], day=1))
    # Preserve the observed original schema, including year and R_F.
    # Q returns were stored in percent in the original output; StandardScaler
    # handles their different scale. These choices are not a new model design.
    q = q.drop(columns="month").rename(columns=lambda name: name.replace("R_", "").lower())
    q = q[["date", "year", "f", "mkt", "me", "ia", "roe", "eg"]]

    raw = pd.read_csv(RAW / "goyal_welch_monthly.csv", thousands=",")
    raw = raw.loc[raw["yyyymm"] <= 202212].copy()
    macro = raw.assign(
        date=pd.to_datetime(raw["yyyymm"].astype(str), format="%Y%m"),
        rp_div=(np.log(raw["Index"] + raw["D12"]).diff() - np.log1p(raw["Rfree"])).shift(-1),
        dp=np.log(raw["D12"]) - np.log(raw["Index"]),
        # This matches all ten visible historical rows in the supplied original
        # notebook. It is dividend growth, not the standard log dividend yield.
        dy=np.log(raw["D12"]).diff(),
        ep=np.log(raw["E12"]) - np.log(raw["Index"]),
        de=np.log(raw["D12"]) - np.log(raw["E12"]),
        tms=raw["lty"] - raw["tbl"],
        dfy=raw["BAA"] - raw["AAA"],
    )[["date", "rp_div", "dp", "dy", "ep", "de", "svar", "b/m", "ntis", "tbl", "lty", "ltr", "tms", "dfy", "infl"]]

    tables = {
        "factors_ff3_monthly": ff,
        "factors_q_monthly": q,
        "macro_predictors": macro,
        "industries_ff_monthly": industries,
    }
    for name, frame in tables.items():
        frame = frame.loc[frame["date"].between("1960-01-01", "2022-12-01")].dropna().copy()
        assert not frame["date"].duplicated().any(), name
        assert np.isfinite(frame.drop(columns="date").to_numpy()).all(), name
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        tables[name] = frame.sort_values("date").reset_index(drop=True)

    common = set.intersection(*(set(frame["date"]) for frame in tables.values()))
    expected = set(pd.date_range("1967-01-01", "2022-11-01", freq="MS").strftime("%Y-%m-%d"))
    assert common == expected, "Unexpected sample coverage"
    assert len(industries.columns) == 11
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Replace an existing database")
    args = parser.parse_args()
    target = ROOT / "tidy_finance.sqlite"
    if target.exists() and not args.replace:
        raise FileExistsError(f"{target} already exists; use --replace to rebuild")
    download_missing()
    tables = build_tables()
    temporary = ROOT / "tidy_finance.building.sqlite"
    if temporary.exists():
        raise FileExistsError(temporary)
    with sqlite3.connect(temporary) as conn:
        for name, frame in tables.items():
            frame.to_sql(name, conn, index=False)
            conn.execute(f'CREATE UNIQUE INDEX "{name}_date" ON "{name}" (date)')
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    temporary.replace(target)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Reconstructed from public downloads; not the missing original SQLite file",
        "sources": [
            {"file": name, "source_urls": urls, "sha256": hashlib.sha256((RAW / name).read_bytes()).hexdigest()}
            for name, urls in SOURCES.items()
        ],
        "tables": {
            name: {"rows": len(frame), "columns": list(frame.columns), "start": frame["date"].min(), "end": frame["date"].max()}
            for name, frame in tables.items()
        },
        "merged_sample": {"start": "1967-01-01", "end": "2022-11-01", "months": 671, "industries": 10, "rows": 6710, "features": 153},
        "compatibility_notes": [
            "FF factors and value-weighted industry total returns use the downloaded 202609 CRSP vintage, scaled from percent to decimal.",
            "Q factors use the 2022 release. Preserve year and f and percentage units as observed in the original notebook output.",
            "Macro raw CSV is PredictorData2023, available through the Tidy Finance reference link. Raw input is truncated at 202212 before the one-month lead target is formed.",
            "The macro column dy reproduces the observed original dividend log growth (diff(log(D12))); it is not the standard dividend-yield definition.",
            "rp_div is included for schema compatibility and dropped by the notebook before modeling; its original formula cannot be verified from stored output.",
            "Source revisions mean historical notebook metrics are not guaranteed to reproduce. No notebook content or outputs are overwritten.",
        ],
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for name, info in manifest["tables"].items():
        print(name, info["rows"], info["start"], info["end"])
    print(f"Created {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

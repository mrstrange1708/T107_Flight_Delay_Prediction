"""Phase 2 data collection + labelling + Phase 5 splitting.
Owner: Rishik."""
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_RAW, DELAY_THRESHOLD_MIN, SPLITS, WARMUP_END


def download_raw(force: bool = False) -> Path:
    """Get the NYC 2013 flights tables (the Kaggle 'Flights' data) into data/raw/.

    The data ship inside the `nycflights13` package source on PyPI. We download the source
    archive with pip and extract the CSVs directly, which avoids the package's broken installer
    on new Python versions. If you already have Kaggle's flights.csv, put it in data/raw/.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if (DATA_RAW / "flights.csv").exists() and (DATA_RAW / "planes.csv").exists() and not force:
        return DATA_RAW
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "download", "nycflights13==0.0.3", "--no-deps",
                        "--no-binary", ":all:", "-d", tmp, "-q"], check=True)
        archive = next(Path(tmp).glob("nycflights13-*.tar.gz"))
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                name = Path(m.name).name
                if "/data/" in m.name and name in {"flights.csv.zip", "planes.csv", "airlines.csv", "airports.csv", "weather.csv"}:
                    with tar.extractfile(m) as src, open(DATA_RAW / name, "wb") as dst:
                        dst.write(src.read())
    with zipfile.ZipFile(DATA_RAW / "flights.csv.zip") as z:
        inner = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__"))
        with z.open(inner) as src, open(DATA_RAW / "flights.csv", "wb") as dst:
            dst.write(src.read())
    (DATA_RAW / "flights.csv.zip").unlink()
    return DATA_RAW


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    download_raw()
    flights = pd.read_csv(DATA_RAW / "flights.csv")
    flights = flights.drop(columns=[c for c in ("Unnamed: 0", "id", "name") if c in flights.columns])
    planes = pd.read_csv(DATA_RAW / "planes.csv")
    return flights, planes


def hhmm_to_minutes(x: pd.Series) -> pd.Series:
    x = x.astype(int) % 2400
    return (x // 100) * 60 + x % 100


def build_dataset(flights: pd.DataFrame, planes: pd.DataFrame) -> pd.DataFrame:
    """One row per scheduled flight, with the label and static aircraft attributes attached."""
    df = flights.copy()
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["sched_dep_min"] = hhmm_to_minutes(df["sched_dep_time"])
    df["sched_arr_min"] = hhmm_to_minutes(df["sched_arr_time"])
    df["sched_hour"] = df["sched_dep_min"] // 60
    df["arr_hour"] = df["sched_arr_min"] // 60
    # Cancelled flights never departed; diverted flights departed but never reached the destination.
    df["cancelled"] = df["dep_time"].isna()
    df["diverted"] = df["dep_time"].notna() & df["arr_delay"].isna()
    # Label: 1 if arrival delay >= 15 min, NaN where no arrival delay exists (cancelled/diverted).
    df["delayed"] = np.where(df["arr_delay"].isna(), np.nan, (df["arr_delay"] >= DELAY_THRESHOLD_MIN).astype(float))
    p = planes[["tailnum", "year", "seats", "engines"]].rename(
        columns={"year": "plane_year", "seats": "plane_seats", "engines": "plane_engines"})
    df = df.merge(p, on="tailnum", how="left")
    df["plane_age"] = 2013 - df["plane_year"]
    df["tailnum"] = df["tailnum"].fillna("UNKNOWN")
    return df.sort_values(["date", "sched_dep_min", "origin", "carrier", "flight"]).reset_index(drop=True)


def split(df: pd.DataFrame, labelled_only: bool = True) -> dict:
    """Phase 5: chronological split. Returns {'train','valid','calib','test'} frames."""
    d = df[df["delayed"].notna()] if labelled_only else df
    return {k: d[(d["date"] >= a) & (d["date"] <= b)] for k, (a, b) in SPLITS.items()}

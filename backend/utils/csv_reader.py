from __future__ import annotations

import io

import pandas as pd


def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    if contents.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ["utf-16"]
    else:
        encodings = ["utf-8-sig", "cp1252", "latin1"]

    for encoding in encodings:
        try:
            text = contents.decode(encoding)
        except UnicodeDecodeError:
            continue
        return pd.read_csv(io.StringIO(text))

    raise UnicodeDecodeError(
        "csv",
        contents,
        0,
        len(contents),
        "Could not decode CSV using UTF-8, Windows-1252, or Latin-1",
    )

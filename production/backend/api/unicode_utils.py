import re
import pandas as pd

_ESCAPE_RE = re.compile(r'\\u([0-9a-fA-F]{4})')

def fix_unicode(s: str) -> str:
    """Decode literal \\uXXXX sequences stored as plain text in Wyscout data."""
    if not isinstance(s, str):
        return s
    return _ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), s)

def fix_df_unicode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply fix_unicode to all string columns of a DataFrame in-place."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda v: fix_unicode(v) if isinstance(v, str) else v)
    return df

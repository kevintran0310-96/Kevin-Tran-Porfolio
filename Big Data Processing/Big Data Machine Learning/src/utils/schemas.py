import json
from pyspark.sql.types import StructType

def load_schema(json_path: str, table: str) -> StructType | None:
    try:
        with open(json_path) as f:
            cfg = json.load(f)
        if not cfg.get("use_explicit_schema"):
            return None
        # TODO: build StructType from cfg['tables'][table]
        return None
    except Exception:
        return None

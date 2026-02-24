import json
from pathlib import Path

import requests
import pandas as pd

BASE_API = "https://data.stats.gov.cn/easyquery.htm"
BASE_EN  = "https://data.stats.gov.cn/english/easyquery.htm"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15")

def post_json(sess, data, referer):
    r = sess.post(
        BASE_API,
        data=data,
        headers={
            "User-Agent": UA,
            "Referer": referer,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=60,
    )
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        raise RuntimeError("Not JSON. First 300 chars:\n" + r.text[:300])

def extract_tree(j):
    # Case 1: list returned directly
    if isinstance(j, list):
        return j
    # Case 2: dict with returndata
    if isinstance(j, dict):
        if "returndata" in j:
            return j["returndata"]
        # Sometimes nested
        if "data" in j and isinstance(j["data"], list):
            return j["data"]
        raise RuntimeError("Dict JSON without returndata. Keys=" + ",".join(j.keys()))
    raise RuntimeError("Unexpected JSON type: " + str(type(j)))

def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    sess = requests.Session()

    # Try a few cn/dbcode combos; SAVE first success
    cn_candidates = ["E0102", "E0101", "E0103", "C01"]  # include a Chinese default as fallback
    db_candidates = ["fsnd", "hgnd", "csyd"]            # add city-year fallback

    for cn in cn_candidates:
        # boot page for cookies
        boot = sess.get(f"{BASE_EN}?cn={cn}", headers={"User-Agent": UA}, timeout=30)
        if boot.status_code != 200:
            continue
        referer = f"{BASE_EN}?cn={cn}"

        for db in db_candidates:
            try:
                print(f"Trying: cn={cn}, db={db} -> getTree zb")
                j_zb = post_json(sess, {"m":"getTree","dbcode":db,"id":"zb","wdcode":"zb","cn":cn}, referer)
                tree_zb = extract_tree(j_zb)
                df_zb = pd.DataFrame(tree_zb)
                if df_zb.empty:
                    raise RuntimeError("zb tree empty")

                print(f"Trying: cn={cn}, db={db} -> getTree reg")
                j_reg = post_json(sess, {"m":"getTree","dbcode":db,"id":"zb","wdcode":"reg","cn":cn}, referer)
                tree_reg = extract_tree(j_reg)
                df_reg = pd.DataFrame(tree_reg)
                if df_reg.empty:
                    raise RuntimeError("reg tree empty")

                df_zb.to_csv(out_dir / f"tree_{db}_zb.csv", index=False, encoding="utf-8-sig")
                df_reg.to_csv(out_dir / f"tree_{db}_reg.csv", index=False, encoding="utf-8-sig")
                print("\nSUCCESS.")
                print("Saved:", out_dir / f"tree_{db}_zb.csv")
                print("Saved:", out_dir / f"tree_{db}_reg.csv")
                print(f"Use these with dbcode={db}, cn={cn} in the downloader.\n")
                return

            except Exception as e:
                print("  failed:", e)

    raise RuntimeError("Failed for all cn/db combinations. (Likely blocked / requires different endpoint.)")

if __name__ == "__main__":
    main()
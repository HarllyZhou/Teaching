import json
import re
import time
from pathlib import Path

import requests
import pandas as pd

BASE_API = "https://data.stats.gov.cn/easyquery.htm"
BASE_EN  = "https://data.stats.gov.cn/english/easyquery.htm"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)

def preview(s: str, n: int = 300) -> str:
    return s[:n] + ("..." if len(s) > n else "")

def post_json(sess: requests.Session, data: dict, referer: str):
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
        raise RuntimeError("Not JSON. First chars:\n" + preview(r.text))

def extract_tree_payload(j):
    # getTree sometimes returns a list directly, or a dict with 'returndata'
    if isinstance(j, list):
        return j
    if isinstance(j, dict) and "returndata" in j:
        return j["returndata"]
    if isinstance(j, dict):
        raise RuntimeError("Unexpected dict keys: " + ",".join(j.keys()))
    raise RuntimeError("Unexpected JSON type: " + str(type(j)))

def boot(sess: requests.Session, cn: str):
    url = f"{BASE_EN}?cn={cn}"
    r = sess.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()

def get_zb_tree(sess: requests.Session, cn: str, dbcode: str):
    referer = f"{BASE_EN}?cn={cn}"
    j = post_json(sess, {"m":"getTree","dbcode":dbcode,"id":"zb","wdcode":"zb","cn":cn}, referer)
    payload = extract_tree_payload(j)
    df = pd.DataFrame(payload)
    if df.empty:
        raise RuntimeError("zb tree empty")
    return df

def mk_dfwds(wdcode: str, valuecode: str) -> str:
    return json.dumps([{"wdcode": wdcode, "valuecode": valuecode}], ensure_ascii=False)

def querydata_raw(sess: requests.Session, cn: str, dbcode: str, zb_code: str):
    referer = f"{BASE_EN}?cn={cn}"
    k1 = str(int(time.time() * 1000))
    j = post_json(sess, {
        "m":"QueryData",
        "dbcode":dbcode,
        "rowcode":"reg",
        "colcode":"sj",
        "dfwds": mk_dfwds("zb", zb_code),
        "k1": k1,
        "cn": cn
    }, referer)
    if not isinstance(j, dict) or "returndata" not in j:
        raise RuntimeError("QueryData unexpected structure. Preview:\n" + preview(json.dumps(j, ensure_ascii=False)))
    return j["returndata"]

def extract_reg_map_from_querydata(returndata: dict) -> pd.DataFrame:
    """
    In many StatChina QueryData responses, province codes+names live in returndata['wdnodes']
    """
    wdnodes = returndata.get("wdnodes", None)
    if wdnodes is None:
        raise RuntimeError("No wdnodes in QueryData returndata (cannot infer reg mapping).")

    # wdnodes is usually a list of dimensions; find the reg dimension
    reg_dim = None
    for dim in wdnodes:
        if dim.get("wdcode") == "reg":
            reg_dim = dim
            break
    if reg_dim is None:
        raise RuntimeError("wdnodes present but no reg dimension found.")

    nodes = reg_dim.get("nodes", None)
    if nodes is None:
        raise RuntimeError("reg dimension has no nodes.")

    df = pd.DataFrame(nodes)
    # typical columns: code, cname, name
    # normalize to reg/prov
    if "code" in df.columns:
        df = df.rename(columns={"code":"reg"})
    elif "id" in df.columns:
        df = df.rename(columns={"id":"reg"})
    else:
        raise RuntimeError("reg nodes have no code/id column. Columns=" + ",".join(df.columns))

    if "cname" in df.columns:
        df["prov"] = df["cname"]
    elif "name" in df.columns:
        df["prov"] = df["name"]
    else:
        # fallback: keep whatever text col exists
        text_cols = [c for c in df.columns if "name" in c.lower()]
        if not text_cols:
            raise RuntimeError("Cannot find province name column in reg nodes.")
        df["prov"] = df[text_cols[0]]

    return df[["reg","prov"]].drop_duplicates().sort_values("reg")

def pick_sample_leaf_zb(tree_zb: pd.DataFrame) -> str:
    """
    Pick one indicator id with isParent==False to use as a sample QueryData pull.
    """
    if "isParent" in tree_zb.columns:
        leaf = tree_zb[tree_zb["isParent"] == False]
        if not leaf.empty:
            return str(leaf.iloc[0]["id"])
    # fallback: just take first id
    return str(tree_zb.iloc[0]["id"])

def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    sess = requests.Session()

    cn_candidates = ["E0102", "E0101", "E0103", "C01"]
    db_candidates = ["fsnd", "csyd"]  # hgnd in your logs gave 404; skip it

    chosen = None
    tree_zb = None

    for cn in cn_candidates:
        try:
            boot(sess, cn)
        except Exception:
            continue

        for db in db_candidates:
            try:
                print(f"Trying zb tree: cn={cn}, db={db}")
                tree_zb = get_zb_tree(sess, cn, db)
                chosen = (cn, db)
                break
            except Exception as e:
                print("  failed:", e)

        if chosen:
            break

    if not chosen:
        raise RuntimeError("Could not fetch zb tree in tried cn/db combinations.")

    cn, db = chosen
    print(f"\nSUCCESS zb tree: cn={cn}, db={db}")
    tree_zb_path = out_dir / f"tree_{db}_zb.csv"
    tree_zb.to_csv(tree_zb_path, index=False, encoding="utf-8-sig")
    print("Saved:", tree_zb_path)

    # Find a sample leaf indicator and use it to infer province mapping via QueryData metadata
    sample_zb = pick_sample_leaf_zb(tree_zb)
    print("Using sample leaf zb for region mapping:", sample_zb)

    rd = querydata_raw(sess, cn, db, sample_zb)
    reg_map = extract_reg_map_from_querydata(rd)
    reg_map_path = out_dir / f"regmap_{db}.csv"
    reg_map.to_csv(reg_map_path, index=False, encoding="utf-8-sig")
    print("Saved:", reg_map_path)

    # Show quick keyword matches so you can find real zb codes fast
    if all(c in tree_zb.columns for c in ["id","name"]):
        def show(pat):
            hits = tree_zb[tree_zb["name"].astype(str).str.contains(pat, na=False)][["id","name"]].head(20)
            print(f"\nMatches for: {pat}\n{hits.to_string(index=False)}")

        for pat in ["一般公共预算收入", "税收收入", "非税收入", "政府性基金收入", "土地", "出让"]:
            show(pat)

    print("\nNext:")
    print("1) Open data/tree_{}_zb.csv and copy the right 'id' as zb codes.".format(db))
    print("2) Tell me (or upload the CSV) and I’ll give you the exact zb_list for fiscal variables.")

if __name__ == "__main__":
    main()
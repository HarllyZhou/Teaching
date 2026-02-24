import json
import re
import time
from pathlib import Path

import requests

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas not installed. Run: python3 -m pip install pandas requests")


BASE_API = "https://data.stats.gov.cn/easyquery.htm"
BASE_EN  = "https://data.stats.gov.cn/english/easyquery.htm"


UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def preview_text(s: str, n: int = 400) -> str:
    s = s.replace("\r", "")
    return s[:n] + ("..." if len(s) > n else "")


def boot_session(session: requests.Session, cn: str) -> None:
    url = f"{BASE_EN}?cn={cn}"
    r = session.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()


def post_json(session: requests.Session, data: dict, referer: str):
    headers = {
        "User-Agent": UA,
        "Referer": referer,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }
    r = session.post(BASE_API, data=data, headers=headers, timeout=60)
    r.raise_for_status()
    txt = r.text
    try:
        return r.json()
    except Exception:
        raise RuntimeError(
            f"JSON parse failed. Content-Type={r.headers.get('Content-Type','')}\n"
            f"Response begins:\n{preview_text(txt)}"
        )


def normalize_tree_json(j):
    """
    getTree may return:
    - dict with 'returndata'
    - list of nodes directly
    """
    if isinstance(j, list):
        return j
    if isinstance(j, dict):
        if "returndata" in j:
            return j["returndata"]
        # sometimes the payload is under different key names
        for k in ["data", "result", "datas"]:
            if k in j:
                return j[k]
        raise RuntimeError(f"Dict JSON but no returndata-like field. Keys={list(j.keys())}")
    raise RuntimeError(f"Unexpected JSON type: {type(j)}")


def normalize_query_json(j):
    """
    QueryData usually returns dict with returndata/datanodes, but be defensive.
    """
    if isinstance(j, dict) and "returndata" in j:
        return j["returndata"]
    # sometimes it may return a list (rare); dump for debugging
    raise RuntimeError(f"Unexpected QueryData JSON structure: {type(j)}")


def get_tree(session: requests.Session, dbcode: str, wdcode: str, cn: str, id_: str = "zb"):
    referer = f"{BASE_EN}?cn={cn}"
    payload = {"m": "getTree", "dbcode": dbcode, "id": id_, "wdcode": wdcode, "cn": cn}
    j = post_json(session, payload, referer=referer)
    return normalize_tree_json(j)


def find_in_tree(tree_df, pattern, n=30):
    hits = tree_df[tree_df["name"].astype(str).str.contains(pattern, na=False)]
    cols = [c for c in ["id", "name", "pid", "isParent"] if c in hits.columns]
    return hits[cols].head(n)


def mk_dfwds(wdcode: str, valuecode: str) -> str:
    return json.dumps([{"wdcode": wdcode, "valuecode": valuecode}], ensure_ascii=False)


def query_data(session: requests.Session, dbcode: str, zb_code: str, cn: str):
    referer = f"{BASE_EN}?cn={cn}"
    k1 = str(int(time.time() * 1000))
    payload = {
        "m": "QueryData",
        "dbcode": dbcode,
        "rowcode": "reg",
        "colcode": "sj",
        "dfwds": mk_dfwds("zb", zb_code),
        "k1": k1,
        "cn": cn,
    }
    j = post_json(session, payload, referer=referer)
    rd = normalize_query_json(j)
    nodes = rd.get("datanodes", None)
    if nodes is None:
        raw = json.dumps(j, ensure_ascii=False, indent=2)
        raise RuntimeError("No datanodes. Raw begins:\n" + preview_text(raw))

    rows = []
    for node in nodes:
        wds = node.get("wds", "")
        data_obj = node.get("data", {})
        val = data_obj.get("data", None)

        m_zb  = re.search(r"zb\.([^_]+)", wds)
        m_reg = re.search(r"reg\.([^_]+)", wds)
        m_sj  = re.search(r"sj\.(\d{4})", wds)

        rows.append({
            "reg": m_reg.group(1) if m_reg else None,
            "year": int(m_sj.group(1)) if m_sj else None,
            "value": float(val) if val not in (None, "") else None,
            "zb": m_zb.group(1) if m_zb else None
        })

    import pandas as pd
    df = pd.DataFrame(rows).dropna(subset=["reg", "year"])
    return df


def main():
    out_dir = Path("data")
    ensure_dir(out_dir)

    cn_candidates = ["E0102", "E0101", "E0103"]
    db_candidates = ["fsnd", "hgnd"]

    sess = requests.Session()

    chosen = None
    tree_zb = tree_reg = None

    for cn in cn_candidates:
        try:
            boot_session(sess, cn=cn)
        except Exception as e:
            print(f"[boot failed] cn={cn}: {e}")
            continue

        for db in db_candidates:
            try:
                print(f"Trying getTree zb: cn={cn}, db={db}")
                zb_raw = get_tree(sess, dbcode=db, wdcode="zb", cn=cn, id_="zb")
                print(f"Trying getTree reg: cn={cn}, db={db}")
                reg_raw = get_tree(sess, dbcode=db, wdcode="reg", cn=cn, id_="zb")

                import pandas as pd
                tree_zb = pd.DataFrame(zb_raw)
                tree_reg = pd.DataFrame(reg_raw)

                if tree_zb.empty or tree_reg.empty:
                    raise RuntimeError("Tree returned but empty.")

                chosen = (cn, db)
                break
            except Exception as e:
                print(f"[getTree failed] cn={cn}, db={db}: {e}")

        if chosen:
            break

    if not chosen:
        raise RuntimeError("Could not retrieve trees under tried cn/db combinations.")

    cn, db = chosen
    print(f"\nSUCCESS: cn={cn}, dbcode={db}\n")

    tree_zb_path = out_dir / f"tree_{db}_zb.csv"
    tree_reg_path = out_dir / f"tree_{db}_reg.csv"
    tree_zb.to_csv(tree_zb_path, index=False, encoding="utf-8-sig")
    tree_reg.to_csv(tree_reg_path, index=False, encoding="utf-8-sig")
    print(f"Saved {tree_zb_path}")
    print(f"Saved {tree_reg_path}")

    print("\n--- Candidate matches (inspect 'id') ---")
    for pat in ["一般公共预算收入", "税收收入", "非税收入", "政府性基金收入", "土地", "出让"]:
        hits = find_in_tree(tree_zb, pat, n=15)
        print(f"\nPattern: {pat}")
        print(hits.to_string(index=False))

    print("\nNext step:")
    print("1) Open the saved tree CSV and copy the exact 'id' for the series you want.")
    print("2) Paste those ids into zb_list below and re-run to download the panel.\n")

    # -------- SET THESE AFTER YOU FIND THE REAL CODES --------
    zb_list = {
        "gpb_rev_total": None,
        "gpb_tax_rev": None,
        "gpb_nontax_rev": None,
        "govfund_rev": None,
        "land_convey_rev": None
    }

    if any(v is None for v in zb_list.values()):
        print("STOP: zb_list not set yet (all None).")
        return

    # Download chosen series and merge
    import pandas as pd
    panel = None
    for name, zb in zb_list.items():
        print(f"Downloading {name} (zb={zb})...")
        df = query_data(sess, dbcode=db, zb_code=zb, cn=cn)
        df = df.rename(columns={"value": name}).drop(columns=["zb"])
        panel = df if panel is None else panel.merge(df, on=["reg", "year"], how="outer")

    reg_map = tree_reg.rename(columns={"id": "reg", "name": "prov"})[["reg", "prov"]]
    panel = panel.merge(reg_map, on="reg", how="left")

    out_panel = out_dir / "panel_province_year.csv"
    panel.to_csv(out_panel, index=False, encoding="utf-8-sig")
    print(f"Saved {out_panel}")


if __name__ == "__main__":
    main()
import json
import time
import requests
import pandas as pd

API_CANDIDATES = [
    "https://data.stats.gov.cn/easyquery.htm",
    "https://data.stats.gov.cn/english/easyquery.htm",
    "https://www.stats.gov.cn/easyquery.htm",
    "https://www.stats.gov.cn/english/easyquery.htm",
]

BASE_EN = "https://data.stats.gov.cn/english/easyquery.htm"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15")

def mk_dfwds(wdcode: str, valuecode: str) -> str:
    return json.dumps([{"wdcode": wdcode, "valuecode": valuecode}], ensure_ascii=False)

def post_json_any(sess: requests.Session, payload: dict, referer: str):
    last_err = None
    for api in API_CANDIDATES:
        try:
            r = sess.post(
                api,
                data=payload,
                headers={
                    "User-Agent": UA,
                    "Referer": referer,
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=60,
            )
            # if 404, try next
            if r.status_code == 404:
                last_err = RuntimeError(f"404 on {api}")
                continue
            r.raise_for_status()
            return r.json(), api
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All API endpoints failed. Last error: {last_err}")

def querydata(sess: requests.Session, cn: str, dbcode: str, zb_code: str):
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
    j, api_used = post_json_any(sess, payload, referer)
    if "returndata" not in j or "datanodes" not in j["returndata"]:
        raise RuntimeError("QueryData returned unexpected JSON structure. api_used=" + api_used)
    print("QueryData succeeded using:", api_used)
    return j["returndata"]

if __name__ == "__main__":
    sess = requests.Session()

    # bootstrap cookies
    cn = "E0102"
    boot = sess.get(f"{BASE_EN}?cn={cn}", headers={"User-Agent": UA}, timeout=30)
    boot.raise_for_status()

    # Use a REAL leaf zb from your tree file, not "A0H"
    # For testing just plug in one id you see in tree_fsnd_zb.csv
    zb_test = "A080101"  # <-- REPLACE with an actual leaf id from your saved tree

    rd = querydata(sess, cn=cn, dbcode="fsnd", zb_code=zb_test)
    print("Keys in returndata:", rd.keys())
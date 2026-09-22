#!/usr/bin/env python3
"""Generate synthetic panel data for fixed-effects illustration."""

import csv
import random
from pathlib import Path

random.seed(4999)

REGIONS = [chr(ord("A") + i) for i in range(26)]  # A through Z
YEARS = list(range(2000, 2027))

# Time-invariant region characteristics
# Coastal / high-productivity regions: higher min wage, lower unemployment
region_effects = {}
base_min_wage = {}
for i, region in enumerate(REGIONS):
    if i < 13:
        region_effects[region] = random.uniform(3.0, 5.0)   # higher unemployment
        base_min_wage[region] = random.uniform(6.5, 8.5)    # lower min wage
    else:
        region_effects[region] = random.uniform(1.5, 3.5)   # lower unemployment
        base_min_wage[region] = random.uniform(9.0, 11.5)   # higher min wage

# National minimum-wage trend (common time component)
year_trend = {year: 0.12 * (year - 2000) for year in YEARS}

# Macro shocks (year fixed effects on unemployment)
year_shocks = {
    2001: 0.8,
    2008: 1.2,
    2009: 2.5,
    2020: 3.0,
    2021: 1.5,
}

rows = []
for region in REGIONS:
    for year in YEARS:
        min_wage = (
            base_min_wage[region]
            + year_trend[year]
            + random.gauss(0, 0.15)
        )
        min_wage = round(max(5.15, min_wage), 2)

        # Within-region effect: +0.25 pp unemployment per $1 min-wage increase
        within_effect = 0.25 * (min_wage - base_min_wage[region] - year_trend[year])

        unemployment = (
            region_effects[region]
            + year_shocks.get(year, 0.0)
            + within_effect
            + random.gauss(0, 0.35)
        )
        unemployment = round(max(1.5, min(unemployment, 18.0)), 2)

        rows.append(
            {
                "region": region,
                "year": year,
                "min_wage": min_wage,
                "unemployment_rate": unemployment,
            }
        )

out_path = Path(__file__).with_name("panel_minwage_unemployment.csv")
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["region", "year", "min_wage", "unemployment_rate"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} observations to {out_path}")

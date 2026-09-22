#!/usr/bin/env python3
"""Estimate min-wage effects on unemployment under four FE specifications."""

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = Path(__file__).with_name("panel_minwage_unemployment.csv")
TEX_PATH = Path(__file__).with_name("fixed_effects_results.tex")

SPECS = [
    ("(1) No fixed effects", "unemployment_rate ~ min_wage"),
    ("(2) Year fixed effects", "unemployment_rate ~ min_wage + C(year)"),
    ("(3) Region fixed effects", "unemployment_rate ~ min_wage + C(region)"),
    (
        "(4) Region and year fixed effects",
        "unemployment_rate ~ min_wage + C(region) + C(year)",
    ),
]


def fit_model(formula: str, data: pd.DataFrame):
    model = smf.ols(formula, data=data)
    return model.fit(cov_type="HC1")


def stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def coef_cell(result) -> tuple[str, str]:
    coef = result.params["min_wage"]
    se = result.bse["min_wage"]
    pval = result.pvalues["min_wage"]
    return f"{coef:0.3f}{stars(pval)}", f"({se:0.3f})"


def write_tex(results: list[tuple[str, object]]) -> None:
    coefs, ses = zip(*(coef_cell(res) for _, res in results))
    nobs = int(results[0][1].nobs)
    r2 = " & ".join(f"{res.rsquared:0.3f}" for _, res in results)
    adj_r2 = " & ".join(f"{res.rsquared_adj:0.3f}" for _, res in results)

    tex = """\\documentclass[12pt]{article}

\\usepackage[margin=1in]{geometry}
\\usepackage{booktabs}
\\usepackage{amsmath}
\\usepackage{setspace}
\\onehalfspacing

\\title{Fixed Effects Illustration:\\\\Minimum Wage and Unemployment}
\\author{ECON 4999B Course Material}
\\date{\\today}

\\begin{document}

\\maketitle

\\section{Data}

We use a balanced panel of 26 regions (A--Z) and 27 years (2000--2026), for
$26 \\times 27 = 702$ observations. The outcome variable is
\\texttt{unemployment\\_rate} (percent) and the regressor of interest is
\\texttt{min\\_wage} (dollars).

\\section{Specifications}

We estimate
\\begin{equation}
    \\text{unemployment\\_rate}_{it} = \\alpha + \\beta\\,\\text{min\\_wage}_{it}
    + \\mu_i + \\lambda_t + \\varepsilon_{it},
\\end{equation}
where $\\mu_i$ denotes region fixed effects and $\\lambda_t$ denotes year fixed
effects. The four columns differ by which fixed effects are included:
\\begin{enumerate}
    \\item[(1)] pooled OLS (no fixed effects);
    \\item[(2)] year fixed effects only;
    \\item[(3)] region fixed effects only;
    \\item[(4)] region and year fixed effects (two-way FE).
\\end{enumerate}
All models use heteroskedasticity-robust standard errors (HC1).

\\section{Results}

\\begin{table}[htbp]
\\centering
\\caption{Effect of minimum wage on unemployment under alternative specifications}
\\label{tab:fe_results}
\\begin{tabular}{lcccc}
\\toprule
 & (1) & (2) & (3) & (4) \\\\
\\midrule
\\multicolumn{5}{l}{\\textit{Dependent variable: unemployment rate (\\%)}} \\\\
\\addlinespace
Min wage & COEF0 & COEF1 & COEF2 & COEF3 \\\\
 & SE0 & SE1 & SE2 & SE3 \\\\
\\addlinespace
Fixed effects & None & Year & Region & Region, Year \\\\
Observations & \\multicolumn{4}{c}{NOBS} \\\\
$R^2$ & __R2__ \\\\
Adjusted $R^2$ & __ADJ_R2__ \\\\
\\bottomrule
\\end{tabular}

\\medskip
\\footnotesize
\\textit{Notes:} HC1 robust standard errors in parentheses.
$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\\end{table}

\\section{Discussion}

Without fixed effects, the estimated minimum-wage coefficient reflects both
within-region variation and persistent cross-region differences. Regions with
historically higher minimum wages also differ in other ways (industry mix,
demographics, labor-market institutions), so the pooled estimate can be
misleading.

Adding region fixed effects (column 3) uses only within-region changes in the
minimum wage, removing time-invariant regional confounders. Adding year fixed
effects (column 2) removes common macro shocks shared by all regions in a given
year. Two-way fixed effects (column 4) combine both: identification comes from
region-specific deviations from national trends.

In this synthetic dataset, the pooled estimate is negative because high-wage
regions have lower baseline unemployment for reasons unrelated to minimum-wage
changes. Once region fixed effects are included, the coefficient turns positive,
consistent with the within-region relationship built into the data-generating
process.

\\end{document}
"""

    tex = (
        tex.replace("COEF0", coefs[0])
        .replace("COEF1", coefs[1])
        .replace("COEF2", coefs[2])
        .replace("COEF3", coefs[3])
        .replace("SE0", ses[0])
        .replace("SE1", ses[1])
        .replace("SE2", ses[2])
        .replace("SE3", ses[3])
        .replace("NOBS", str(nobs))
        .replace("__ADJ_R2__", adj_r2)
        .replace("__R2__", r2)
    )
    TEX_PATH.write_text(tex)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    results = [(label, fit_model(formula, data)) for label, formula in SPECS]

    print("Minimum wage coefficient by specification")
    print("-" * 60)
    for label, res in results:
        print(
            f"{label:32s}  beta={res.params['min_wage']:7.3f}  "
            f"se={res.bse['min_wage']:0.3f}  "
            f"p={res.pvalues['min_wage']:0.4f}  "
            f"R2={res.rsquared:0.3f}"
        )

    write_tex(results)
    print(f"\nWrote {TEX_PATH}")


if __name__ == "__main__":
    main()

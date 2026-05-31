# Sector Health and Earnings Quality Across the 2014–2018 Market Cycle

## Which sectors have the most companies represented, and how does that distribution change year to year?

![company counts by sector and year](report-data/which-sectors-have-the-most-companies-represented,-and-how-does-that-distribution-change-year-to-year/plot-sector-distribution-by-year-stacked.png)

![trend of top 6 sectors over years](report-data/which-sectors-have-the-most-companies-represented,-and-how-does-that-distribution-change-year-to-year/plot-top-6-sector-trends.png)

Summary of which sectors have the most companies and how that changed year-to-year

1) Top sectors overall
- Financial Services is the largest sector by far across the five years (total unique tickers = 4,720).
- Healthcare is 2nd (3,305) and Technology is 3rd (3,126). The next largest are Industrials (2,768) and Consumer Cyclical (2,471).

2) Top sectors each year
- For every year 2014–2018 the top three sectors by number of unique companies are the same: 1) Financial Services, 2) Healthcare, 3) Technology.
- Yearly top-3 counts (2014 → 2018):
  - 2014: Financial Services 660, Healthcare 582, Technology 576
  - 2015: Financial Services 769, Healthcare 628, Technology 606
  - 2016: Financial Services 1220, Healthcare 686, Technology 643
  - 2017: Financial Services 1247, Healthcare 718, Technology 665
  - 2018: Financial Services 824, Healthcare 691, Technology 636

3) Distribution and notable changes year-to-year (percent shares of companies by year)
- Financial Services share: 17.33% (2014) → 18.67% (2015) → 25.43% (2016) → 25.14% (2017) → 18.76% (2018).
  - Interpretation: a pronounced spike in 2016–2017 (both count and percent share) followed by a drop in 2018 (counts drop from 1,247 in 2017 to 824 in 2018).
- Healthcare share: roughly stable, ~15% each year (15.28 → 15.24 → 14.30 → 14.48 → 15.73).
- Technology share: small dip mid-period then partial recovery (15.13 → 14.71 → 13.40 → 13.41 → 14.48).
- Industrials and Consumer Cyclical are steady middle-weight sectors (Industrials ~11.8–13.2%; Consumer Cyclical ~10.5–12.0%).
- Smaller sectors (Basic Materials, Real Estate, Energy, Consumer Defensive, Utilities, Communication Services) each contribute between ~1.9% and ~6.4% in any given year and show only modest year-to-year change.

4) Bottom-line takeaway
- Financial Services dominates the sample overall and drives the largest year-to-year swings (notably the 2016–2017 spike and subsequent fall). Healthcare and Technology are consistently the next-largest sectors and remain fairly stable in share. Most other sectors show only incremental year-to-year variation.

## Which sector saw the largest improvement in median EPS over the 5-year window?

Basic Materials and Utilities showed the largest improvement in median EPS over the 2014→2018 window. Each sector's median EPS rose by 0.25 (Basic Materials: 0.40 → 0.65; Utilities: 1.68 → 1.93).

## For the top-performing EPS sector, which individual companies drove that improvement — and were any consistent outliers?

Short answer
- The single largest numeric contributor to Utilities' EPS improvement is OPTT (Total_EPS_Change = +168.76), but it is a recovery from an extreme negative start (EPS_2014 = -182.0) and is a volatility outlier (eps_std_zscore ≈ 8.76). Treat OPTT as a noisy, recovery-driven contributor rather than a steady growth driver.
- Other top-ranked contributors by Total_EPS_Change (from the computed top-20) include EDN (+24.19) and FCEL (+15.24). Both also began with materially negative EPS in 2014 (EDN = -17.4, FCEL = -24.24), so their large gains are largely recoveries.
- The more credible, sustained contributors are companies that (a) appear in the top-20 by Total_EPS_Change and (b) have positive EPS across the full 2014–2018 span. Notable examples from the top-20 are NEE (Total_EPS_Change = +8.36; EPS positive every year) and PAM (+7.94; EPS positive every year). These firms represent steadier, multi-year EPS improvement in the sector.

Quantitative context from the run
- Utilities tickers identified (sector_mode == 'Utilities'): 110.
- Tickers with both 2014 and 2018 EPS (included in the ranking): 93; top-20 were returned.
- Top-1 by raw change: OPTT (EPS_2014 = -182.00 → EPS_2018 = -13.24; change = +168.76), but flagged as large_negative_start and volatile.
- Of the top-20 list produced, 6 tickers show EPS_2014 <= -1.0 (flag_large_negative_start True): OPTT, EDN, FCEL, EBR, SKYS, AT — i.e., many large numeric changes are recoveries.
- Volatile outliers (eps_std z-score > 2.0) found: CEQP (z ≈ 2.63), OPTT (z ≈ 8.76), VST (z ≈ 3.14). These are high-dispersion EPS histories and should be downweighted when attributing steady sector improvement.
- Consistent-positive tickers (EPS > 0 in all five years) — a strict list of steady performers — includes many Utilities names; examples that also show meaningful Total_EPS_Change and appear in the top-20: NEE, PAM. The full consistent-positive list was produced programmatically (includes AEE, AEP, NEE, PAM, SR, CORR, OGS, D, UGI, etc.).

Interpretation / recommendations
- Don't treat the largest Total_EPS_Change values as proof of broad, sustainable sector recovery without checking starting points and volatility. OPTT and several other large changers are recoveries from negative EPS and are volatile; they inflate headline sector improvement but are not evidence of steady, across-the-board gains.
- For identifying steady drivers of Utilities' EPS strength, focus on tickers that (1) are consistent-positive across 2014–2018 and (2) show positive Total_EPS_Change without a large negative 2014. NEE and PAM are good examples from the computed top-20: they provide clear, sustained EPS improvement.
- Volatile tickers (CEQP, OPTT, VST) should be inspected case-by-case (e.g., one-time charge reversals, accounting changes, asset sales) before treating them as sector drivers.

## How did the median debt-to-equity ratio evolve across all sectors from 2014 to 2018?

![median debt-to-equity (2014-2018)](report-data/how-did-the-median-debt-to-equity-ratio-evolve-across-all-sectors-from-2014-to-2018/plot-median-debt-to-equity-2014-2018.png)

The median Debt-to-Equity ratio rose steadily across all sectors from 2014 to 2018. Year-by-year medians were:
- 2014: 0.3893
- 2015: 0.4268
- 2016: 0.4352
- 2017: 0.4499
- 2018: 0.4650

Overall, the median increased from 0.3893 in 2014 to 0.4650 in 2018, a rise of 0.0757 (≈19.45% increase). Note: some years (2016–2017) had higher proportions of missing Debt-to-Equity values (about 15–17%), which could affect the medians; no outlier trimming was applied.

## Based on EPS growth and leverage trends, which sector appears most financially resilient entering a hypothetical 2019 downturn?

Conclusion: Financial Services is the most financially resilient sector entering the hypothetical 2019 downturn, based on the composite that combines EPS growth and leverage/coverage trends. Financial Services ranks first with a Resilience_Score of 0.5789.

Why Financial Services placed first (key numbers from the analysis):
- Composite Resilience_Score: 0.5789 (highest among sectors).
- Averaged metrics (2014–2018): Avg_EPS = 0.777; Avg_Debt/Equity = 0.4717; Avg_Debt/Assets = 0.158; Avg_NetDebt/EBITDA = 1.2964; Avg_InterestCoverage = 18.8933.
- Component z-scores used in the composite (cross-sectional z on averaged metrics, with leverage z-scores inverted so higher = better): z_EPS = 0.8644, z_DE (inverted) = 0.1092, z_DA (inverted) = 1.5816, z_NDE (inverted) = 0.0361, z_IC_used (log-transformed z of interest coverage) = 0.3035. The mean of these five components produced the final score 0.5789.

Notes on methodology and important caveats that affect interpretation:
- Composite components: Avg_EPS, Avg_Debt/Equity, Avg_Debt/Assets, Avg_NetDebt/EBITDA, Avg_InterestCoverage. Each component was standardized (z-score, ddof=1) across sectors; leverage measures (DE, DA, NDE) were inverted so higher composite values reflect better resilience. All components were equally weighted in the final score.
- Interest Coverage was log-transformed before standardization for the composite. This decision was automatic because the interest-coverage distribution had extreme values (min <= 0 producing an infinite max/min ratio), so a shift+log was used to reduce distortion from large positive/negative outliers.
- The ranking produced (top to bottom) was: Financial Services; Consumer Defensive; Communication Services; Technology; Basic Materials; Real Estate; Healthcare; Energy; Consumer Cyclical; Utilities; Industrials.
- Caveats: several sectors show extreme or negative average Interest Coverage (e.g., Healthcare avg IC = -300.6), so the log+shift transform materially affects the IC component and therefore the composite. The composite assumes equal weights and simple means; different weighting schemes (e.g., putting more emphasis on leverage or EPS) or alternative outlier treatments (winsorization, median-based scaling) could change the ordering.

Bottom line: Under the specified equal-weight z-score composite (with log-transformed interest coverage due to extreme values), Financial Services appears most resilient entering a 2019 downturn, driven by above-average EPS growth and favorable leverage/coverage z-scores.

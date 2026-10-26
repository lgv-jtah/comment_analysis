# Vote-ratio equivalence test: report

Run: 20261026T094931373229Z

H0 (non-equivalence) vs H1 (equivalence): the comment favor share and the vote pro share are declared equivalent iff the confidence interval for their difference lies strictly inside (-10, +10) percentage points. The test is a clustered-bootstrap TOST; the 90% interval drives the primary verdict and the 95% interval is reported alongside.

## Interpretation (read first)

> Equivalence does not prove that stance detection is unnecessary; non-equivalence does not prove that it is necessary. This test answers only whether the overall favor share of comments matches the overall share of thumbs up among votes.

> The comment labels are frozen gold data, while the votes are a live, monotonically growing quantity read at the snapshot time below. The two sides are therefore measured at different times; treat the comparison as between a frozen comment-label set and a later vote snapshot.

Vote snapshot taken at: 20261026T094846675419Z.

## Counts

| Quantity | Value |
| --- | --- |
| Comments | 246 |
| Pairs | 707 |
| Contributions (fixed set) | 115 |
| Contributions with votes | 107 |
| Contributions with zero votes | 8 |
| Contributions missing votes | 0 |
| Total thumbs up (pro) | 780 |
| Total thumbs down (contra) | 97 |
| Mixed comments | 11 |
| Neutral comments | 74 |
| Generic-approval pairs (strict) | 67 |
| Generic-approval pairs (broad) | 67 |
| Generic-approval comments (strict) | 30 |
| Generic-approval comments (broad) | 30 |

## Primary result (comment unit, mixed excluded, labels as-is)

- Favor share: 80.7% (in_favor 130, against 31, neutral 74, mixed 11)
- Neutral share: 30.1%
- Vote pro share: 88.9% (pro 780, contra 97)
- Difference (favor - vote): -8.2 pp
- 90% (primary): NOT equivalent (CI [-14.6, -1.1] pp)
- 95%: NOT equivalent (CI [-15.8, 0.5] pp)

Bootstrap: 10000 resamples, seed 20261026, 0 dropped for a zero denominator.

## Sensitivity analyses

### (a) Pair level

- Favor share: 80.4% (in_favor 246, against 60, neutral 401, mixed 0)
- Neutral share: 56.7%
- Vote pro share: 88.9% (pro 780, contra 97)
- Difference (favor - vote): -8.5 pp
- 90% (primary): NOT equivalent (CI [-15.3, -0.9] pp)
- 95%: NOT equivalent (CI [-16.7, 0.5] pp)

### (b) Generic approval -> neutral

- Favor share: 79.9% (in_favor 123, against 31, neutral 81, mixed 11)
- Neutral share: 32.9%
- Vote pro share: 88.9% (pro 780, contra 97)
- Difference (favor - vote): -9.1 pp
- 90% (primary): NOT equivalent (CI [-15.4, -1.9] pp)
- 95%: NOT equivalent (CI [-16.7, -0.5] pp)

### (b) Generic approval -> in_favor

- Favor share: 82.6% (in_favor 147, against 31, neutral 57, mixed 11)
- Neutral share: 23.2%
- Vote pro share: 88.9% (pro 780, contra 97)
- Difference (favor - vote): -6.4 pp
- 90% (primary): NOT equivalent (CI [-12.4, 0.6] pp)
- 95%: NOT equivalent (CI [-13.6, 2.0] pp)

### (c) Per-contribution unweighted mean

- Favor share: 80.0% (in_favor 130, against 31, neutral 74, mixed 11)
- Neutral share: 30.1%
- Vote pro share: 89.1% (pro 780, contra 97)
- Difference (favor - vote): -9.0 pp
- 90% (primary): NOT equivalent (CI [-15.7, -2.6] pp)
- 95%: NOT equivalent (CI [-17.1, -1.5] pp)

### (d) Mixed comments counted as a half

- Favor share: 78.8% (in_favor 130, against 31, neutral 74, mixed 11)
- Neutral share: 30.1%
- Vote pro share: 88.9% (pro 780, contra 97)
- Difference (favor - vote): -10.2 pp
- 90% (primary): NOT equivalent (CI [-16.5, -2.9] pp)
- 95%: NOT equivalent (CI [-17.7, -1.6] pp)


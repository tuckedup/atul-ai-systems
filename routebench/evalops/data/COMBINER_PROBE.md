# Combiner probe: dev-internal grouped cross-validation

**Not a result.** The acceptance protocol fits on TRAIN and measures once on TEST. The judgment cache holds dev judgments only -- the train split was never judged -- so this probe fits and selects inside dev by grouped k-fold CV. Its numbers are optimistic relative to a held-out measurement and must not be used to freeze a bundle or to claim a kappa.

- Components: `v0-generic-4o-mini`, `v1-rubric-4o-mini`, `v2-rubric-noref-4o-mini`, `v3-rubric-noboundary-4o-mini`, `v4-rubric-4.1-mini`, `v5-rubric-4.1`, `v6-rubric-fewshot-4.1-mini`, `v7-rubric-fewshot-4.1`
- Provider calls: 0 (reads the judgment cache only)
- Joint coverage: 229/371 dev cases (61.7%) have a usable judgment from every component
- Missing judgments by component: `{'v1-rubric-4o-mini': 11, 'v2-rubric-noref-4o-mini': 7, 'v3-rubric-noboundary-4o-mini': 14, 'v4-rubric-4.1-mini': 2, 'v5-rubric-4.1': 51, 'v6-rubric-fewshot-4.1-mini': 6, 'v7-rubric-fewshot-4.1': 82}`
- Evidence features: 0/2968 loaded judgments carry evidence_location. The evidence-validity feature family is therefore all-zero here: this probe does NOT test whether those signals help. Re-judging with the current code is required to test it.

## all_admissible_labels

n=229 over 160 independent groups, human prevalence 0.555, 5-fold grouped CV. Task mix: `{'code': 53, 'reason': 9, 'sql': 64, 'summarize': 87, 'tool_use': 16}`.

| contender | kind | out-of-fold kappa | agreement | confusion |
|---|---|---|---|---|
| `v0-generic-4o-mini` | single | 0.1070 | 0.555 | tp=71 fp=46 fn=56 tn=56 |
| `v1-rubric-4o-mini` | single | 0.3738 | 0.699 | tp=106 fp=48 fn=21 tn=54 |
| `v2-rubric-noref-4o-mini` | single | 0.2771 | 0.659 | tp=112 fp=63 fn=15 tn=39 |
| `v3-rubric-noboundary-4o-mini` | single | 0.2956 | 0.664 | tp=106 fp=56 fn=21 tn=46 |
| `v4-rubric-4.1-mini` | single | 0.4689 | 0.747 | tp=116 fp=47 fn=11 tn=55 |
| `v5-rubric-4.1` | single | 0.4999 | 0.760 | tp=114 fp=42 fn=13 tn=60 |
| `v6-rubric-fewshot-4.1-mini` | single | 0.4570 | 0.742 | tp=118 fp=50 fn=9 tn=52 |
| `v7-rubric-fewshot-4.1` | single | 0.4442 | 0.734 | tp=112 fp=46 fn=15 tn=56 |
| `mean-score` | unweighted | 0.4752 | 0.747 | tp=110 fp=41 fn=17 tn=61 |
| `majority-vote` | unweighted | 0.4646 | 0.742 | tp=111 fp=43 fn=16 tn=59 |
| `logistic[score]C=0.1` | learned | 0.4678 | 0.747 | tp=117 fp=48 fn=10 tn=54 |
| `logistic[score]C=1.0` | learned | 0.4796 | 0.751 | tp=115 fp=45 fn=12 tn=57 |
| `logistic[criteria]C=0.1` | learned | -0.1259 | 0.476 | tp=97 fp=90 fn=30 tn=12 |
| `logistic[criteria]C=1.0` | learned | -0.1259 | 0.476 | tp=97 fp=90 fn=30 tn=12 |
| `logistic[score+criteria]C=0.1` | learned | 0.4862 | 0.755 | tp=118 fp=47 fn=9 tn=55 |
| `logistic[score+criteria]C=1.0` | learned | 0.4796 | 0.751 | tp=115 fp=45 fn=12 tn=57 |
| `logistic[score+criteria+evidence]C=0.1` | learned | 0.4862 | 0.755 | tp=118 fp=47 fn=9 tn=55 |
| `logistic[score+criteria+evidence]C=1.0` | learned | 0.4796 | 0.751 | tp=115 fp=45 fn=12 tn=57 |

Strongest single component: **`v5-rubric-4.1`** (kappa 0.4999). Best overall: **`v5-rubric-4.1`** (kappa 0.4999). Margin: **+0.0000**. Combiner helps: **False**.

## human_judgment_of_response

n=87 over 41 independent groups, human prevalence 0.655, 5-fold grouped CV. Task mix: `{'summarize': 87}`.

| contender | kind | out-of-fold kappa | agreement | confusion |
|---|---|---|---|---|
| `v0-generic-4o-mini` | single | 0.1212 | 0.471 | tp=13 fp=2 fn=44 tn=28 |
| `v1-rubric-4o-mini` | single | 0.4707 | 0.782 | tp=53 fp=15 fn=4 tn=15 |
| `v2-rubric-noref-4o-mini` | single | 0.6027 | 0.828 | tp=52 fp=10 fn=5 tn=20 |
| `v3-rubric-noboundary-4o-mini` | single | 0.4707 | 0.782 | tp=53 fp=15 fn=4 tn=15 |
| `v4-rubric-4.1-mini` | single | 0.5581 | 0.816 | tp=54 fp=13 fn=3 tn=17 |
| `v5-rubric-4.1` | single | 0.5892 | 0.828 | tp=54 fp=12 fn=3 tn=18 |
| `v6-rubric-fewshot-4.1-mini` | single | 0.5264 | 0.805 | tp=54 fp=14 fn=3 tn=16 |
| `v7-rubric-fewshot-4.1` | single | 0.6440 | 0.851 | tp=55 fp=11 fn=2 tn=19 |
| `mean-score` | unweighted | 0.6092 | 0.828 | tp=51 fp=9 fn=6 tn=21 |
| `majority-vote` | unweighted | 0.5727 | 0.816 | tp=52 fp=11 fn=5 tn=19 |
| `logistic[score]C=0.1` | learned | 0.4177 | 0.770 | tp=55 fp=18 fn=2 tn=12 |
| `logistic[score]C=1.0` | learned | 0.6199 | 0.839 | tp=54 fp=11 fn=3 tn=19 |
| `logistic[criteria]C=0.1` | learned | 0.6440 | 0.851 | tp=55 fp=11 fn=2 tn=19 |
| `logistic[criteria]C=1.0` | learned | 0.6199 | 0.839 | tp=54 fp=11 fn=3 tn=19 |
| `logistic[score+criteria]C=0.1` | learned | 0.6440 | 0.851 | tp=55 fp=11 fn=2 tn=19 |
| `logistic[score+criteria]C=1.0` | learned | 0.6199 | 0.839 | tp=54 fp=11 fn=3 tn=19 |
| `logistic[score+criteria+evidence]C=0.1` | learned | 0.6440 | 0.851 | tp=55 fp=11 fn=2 tn=19 |
| `logistic[score+criteria+evidence]C=1.0` | learned | 0.6199 | 0.839 | tp=54 fp=11 fn=3 tn=19 |

Strongest single component: **`v7-rubric-fewshot-4.1`** (kappa 0.6440). Best overall: **`v7-rubric-fewshot-4.1`** (kappa 0.6440). Margin: **+0.0000**. Combiner helps: **False**.

## gold_oracle

n=142 over 119 independent groups, human prevalence 0.493, 5-fold grouped CV. Task mix: `{'code': 53, 'reason': 9, 'sql': 64, 'tool_use': 16}`.

| contender | kind | out-of-fold kappa | agreement | confusion |
|---|---|---|---|---|
| `v0-generic-4o-mini` | single | 0.2161 | 0.606 | tp=58 fp=44 fn=12 tn=28 |
| `v1-rubric-4o-mini` | single | 0.2565 | 0.627 | tp=54 fp=37 fn=16 tn=35 |
| `v2-rubric-noref-4o-mini` | single | 0.1200 | 0.556 | tp=60 fp=53 fn=10 tn=19 |
| `v3-rubric-noboundary-4o-mini` | single | 0.1868 | 0.592 | tp=53 fp=41 fn=17 tn=31 |
| `v4-rubric-4.1-mini` | single | 0.4114 | 0.704 | tp=62 fp=34 fn=8 tn=38 |
| `v5-rubric-4.1` | single | 0.4390 | 0.718 | tp=61 fp=31 fn=9 tn=41 |
| `v6-rubric-fewshot-4.1-mini` | single | 0.3979 | 0.697 | tp=64 fp=37 fn=6 tn=35 |
| `v7-rubric-fewshot-4.1` | single | 0.3409 | 0.669 | tp=58 fp=35 fn=12 tn=37 |
| `mean-score` | unweighted | 0.3104 | 0.655 | tp=48 fp=27 fn=22 tn=45 |
| `majority-vote` | unweighted | 0.3543 | 0.676 | tp=56 fp=32 fn=14 tn=40 |
| `logistic[score]C=0.1` | learned | 0.4250 | 0.711 | tp=61 fp=32 fn=9 tn=40 |
| `logistic[score]C=1.0` | learned | 0.4529 | 0.725 | tp=61 fp=30 fn=9 tn=42 |
| `logistic[criteria]C=0.1` | learned | 0.1006 | 0.549 | tp=44 fp=38 fn=26 tn=34 |
| `logistic[criteria]C=1.0` | learned | 0.2161 | 0.606 | tp=58 fp=44 fn=12 tn=28 |
| `logistic[score+criteria]C=0.1` | learned | 0.3829 | 0.690 | tp=59 fp=33 fn=11 tn=39 |
| `logistic[score+criteria]C=1.0` | learned | 0.4668 | 0.732 | tp=61 fp=29 fn=9 tn=43 |
| `logistic[score+criteria+evidence]C=0.1` | learned | 0.3829 | 0.690 | tp=59 fp=33 fn=11 tn=39 |
| `logistic[score+criteria+evidence]C=1.0` | learned | 0.4668 | 0.732 | tp=61 fp=29 fn=9 tn=43 |

Strongest single component: **`v5-rubric-4.1`** (kappa 0.4390). Best overall: **`logistic[score+criteria+evidence]C=1.0`** (kappa 0.4668). Margin: **+0.0278**. Combiner helps: **True**.


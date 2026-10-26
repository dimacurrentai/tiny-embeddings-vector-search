# CLINC150 subset evaluation

200 English snippets: 20 intents, 10 snippets per intent. This is a derived
intent-retrieval evaluation, not the full CLINC150 benchmark and not a set of
human-rated pairwise similarity scores. Same-intent snippets are treated as
relevant; different-intent snippets are treated as irrelevant. This is a proxy:
two texts with different intents can still be semantically related.

Source: Stefan Larson et al., **An Evaluation Dataset for Intent Classification
and Out-of-Scope Prediction**, EMNLP-IJCNLP 2019.
Paper: https://aclanthology.org/D19-1131/
Repository: https://github.com/clinc/oos-eval
Full author attribution and citation: [UPSTREAM_README.md](UPSTREAM_README.md).

The data is distributed under **Creative Commons Attribution 3.0 Unported**.
The original [LICENSE](LICENSE) applies to this fixture, independently of the
service code. Retain its license and attribution when redistributing.

`provenance.json` records the immutable source revision, source SHA-256, and
changes. Selection was fixed before measuring model performance: the first 20
alphabetically sorted intents and first 10 test rows per intent in source order.
No rows were selected or removed based on model results. This selection is small
and biased toward those 20 intents; it does not represent the whole dataset.

Reproduce with `uv run python scripts/vendor_clinc150.py`. This command downloads
the original data, but normal evaluation reads only the committed JSON.

For each of the 200 queries, search the other 199 snippets (never itself).
Report top-1 accuracy, hit rate at 5, recall at 5, and mean reciprocal rank at 5.
There are exactly 9 relevant candidates per query, so recall at 5 cannot exceed
5/9. The aggregate regression floors were chosen before the first evaluation:
top-1 accuracy >= 0.70 and hit rate at 5 >= 0.85. These are regression guards,
not claims about performance on unseen data. Do not tune them against this fixture.

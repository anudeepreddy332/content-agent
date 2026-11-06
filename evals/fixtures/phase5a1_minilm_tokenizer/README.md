# Frozen offline MiniLM tokenizer fixture

These are the exact three tokenizer files from the pre-existing local snapshot
of `sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, copied without network access.
Their SHA-256 values are frozen in
`reports/phase5/phase5a0/baseline_a_manifest.json` under
`embedding.tokenizer_file_sha256` and are checked before loading.

Only tokenization is performed. No embeddings or model weights are included.
The fixture makes offline qualification reproducible without a populated
Hugging Face cache. Do not regenerate, normalize, or replace these files as part
of Candidate B/C work.

# Frozen BGE reranker tokenizer fixture

These files were copied from the existing local snapshot of
`BAAI/bge-reranker-base` revision
`2cfc18c9415c912f9d8155881c133215df768a70`, without a download.
The adjacent Phase 5A2J contract pins their exact SHA-256 identities.

The fixture supports offline pair-length and tokenizer tests without model
weights or a populated Hugging Face cache. Full BGE inference still requires
the complete pinned local model snapshot and refuses to download or substitute
it. Default CI uses this fixture only; full-model BGE evaluation remains a
manual experiment gate.

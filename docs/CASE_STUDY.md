# content-agent: an AI writing system with a human in control

> **Historical portfolio narrative.** This document describes the June 2026 supervised demo and is
> not the current engineering-state or enterprise-release authority. Some durability, attribution,
> security, and production-readiness statements are superseded. Read `PROJECT_STATUS.md`,
> `architecture.md`, and `docs/EXPERIMENT_LEDGER.md` for current accepted state and evidence.

## The one-line version

Type a topic, press Enter, and watch an article get researched, drafted, fact-checked, and published to a live website within minutes, with a person approving the work at two checkpoints before anything goes public.

## What the system does

content-agent is an automated writing pipeline for themachinist.org, a technical publication for machine learning and software engineers. You give it a subject, for example "Why batch normalization speeds up training." It searches the web and a local library of documents for sources, writes a structured article grounded in what it found, checks its own claims against those sources, reflects on the result, and pauses for a human to review. Once approved, it renders the final web page, pauses again for a layout review, and publishes to a live site after one more deliberate human action.

One principle runs through the whole design. The model does the heavy lifting, but a person decides what gets published. Nothing reaches the public site without explicit human approval.

## How it is built

### The pipeline

The system is a single agent built on LangGraph, a framework that describes an AI workflow as a graph of steps (called nodes) with rules for moving between them. "Single agent" means one coordinated process rather than several AI agents handing work to each other.

The flow looks like this:

```
retrieve → draft → verify → reflect → (revise or continue)
   → HUMAN GATE 1 (content) → render HTML
   → HUMAN GATE 2 (layout) → publish
```

Here is each stage in plain terms:

- **Retrieve.** Gather sources from two places at once: a live web search through Tavily (a search API built for AI use) and a local knowledge base of curated documents. The knowledge base uses hybrid retrieval, which combines two ways of finding text. One is dense vector search, which matches by meaning using a small model called all-MiniLM-L6-v2 that turns text into numbers representing its meaning. The other is BM25, which matches by keywords. The two ranked lists are blended with a method called Reciprocal Rank Fusion.
- **Draft.** Write the article using only the retrieved sources, under a rule to either cite a source for a specific claim or keep the claim general when no source supports it.
- **Verify.** Re-read the draft and check each factual claim against the sources, labeling it verified, weak, or unverified.
- **Reflect.** Score the draft and decide whether it needs another pass.
- **Revise or continue.** If the article is weakly grounded, send it back to drafting with a report of which claims failed, so the next pass can fix them. Otherwise move forward.
- **Human gate 1 (content).** A person reads the draft, the grounding report, and the reflection, then approves, rejects, or sends feedback.
- **Render HTML.** Turn the approved content into a styled web page.
- **Human gate 2 (layout).** The person reviews how the page looks. The text is frozen here. Feedback at this gate changes only design and layout, never content.
- **Publish.** Merge the article into the website's repository, and after one final human action, push it live.

The two-gate structure carries real weight. The first gate is about what the article says. The second is about how it looks. Keeping them separate means a layout tweak can never quietly alter a fact that was already approved.

The model doing the writing and checking is DeepSeek, reached over an API. Every model call has a timeout and automatic retries, so a slow or failed request does not crash a run.

### Keeping a human in the loop, safely

The review steps are durable. A FastAPI web server runs the pipeline, and run state is saved to disk through a checkpointing layer (SqliteSaver). When the pipeline reaches a human gate, it pauses and stores everything it needs to resume later. A reviewer can come back after a break, or even after the server restarts, and pick up where the run left off.

### The stack, and why each piece is there

- **LangGraph** to express the pipeline as a clear, inspectable graph.
- **DeepSeek** as the language model behind drafting, verifying, and reflecting.
- **Tavily** for live web search.
- **Qdrant with BM25 and Reciprocal Rank Fusion** for the local knowledge base, so retrieval catches both meaning and exact terms.
- **FastAPI with SqliteSaver** for the API and durable human review.
- **GitPython and Netlify** for publishing: the system commits to a website repository, and Netlify serves the result.
- **Docker** so the whole thing runs the same way on any machine.
- **uv** as the Python package manager, for fast and reproducible installs.

### The cloud deployment

The image is built once on a developer machine and pulled onto an AWS EC2 server. This is the standard build-once, ship-the-artifact pattern, and it keeps the small server from doing slow, fragile builds. A reverse proxy called Caddy sits in front of the application and provides automatic HTTPS, so the live demo has a real secure web address. The published articles are hosted on Netlify. The vector database runs in its own container with no public port, reachable only by the application.

## The road to here

The project grew in phases, each one building skill for the next.

- **Phases 1 to 3.** A command-line agent first, then a retrieval-augmented agent that could answer from a document library, then an agent that could read and fix code. These established the core patterns: tool use, retrieval, and structured reasoning.
- **Phase 4A, the grounding arc (M1 to M5).** This is where the writing became trustworthy. The work focused on making every claim traceable to a source, measuring grounding honestly, and tracking each claim back to the specific chunk of text that supports it.
- **Phase 4B, the freeze sprint (B1 to B8).** A compressed run of production hardening under a real deadline: input sanitization, a suite of failure-injection tests, a two-tier continuous integration setup with enforced gates, durable human review, a non-root container with the vector store isolated, validated publishing, and a tested rollback path.
- **Post-freeze (P2.1 to P2.3).** A content-frozen layout gate, automatic updating of the site's article index when a new piece publishes, and a careful experiment on revision depth.

## The hard problems, and what they taught me

The interesting engineering was rarely the model. It was in the problems that did not announce themselves.

**The grounding investigation.** Early on, math-heavy topics produced weakly grounded drafts, and the natural suspicion was that the agent's search queries were too vague. The plan was to rewrite queries to be more specific. Testing rejected that idea. The real driver was retrieval freshness. When the search cache held stale or empty results, grounding collapsed. Once retrieval was fresh, even generic queries recovered about ninety percent of the formula claims. The lesson: test the assumed cause before building the fix, because the obvious hypothesis was wrong.

**The verifier exoneration.** On some healthy topics, the count of solid verified claims stayed flat, which looked like the claim-checker was being too harsh. A careful adjudication against an independent judgment showed all twenty-seven substantive claims were verified correctly. The checker was right. The flat number was a ceiling set by how much source material existed, not a defect. The lesson: prove a component is actually wrong before fixing it, because suspicion is not evidence.

**The telemetry bug that hid in plain sight.** A chained assignment in the code had been writing the web-search timing into the knowledge-base timing field since the project began. The numbers looked plausible, so it went unnoticed for a long time, and only surfaced when the two timings were split apart for analysis. The lesson: plausible-looking metrics can be quietly wrong, so cross-check them.

**The rollback that could destroy what it was meant to save.** The first version of the rollback script used a force-delete that could erase article content instead of cleanly reverting it. It was caught before it ever touched real content and replaced with a history-preserving revert. The lesson: a recovery tool that can lose data is worse than none, so treat rollback code with the same care as the thing it protects.

**The filename mismatch.** The HTML generator was stamping run identifiers into filenames, so the published file name did not match what the index page linked to, which broke the logic that tells a brand-new article apart from an update to an existing one. The lesson: small naming inconsistencies break downstream logic in ways that are hard to trace back.

**Code that was not the code I was reading.** After migrating the vector database from one tool to another, an old package was still installed and a stale source file lived inside the virtual environment, so the program was running something other than what the repository contained. The fix came from asking Python directly where a function was being loaded from. The lesson: when behavior contradicts the source in front of you, check what is actually installed and running.

**A limitation kept in the open.** The system saves full run state to disk, so a restart does not lose work. But the web layer keeps a separate in-memory map of active runs that is not rebuilt on restart, so a run paused for review can become unreachable through the API after a restart even though its state is safe on disk. This was written down as a known limitation, with a clear path to close it, rather than papered over. The lesson: documented limitations are honest engineering.

## The decisions that shaped the system

**One agent, not many.** A multi-agent design was considered and set aside. A single, well-structured agent did the job, and extra agents would have added coordination and new failure modes without a matching benefit. Complexity had to earn its place, and here it did not.

**Measuring quality the right way.** The first quality metric was the fraction of claims that verified. That metric punishes a careful draft that makes few, solid claims. The primary metric became SV, a count of substantive claims that checked out, with the unverified rate capped at fifteen percent as a grounding gate. The team also voided "lower the unverified rate" as a goal on its own, because you can lower it simply by making claims vaguer, which is the opposite of quality. So substance must not drop when the unverified rate improves.

**Killing the blind re-roll.** An early revision strategy re-drafted the whole article from scratch when grounding was weak, with no information about what had gone wrong. It made things worse about as often as better. It was replaced with feedback-driven revision, where the next draft receives a precise report of which claims failed and how to address them.

**Holding the revision limit at two.** A pre-registered experiment tested whether allowing a third revision pass improved grounding. It did not. On healthy drafts the third pass never even triggered, and in the one case where it did, it lowered the unverified rate only by making claims vaguer, with no gain in substantive verified claims. The experiment was rejected and the limit stayed at two.

**Cutting tracing from the freeze.** LangSmith, a tracing tool for language-model applications, was deliberately left out of the production freeze to keep the scope tight and the deadline honest. It was deferred, not abandoned, and is being added afterward as an opt-in feature.

## How I know it works

**Pre-registered experiments.** Before running a test, the variants, topics, repetitions, and exact pass-or-fail criteria were written down. That removes the temptation to run a test and decide afterward what counts as success.

**A calibrated verifier.** The claim-checker was measured against an independent judgment to confirm it was neither too strict nor too lenient.

**A golden fixture.** A fixed input with a known-correct verifier output runs inside the test suite, so a change that quietly breaks verification fails the build.

**Enforced CI gates.** The automated checks that run on every change enforce real pass-or-fail results, after an earlier version was found to report success no matter what happened.

**Fork validation.** Every change to the publishing logic was tested against a copy of the website first, so a bug could never damage the real site.

## Engineering around the model

A model is only as useful as the system around it. This one was built to be measured, observed, and controlled.

Every run writes structured logs and a telemetry record. Each record carries a content hash of the prompts that produced it, so results from different prompt versions are never compared as if they were the same. Claims are tracked back to the exact source chunk that supports them, which makes the grounding auditable rather than a matter of trust.

The application runs in Docker as a non-root user, with the vector database sealed off on a private network. The live demo is served over HTTPS through Caddy, and the publishing step is a separate, deliberate human action, which preserves the rule that the agent never publishes on its own. A single-page web interface shows the pipeline running in real time, streaming each step as it completes, with both review gates built into the same screen.

## What this project is really about

This is not a demonstration that an AI can write an article. It is a demonstration of a way to build AI systems where the model is one component inside a disciplined process: measured honestly, tested before it is trusted, kept under human control, and shipped to real infrastructure. The work that mattered was the metric that tells the truth instead of a flattering one, the experiments that killed bad ideas cheaply, the bugs found by doubting numbers that looked fine, and the choice to keep a person at the controls from the first draft to the moment an article goes live.

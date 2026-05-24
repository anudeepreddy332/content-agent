# html_template.md — Article Page Generation Spec
# Used by: html_gen_node in agent/nodes.py
# Derived from: supervised-learning-models.html + dev_agent.md
#
# RULES FOR html_gen_node:
# - Inject all PLACEHOLDER values from AgentState before returning
# - Do not add any CSS classes not defined in the <style> block below
# - Do not invent content — use draft_sections exactly
# - Do not hardcode hex colors except inside .sl-code-block (dark terminal background)
# - All external links must have target="_blank" rel="noopener noreferrer"
# - The <nav> and <footer> blocks below are verbatim from the live site — do not alter them
# - Validate output contains: <!DOCTYPE html>, id="main", <h1>, ← Back to Learning Log
# - The ld+json @type and @context keys must use the full key names shown below

---

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- MUST BE FIRST: Anti-flicker theme script -->
    <script>
        (function() {
            const t = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', t);
        })();
    </script>

    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <!-- SEO block — all placeholders must be filled before output -->
    <title>{{TOPIC}} | The Machinist</title>
    <meta name="description" content="{{META_DESCRIPTION}}">
    <meta name="author" content="Anudeep Reddy">

    <meta property="og:type" content="website">
    <meta property="og:url" content="https://themachinist.org/{{SLUG}}.html">
    <meta property="og:title" content="{{TOPIC}} | The Machinist">
    <meta property="og:description" content="{{META_DESCRIPTION}}">
    <meta property="og:image" content="https://themachinist.org/og-default.png">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{{TOPIC}} | The Machinist">
    <meta name="twitter:description" content="{{META_DESCRIPTION}}">
    <meta name="twitter:image" content="https://themachinist.org/og-default.png">

    <link rel="canonical" href="https://themachinist.org/{{SLUG}}.html">

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <!-- Shared styles -->
    <link rel="stylesheet" href="./shared.css">

    <!-- Page-specific styles — do not duplicate anything from shared.css -->
    <style>
        html { scroll-behavior: smooth; }

        /* Orange accent tokens — matches supervised-learning-models.html exactly */
        :root {
            --orange: #f97316;
            --orange-subtle: rgba(249,115,22,0.08);
            --orange-border: rgba(249,115,22,0.25);
            --muted: #6b7280;
            --border: #e5e7eb;
            --card: #f9fafb;
        }
        [data-theme="dark"] {
            --orange-subtle: rgba(249,115,22,0.1);
            --orange-border: rgba(249,115,22,0.3);
            --muted: #9ca3af;
            --border: #27272a;
            --card: #18181b;
        }

        /* ── Page header ── */
        .sl-header {
            text-align: left;
            padding: calc(var(--spacing-xl) + 60px) var(--spacing-md) var(--spacing-lg);
            background: var(--section-bg);
            border-bottom: 1px solid var(--border);
        }
        .sl-header .container { max-width: 720px; margin: 0 auto; }

        /* ── Breadcrumb ── */
        .sl-breadcrumb {
            display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
            font-size: 0.78rem; color: var(--muted); margin-bottom: 20px;
        }
        .sl-breadcrumb a { color: var(--muted); text-decoration: none; }
        .sl-breadcrumb a:hover { color: var(--text-color, #111); }
        .sl-breadcrumb-sep { opacity: 0.4; }

        /* ── Series label above h1 ── */
        .sl-series-label {
            font-size: 0.72rem; font-weight: 700;
            letter-spacing: 0.12em; text-transform: uppercase;
            color: var(--muted); margin-bottom: 12px;
        }

        /* ── Page title — Tier 2 (Learning Log) ── */
        .sl-header h1 {
            font-size: clamp(1.6rem, 4vw, 2.4rem);
            font-weight: 800; letter-spacing: -0.025em;
            line-height: 1.2; margin-bottom: 14px;
        }

        /* ── Subtitle ── */
        .sl-subtitle {
            font-size: 0.98rem; color: var(--text-secondary);
            line-height: 1.75; max-width: 560px; margin: 0 0 24px;
        }

        /* ── Meta row (read time + difficulty badge) ── */
        .sl-meta {
            display: flex; gap: 18px; flex-wrap: wrap;
            align-items: center; padding-top: 18px;
            border-top: 1px solid var(--border);
            font-size: 0.78rem; color: var(--muted);
        }
        .sl-meta span { display: flex; align-items: center; gap: 5px; }
        .meta-badge {
            font-size: 0.68rem; font-weight: 700;
            letter-spacing: 0.08em; text-transform: uppercase;
            padding: 3px 8px; border-radius: 3px;
            background: var(--orange-subtle);
            border: 1px solid var(--orange-border);
            color: var(--orange);
        }

        /* ── Article body ── */
        .sl-body {
            max-width: 720px;
            margin: 0 auto;
            padding: 52px 24px 80px;
        }
        .sl-body p {
            font-size: 0.96rem; line-height: 1.82;
            color: var(--text-color, #374151); margin-bottom: 18px;
        }
        .sl-body h2 {
            font-size: 1.2rem; font-weight: 700;
            margin: 44px 0 14px; letter-spacing: -0.01em;
            color: var(--text-color, #111);
        }
        .sl-body h3 {
            font-size: 1rem; font-weight: 700;
            margin: 28px 0 10px; color: var(--text-color, #111);
        }
        .sl-body strong { color: var(--text-color, #111); }
        .sl-body ul, .sl-body ol { padding-left: 20px; margin-bottom: 18px; }
        .sl-body li {
            font-size: 0.96rem; line-height: 1.78;
            color: var(--text-color, #374151); margin-bottom: 5px;
        }

        /* ── Definition box (used for Problem Framing) ── */
        .sl-definition {
            border-left: 3px solid var(--text-color, #111);
            background: var(--section-bg);
            border-radius: 0 6px 6px 0;
            padding: 18px 22px; margin: 22px 0 32px;
        }
        .sl-definition p { margin: 0; font-size: 0.95rem; line-height: 1.78; }

        /* ── Callout boxes ── */
        .callout {
            border-radius: 6px; padding: 16px 20px;
            margin: 22px 0; border: 1px solid;
        }
        .callout-label {
            font-size: 0.68rem; font-weight: 800;
            letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px;
        }
        .callout p { margin: 0; font-size: 0.875rem; line-height: 1.72; color: var(--text-color, #374151); }

        /* Orange key-takeaway callout — used for Takeaways section */
        .callout-key {
            background: var(--orange-subtle);
            border-color: var(--orange-border);
            border-left: 3px solid var(--orange);
        }
        .callout-key .callout-label { color: var(--orange); }

        /* Purple info callout — available for use in technical dive */
        .callout-info {
            background: rgba(139,92,246,0.06);
            border-color: rgba(139,92,246,0.2);
            border-left: 3px solid #8b5cf6;
        }
        .callout-info .callout-label { color: #8b5cf6; }

        /* ── Code blocks — dark terminal style ── */
        .sl-code-block {
            background: #1a1a1a;
            border-radius: 7px;
            margin: 22px 0 28px;
            overflow: hidden;
        }
        .sl-code-label {
            background: #111;
            padding: 7px 16px;
            font-size: 0.68rem; font-weight: 700;
            letter-spacing: 0.1em; text-transform: uppercase;
            color: #6b7280;
            border-bottom: 1px solid #2a2a2a;
        }
        .sl-code-block pre {
            margin: 0; padding: 18px 20px;
            overflow-x: auto; white-space: pre;
        }
        .sl-code-block code {
            font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
            font-size: 0.82rem; line-height: 1.7;
            color: #d4d4d4;
        }

        /* ── Inline code ── */
        .sl-body code:not(.sl-code-block code) {
            background: var(--section-bg);
            border: 1px solid var(--border);
            border-radius: 3px;
            padding: 1px 5px;
            font-size: 0.85em;
            color: var(--text-color);
            font-family: 'SF Mono', 'Fira Code', monospace;
        }

        /* ── Horizontal rule ── */
        .sl-hr {
            border: none; border-top: 1px solid var(--border); margin: 38px 0;
        }

        /* ── Sources section ── */
        .sl-sources {
            border-top: 1px solid var(--border);
            margin-top: 52px; padding-top: 28px;
        }
        .sl-sources-label {
            font-size: 0.72rem; font-weight: 700;
            letter-spacing: 0.12em; text-transform: uppercase;
            color: var(--muted); margin-bottom: 14px;
        }
        .sl-sources ol {
            padding-left: 20px; margin: 0;
        }
        .sl-sources li {
            font-size: 0.82rem; color: var(--muted);
            line-height: 1.65; margin-bottom: 7px;
        }
        .sl-sources a {
            color: var(--muted); text-decoration: underline;
            text-underline-offset: 2px;
        }
        .sl-sources a:hover { color: var(--text-color, #111); }

        /* ── Mobile ── */
        @media (max-width: 640px) {
            .sl-body { padding: 36px 16px 60px; }
            .sl-header { padding: calc(var(--spacing-xl, 64px) + 60px) 16px 36px; }
        }
    </style>
</head>
<body>
    <!-- Skip link (accessibility) -->
    <a href="#main" class="skip-link">Skip to main content</a>

    <!-- Nav — verbatim from supervised-learning-models.html. Do not alter. -->
    <nav id="nav" role="navigation" aria-label="Main navigation">
        <div class="container">
            <div class="nav-container">
                <button class="hamburger" aria-label="Toggle mobile menu" aria-expanded="false">
                    <span></span><span></span><span></span>
                </button>
                <ul>
                    <li><a href="index.html#projects">Projects</a></li>
                    <li><a href="index.html#about">About</a></li>
                    <li><a href="index.html#skills">Skills</a></li>
                    <li><a href="index.html#learning-log">Learning Log</a></li>
                    <li><a href="index.html#contact">Contact</a></li>
                </ul>
                <button class="theme-toggle" id="themeToggle" aria-label="Toggle dark mode">
                    <div class="toggle-slider">
                        <svg class="toggle-icon sun-icon" viewBox="0 0 24 24" style="display:block;">
                            <circle cx="12" cy="12" r="5"/>
                            <line x1="12" y1="1" x2="12" y2="3" stroke="currentColor" stroke-width="2"/>
                            <line x1="12" y1="21" x2="12" y2="23" stroke="currentColor" stroke-width="2"/>
                            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" stroke="currentColor" stroke-width="2"/>
                            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" stroke="currentColor" stroke-width="2"/>
                            <line x1="1" y1="12" x2="3" y2="12" stroke="currentColor" stroke-width="2"/>
                            <line x1="21" y1="12" x2="23" y2="12" stroke="currentColor" stroke-width="2"/>
                            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" stroke="currentColor" stroke-width="2"/>
                            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" stroke="currentColor" stroke-width="2"/>
                        </svg>
                        <svg class="toggle-icon moon-icon" viewBox="0 0 24 24" style="display:none;">
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                        </svg>
                    </div>
                </button>
            </div>
        </div>
        <div class="mobile-menu">
            <div class="mobile-menu-header">
                <button class="close-menu" aria-label="Close menu">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>
            </div>
            <a href="index.html#projects">Projects</a>
            <a href="index.html#about">About</a>
            <a href="index.html#skills">Skills</a>
            <a href="index.html#learning-log">Learning Log</a>
            <a href="index.html#contact">Contact</a>
        </div>
    </nav>

    <!-- Page header — inject TOPIC, SERIES_LABEL, SERIES_CONTEXT, META_DESCRIPTION, DIFFICULTY -->
    <header class="sl-header">
        <div class="container">
            <!-- Breadcrumb: Learning Log / {{BREADCRUMB_SECTION}} / {{TOPIC}} -->
            <nav class="sl-breadcrumb" aria-label="Breadcrumb">
                <a href="index.html#learning-log">Learning Log</a>
                <span class="sl-breadcrumb-sep">/</span>
                <span>{{BREADCRUMB_SECTION}}</span>
                <span class="sl-breadcrumb-sep">/</span>
                <span style="color: var(--text-color, #111); font-weight: 500;">{{TOPIC}}</span>
            </nav>

            <!-- Series label above title -->
            <div class="sl-series-label">{{SERIES_LABEL}}</div>

            <h1>{{TOPIC}}</h1>

            <p class="sl-subtitle">{{META_DESCRIPTION}}</p>

            <div class="sl-meta">
                <span>
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                    </svg>
                    {{READ_TIME}} min read
                </span>
                <span class="meta-badge">{{DIFFICULTY}}</span>
            </div>
        </div>
    </header>

    <!-- Main content -->
    <main id="main">
        <div class="sl-body">

            <!-- ── Section 1: Problem Framing ── -->
            <h2>What is {{TOPIC_SHORT}} and why does it matter?</h2>

            <div class="sl-definition">
                <p>{{PROBLEM_FRAMING}}</p>
            </div>

            <hr class="sl-hr">

            <!-- ── Section 2: Technical Deep-Dive ── -->
            <h2>How it works</h2>

            {{TECHNICAL_DIVE}}
            <!-- html_gen_node: render technical_dive as paragraphs.
                 Wrap any inline math or notation in <code> tags.
                 Use <h3> for any subsections within the dive.
                 Use .callout.callout-info for any "Note" or "Warning" asides.
                 Example inline math: w = w - lr * grad → <code>w = w - lr * grad</code> -->

            <hr class="sl-hr">

            <!-- ── Section 3: Code ── -->
            <h2>In practice</h2>

            {{CODE_SNIPPETS}}
            <!-- html_gen_node: render each code block from code_snippets as a .sl-code-block.
                 Detect the language label from the markdown fence (```python → "Python").
                 Preserve all whitespace and indentation exactly.
                 Structure for each block:
                 <div class="sl-code-block">
                     <div class="sl-code-label">Python</div>
                     <pre><code>...code here...</code></pre>
                 </div>
                 If multiple code blocks exist, render each one separately with a <p> of
                 explanatory text between them if the draft provides it. -->

            <hr class="sl-hr">

            <!-- ── Section 4: Takeaways ── -->
            <h2>Key takeaways</h2>

            <div class="callout callout-key">
                <div class="callout-label">What you now know</div>
                {{TAKEAWAYS}}
                <!-- html_gen_node: render takeaways as a <ul> inside the callout div.
                     Each bullet from draft_sections['takeaways'] becomes a <li>.
                     Split on newline. Do not add a <p> wrapper — put <ul> directly.
                     Example:
                     <ul>
                         <li>Gradient descent requires a differentiable loss function.</li>
                         <li>Learning rate is the most important hyperparameter to tune first.</li>
                     </ul> -->
            </div>

            <!-- ── Sources ── -->
            <div class="sl-sources">
                <div class="sl-sources-label">Sources</div>
                <ol>
                    {{SOURCES}}
                    <!-- html_gen_node: render verified sources from grounding_report.
                         Only include sources with status "verified" or "weak" — skip "unverified".
                         Deduplicate by URL. Format each as:
                         <li><a href="{{source_url}}" target="_blank" rel="noopener noreferrer">{{title or domain}}</a></li>
                         If no verified sources exist, render:
                         <li>Sources retrieved via Tavily web search.</li> -->
                </ol>
            </div>

            <!-- Back link — always Learning Log for content-agent articles -->
            <div style="margin-top: 52px;">
                <a href="index.html#learning-log" class="back-link">← Back to Learning Log</a>
            </div>

        </div>
    </main>

    <!-- Footer — verbatim from supervised-learning-models.html. Do not alter. -->
    <footer>
        <div class="container">
            <div class="footer-links">
                <a href="mailto:anudeepreddy332@gmail.com" aria-label="Email">
                    <svg viewBox="0 0 24 24">
                        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                        <polyline points="22,6 12,13 2,6"/>
                    </svg>
                </a>
                <a href="https://linkedin.com/in/anudeep-reddy-mutyala" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                    <svg viewBox="0 0 24 24">
                        <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/>
                        <rect x="2" y="9" width="4" height="12"/>
                        <circle cx="4" cy="4" r="2"/>
                    </svg>
                </a>
                <a href="https://github.com/anudeepreddy332" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                    <svg viewBox="0 0 24 24">
                        <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.464-1.11-1.464-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836a9.59 9.59 0 012.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
                    </svg>
                </a>
            </div>
            <p class="copyright">&copy; 2026 The Machinist. All rights reserved.</p>
        </div>
    </footer>

    <!-- Shared scripts — must be at end of body, after all content -->
    <script src="./shared.js"></script>

    <!-- LD+JSON structured data — inject TOPIC and SLUG -->
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Article",
      "headline": "{{TOPIC}} | The Machinist",
      "author": {
        "@type": "Person",
        "name": "Anudeep Reddy",
        "url": "https://themachinist.org"
      },
      "publisher": {
        "@type": "Organization",
        "name": "The Machinist",
        "url": "https://themachinist.org"
      },
      "url": "https://themachinist.org/{{SLUG}}.html"
    }
    </script>
</body>
</html>
```

---

## Placeholder Reference

All `{{PLACEHOLDER}}` values are injected by html_gen_node from AgentState:

| Placeholder | Source | Notes |
|---|---|---|
| `{{TOPIC}}` | `state["topic"]` | e.g. "Gradient Descent" |
| `{{TOPIC_SHORT}}` | first word(s) of topic | e.g. "gradient descent" (lowercase) |
| `{{SLUG}}` | `state["slug"]` | e.g. "gradient-descent" |
| `{{META_DESCRIPTION}}` | first 155 chars of `problem_framing` | Strip to sentence boundary. No quotes inside. |
| `{{SERIES_LABEL}}` | Derived from `series_context` | e.g. "Concept Exploration" or "Agentic AI" |
| `{{BREADCRUMB_SECTION}}` | Same as SERIES_LABEL | e.g. "Concept Exploration" |
| `{{DIFFICULTY}}` | Hardcoded: "Intermediate" | All content-agent articles are Intermediate |
| `{{READ_TIME}}` | `len(draft_markdown.split()) // 200` | Words per minute estimate, minimum 5 |
| `{{PROBLEM_FRAMING}}` | `state["draft_sections"]["problem_framing"]` | Render as plain text inside `<p>` |
| `{{TECHNICAL_DIVE}}` | `state["draft_sections"]["technical_dive"]` | Render as `<p>` blocks, `<h3>` subsections |
| `{{CODE_SNIPPETS}}` | `state["draft_sections"]["code_snippets"]` | Parse markdown fences → `.sl-code-block` |
| `{{TAKEAWAYS}}` | `state["draft_sections"]["takeaways"]` | Split on newline → `<ul><li>` inside callout |
| `{{SOURCES}}` | `state["grounding_report"]` + `state["web_sources"]` | Verified/weak only, deduped by URL |

---

## html_gen_node Rendering Rules

### technical_dive rendering
- Split on double newline into paragraphs → wrap each in `<p>` tags
- Lines starting with `###` → `<h3>`
- Lines starting with `##` → skip (section heading is already injected above)
- Inline code (backtick-wrapped) → `<code>` tags
- `**bold**` → `<strong>`
- Bullet lists (`- item`) → `<ul><li>` blocks
- Inline math notation (e.g. `w = w - lr * grad`) → `<code>w = w - lr * grad</code>`
- "Note:" or "Warning:" paragraphs → wrap in `.callout.callout-info` div

### code_snippets rendering
- Split on triple-backtick fences
- Extract language label from opening fence (e.g. ` ```python ` → label "Python")
- If no language label, use label "Code"
- Escape HTML entities inside `<code>`: `<` → `&lt;`, `>` → `&gt;`, `&` → `&amp;`
- Preserve all whitespace, newlines, and indentation exactly as written
- Multiple code blocks: render each `.sl-code-block` sequentially
- Any prose between code blocks in the draft: render as `<p>` between blocks

### takeaways rendering
- The takeaways field is a plain string with bullets separated by newline
- Strip leading `- ` or `• ` or `* ` from each line
- Skip empty lines
- Wrap each as `<li>` inside a `<ul>` placed directly inside the `.callout.callout-key` div
- Do not add a `<p>` wrapper around the `<ul>`

### sources rendering
- Collect all entries from `grounding_report` where `status` is "verified" or "weak"
- Deduplicate by `source_url`
- For each, use the domain name as link text if no title is available
- If `grounding_report` is empty or has no verified/weak entries:
  use the top 3 `web_sources` by score instead
- If `web_sources` is also empty: render the single fallback line shown in the comment
- All source links: `target="_blank" rel="noopener noreferrer"`

---

## Pre-Ship Checklist (run programmatically in html_gen_node)

```python
def validate_html(html: str, slug: str) -> list[str]:
    errors = []
    if "<!DOCTYPE html>" not in html:
        errors.append("Missing DOCTYPE")
    if 'id="main"' not in html:
        errors.append('Missing id="main"')
    if "<h1>" not in html:
        errors.append("Missing h1")
    if "← Back to Learning Log" not in html:
        errors.append("Missing back link")
    if f'href="https://themachinist.org/{slug}.html"' not in html:
        errors.append("Missing or wrong canonical URL")
    if 'rel="noopener noreferrer"' not in html:
        errors.append("External links missing rel=noopener")
    if "@context" not in html:
        errors.append("Missing ld+json @context")
    if "skip-link" not in html:
        errors.append("Missing skip link")
    return errors
```

Log all errors to `state["error_log"]`. Do not abort if errors exist —
log them, return the HTML anyway, and let HITL see the warnings.
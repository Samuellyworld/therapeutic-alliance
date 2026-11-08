"""Rebuild the README's self-contained 3D-style architecture SVG (standard library only)."""
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="920" viewBox="0 0 1600 920" role="img" aria-labelledby="title desc">
<title id="title">Fictional notes to language insights</title>
<desc id="desc">A fictional CSV enters spaCy preprocessing. Original text feeds DistilBERT sentiment and DistilRoBERTa emotion classifiers in parallel. All model scores join linguistic features in CSV, charts, HTML, and a provenance manifest.</desc>
<defs>
 <linearGradient id="bg" x2="1" y2="1"><stop stop-color="#0b182a"/><stop offset="1" stop-color="#121629"/></linearGradient>
 <linearGradient id="card" x2=".8" y2="1"><stop stop-color="#24364c"/><stop offset="1" stop-color="#152238"/></linearGradient>
 <linearGradient id="line" x2="1"><stop stop-color="#59e3c3"/><stop offset="1" stop-color="#91a3ff"/></linearGradient>
 <radialGradient id="halo"><stop stop-color="#245c79" stop-opacity=".27"/><stop offset="1" stop-color="#245c79" stop-opacity="0"/></radialGradient>
 <pattern id="grid" width="36" height="36" patternUnits="userSpaceOnUse"><path d="M36 0H0V36" fill="none" stroke="#93adc7" stroke-opacity=".045"/></pattern>
 <filter id="shadow" x="-40%" y="-40%" width="190%" height="210%"><feDropShadow dx="0" dy="20" stdDeviation="16" flood-color="#020812" flood-opacity=".6"/></filter>
 <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#8facd3" stroke-width="1.5"/></marker>
</defs>
<rect width="1600" height="920" rx="28" fill="url(#bg)"/>
<rect width="1600" height="920" rx="28" fill="url(#grid)"/>
<ellipse cx="760" cy="420" rx="800" ry="380" fill="url(#halo)"/>
<g font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif">
<text x="70" y="68" fill="#62dec4" font-size="15" font-weight="700" letter-spacing="3">THERAPEUTIC ALLIANCE / PIPELINE</text>
<text x="70" y="125" fill="#f3f7fc" font-size="43" font-weight="750">Fictional notes. Observable language.</text>
<text x="70" y="162" fill="#9bacbf" font-size="21">A local Python workflow for sentiment, emotion, and linguistic exploration.</text>
''']


def path(d, dashed=False):
    dash = 'stroke-dasharray="7 7"' if dashed else ''
    parts.append(f'<path d="{d}" fill="none" stroke="url(#line)" stroke-width="3" {dash} marker-end="url(#arrow)"/>')


# Connectors are behind the raised cards.
path('M350 371H440')
path('M720 350H760Q780 350 780 330V293Q780 275 800 275H830')
path('M780 350V493Q780 510 800 510H830')
path('M1120 275H1150Q1170 275 1170 295V360H1210')
path('M1120 510H1150Q1170 510 1170 490V430H1210')
path('M580 460V646Q580 666 600 666H1325Q1345 666 1345 646V526', True)
parts.append('<text x="760" y="705" fill="#8ba9bc" font-size="17">LINGUISTIC FEATURES · joined by client + session</text>')


def card(x, y, w, h, accent, number, label, title, lines, foot):
    parts.append(f'<g filter="url(#shadow)"><path d="M{x} {y}l14 -14h{w}v{h}l-14 14Z" fill="#101b2d"/>')
    parts.append(f'<path d="M{x} {y}l14 -14h{w}l-14 14Z" fill="{accent}" opacity=".32"/>')
    parts.append(f'<path d="M{x+w} {y}l14 -14v{h}l-14 14Z" fill="{accent}" opacity=".16"/>')
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="url(#card)" stroke="{accent}" stroke-opacity=".48"/></g>')
    parts.append(f'<rect x="{x+20}" y="{y+20}" width="34" height="30" rx="8" fill="{accent}" fill-opacity=".14"/>')
    parts.append(f'<text x="{x+37}" y="{y+41}" text-anchor="middle" fill="{accent}" font-size="15" font-weight="700">{number}</text>')
    parts.append(f'<text x="{x+67}" y="{y+41}" fill="{accent}" font-size="13" font-weight="700" letter-spacing="1.5">{escape(label)}</text>')
    parts.append(f'<text x="{x+22}" y="{y+82}" fill="#f1f5fc" font-size="25" font-weight="700">{escape(title)}</text>')
    for i, line in enumerate(lines):
        parts.append(f'<text x="{x+22}" y="{y+114+i*25}" fill="#aebed0" font-size="17">{escape(line)}</text>')
    parts.append(f'<path d="M{x+22} {y+h-49}H{x+w-22}" stroke="#8197b3" stroke-opacity=".18"/>')
    parts.append(f'<text x="{x+22}" y="{y+h-23}" fill="{accent}" font-size="14">{escape(foot)}</text>')


card(70, 276, 280, 215, '#63e6c2', '01', 'INPUT', 'Fictional notes',
     ['Two clients · ten sessions', 'Validated CSV rows'], 'client_id / session / note')
card(440, 276, 280, 215, '#69c9ee', '02', 'PREPROCESS', 'spaCy',
     ['Tokenize & split sentences', 'Count words & language ratios'], 'Negation + original text preserved')
card(830, 205, 290, 194, '#9fafff', '03A', 'SENTIMENT', 'DistilBERT',
     ['Positive + negative scores'], 'SST-2 · general English model')
card(830, 440, 290, 194, '#c1a3fb', '03B', 'EMOTION', 'DistilRoBERTa',
     ['All seven emotion scores'], 'Chunk long text · weighted mean')
card(1210, 306, 310, 220, '#f4cd83', '04', 'EXPLORE', 'Reports & provenance',
     ['CSV tables + Matplotlib charts', 'Portable HTML + run manifest'], 'Scores describe text, not outcomes')
parts.append('''
<rect x="70" y="735" width="1450" height="112" rx="16" fill="#1b293d" stroke="#344b62"/>
<rect x="70" y="735" width="5" height="112" rx="2" fill="#62dec4"/>
<text x="100" y="773" fill="#e2ebf6" font-size="20" font-weight="650">Designed for learning. Grounded in limitations.</text>
<text x="100" y="805" fill="#a7b8cc" font-size="18">Fictional English text only · Local inference · Recorded model revisions · No clinical conclusions</text>
<text x="70" y="884" fill="#637a95" font-size="14" letter-spacing="1">PYTHON / SPACY / HUGGING FACE / PANDAS / MATPLOTLIB</text>
</g></svg>''')

output = ROOT / 'docs/assets/pipeline.svg'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('\n'.join(parts) + '\n', encoding='utf-8')
print(output)

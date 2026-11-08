"""Rebuild the README's animated monochrome 3D-style architecture SVG (standard library only)."""
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="920" viewBox="0 0 1600 920" role="img" aria-labelledby="title desc">
<title id="title">Fictional notes to language insights — animated monochrome pipeline</title>
<desc id="desc">A fictional CSV enters spaCy preprocessing. Original text feeds DistilBERT sentiment and DistilRoBERTa emotion classifiers in parallel. All model scores join linguistic features in CSV, charts, HTML, and a provenance manifest. Moving white dashes indicate flow direction; card borders highlight each stage. Animation pauses for reduced-motion preferences.</desc>
<style>
@keyframes flow { from { stroke-dashoffset: 0; } to { stroke-dashoffset: -72; } }
@keyframes stage { 0%, 20%, 100% { stroke-opacity: .42; } 7%, 13% { stroke-opacity: 1; } }
.flow { animation: flow 1.8s linear infinite; }
.stage { animation: stage 6s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) {
 .flow, .stage { animation: none; }
}
</style>
<defs>
 <linearGradient id="bg" x2="1" y2="1"><stop stop-color="#080808"/><stop offset="1" stop-color="#141414"/></linearGradient>
 <linearGradient id="card" x2=".8" y2="1"><stop stop-color="#282828"/><stop offset="1" stop-color="#181818"/></linearGradient>
 <linearGradient id="line" x2="1"><stop stop-color="#a8a8a8"/><stop offset="1" stop-color="#ffffff"/></linearGradient>
 <radialGradient id="halo"><stop stop-color="#808080" stop-opacity=".27"/><stop offset="1" stop-color="#808080" stop-opacity="0"/></radialGradient>
 <pattern id="grid" width="36" height="36" patternUnits="userSpaceOnUse"><path d="M36 0H0V36" fill="none" stroke="#aaaaaa" stroke-opacity=".045"/></pattern>
 <filter id="shadow" x="-40%" y="-40%" width="190%" height="210%"><feDropShadow dx="0" dy="20" stdDeviation="16" flood-color="#000000" flood-opacity=".6"/></filter>
 <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#dddddd" stroke-width="1.5"/></marker>
</defs>
<rect width="1600" height="920" rx="28" fill="url(#bg)"/>
<rect width="1600" height="920" rx="28" fill="url(#grid)"/>
<ellipse cx="760" cy="420" rx="800" ry="380" fill="url(#halo)"/>
<g font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif">
<text x="70" y="68" fill="#ffffff" font-size="15" font-weight="700" letter-spacing="3">THERAPEUTIC ALLIANCE / PIPELINE</text>
<text x="70" y="125" fill="#ffffff" font-size="43" font-weight="750">Fictional notes. Observable language.</text>
<text x="70" y="162" fill="#b8b8b8" font-size="21">A local Python workflow for sentiment, emotion, and linguistic exploration.</text>
''']


def path(d, dashed=False):
    dash = 'stroke-dasharray="7 7"' if dashed else ''
    parts.append(f'<path d="{d}" fill="none" stroke="#666666" stroke-width="3" {dash} marker-end="url(#arrow)"/>')
    parts.append(f'<path class="flow" d="{d}" fill="none" stroke="#ffffff" stroke-width="3" stroke-linecap="round" stroke-dasharray="10 26"/>')


# Connectors are behind the raised cards.
path('M350 371H440')
path('M720 350H760Q780 350 780 330V293Q780 275 800 275H830')
path('M780 350V493Q780 510 800 510H830')
path('M1120 275H1150Q1170 275 1170 295V360H1210')
path('M1120 510H1150Q1170 510 1170 490V430H1210')
path('M580 460V646Q580 666 600 666H1325Q1345 666 1345 646V526', True)
parts.append('<text x="760" y="705" fill="#aaaaaa" font-size="17">LINGUISTIC FEATURES · joined by client + session</text>')


def card(x, y, w, h, accent, number, label, title, lines, foot):
    delay = {"01": 0, "02": 1, "03A": 2, "03B": 2, "04": 4}[number]
    parts.append(f'<g filter="url(#shadow)"><path d="M{x} {y}l14 -14h{w}v{h}l-14 14Z" fill="#101010"/>')
    parts.append(f'<path d="M{x} {y}l14 -14h{w}l-14 14Z" fill="{accent}" opacity=".32"/>')
    parts.append(f'<path d="M{x+w} {y}l14 -14v{h}l-14 14Z" fill="{accent}" opacity=".16"/>')
    parts.append(f'<rect class="stage" style="animation-delay:{delay}s" x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="url(#card)" stroke="{accent}" stroke-opacity=".42"/></g>')
    parts.append(f'<rect x="{x+20}" y="{y+20}" width="34" height="30" rx="8" fill="{accent}" fill-opacity=".14"/>')
    parts.append(f'<text x="{x+37}" y="{y+41}" text-anchor="middle" fill="{accent}" font-size="15" font-weight="700">{number}</text>')
    parts.append(f'<text x="{x+67}" y="{y+41}" fill="{accent}" font-size="13" font-weight="700" letter-spacing="1.5">{escape(label)}</text>')
    parts.append(f'<text x="{x+22}" y="{y+82}" fill="#ffffff" font-size="25" font-weight="700">{escape(title)}</text>')
    for i, line in enumerate(lines):
        parts.append(f'<text x="{x+22}" y="{y+114+i*25}" fill="#bdbdbd" font-size="17">{escape(line)}</text>')
    parts.append(f'<path d="M{x+22} {y+h-49}H{x+w-22}" stroke="#999999" stroke-opacity=".18"/>')
    parts.append(f'<text x="{x+22}" y="{y+h-23}" fill="{accent}" font-size="14">{escape(foot)}</text>')


card(70, 276, 280, 215, '#ffffff', '01', 'INPUT', 'Fictional notes',
     ['Two clients · ten sessions', 'Validated CSV rows'], 'client_id / session / note')
card(440, 276, 280, 215, '#ffffff', '02', 'PREPROCESS', 'spaCy',
     ['Tokenize & split sentences', 'Count words & language ratios'], 'Negation + original text preserved')
card(830, 205, 290, 194, '#ffffff', '03A', 'SENTIMENT', 'DistilBERT',
     ['Positive + negative scores'], 'SST-2 · general English model')
card(830, 440, 290, 194, '#ffffff', '03B', 'EMOTION', 'DistilRoBERTa',
     ['All seven emotion scores'], 'Chunk long text · weighted mean')
card(1210, 306, 310, 220, '#ffffff', '04', 'EXPLORE', 'Reports & provenance',
     ['CSV tables + Matplotlib charts', 'Portable HTML + run manifest'], 'Scores describe text, not outcomes')
parts.append('''
<rect x="70" y="735" width="1450" height="112" rx="16" fill="#1c1c1c" stroke="#484848"/>
<rect x="70" y="735" width="5" height="112" rx="2" fill="#ffffff"/>
<text x="100" y="773" fill="#eeeeee" font-size="20" font-weight="650">Designed for learning. Grounded in limitations.</text>
<text x="100" y="805" fill="#b8b8b8" font-size="18">Fictional English text only · Local inference · Recorded model revisions · No clinical conclusions</text>
<text x="70" y="884" fill="#888888" font-size="14" letter-spacing="1">PYTHON / SPACY / HUGGING FACE / PANDAS / MATPLOTLIB</text>
</g></svg>''')

output = ROOT / 'docs/assets/pipeline.svg'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('\n'.join(parts) + '\n', encoding='utf-8')
print(output)

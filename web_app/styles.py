"""Tema visual Streamlit untuk aplikasi demonstrasi penelitian."""

APP_CSS = r"""
<style>
:root {
    --ink: #2a211c;
    --muted: #766a61;
    --paper: #fffdf8;
    --cream: #f6efe3;
    --terracotta: #a84d32;
    --terracotta-dark: #76301f;
    --gold: #c79136;
    --green: #2f6f5e;
    --line: #e8dccb;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(199,145,54,.10), transparent 27rem),
        linear-gradient(180deg, #fbf7f0 0%, #fffdf9 34rem, #fffdf9 100%);
    color: var(--ink);
}

[data-testid="stHeader"] { background: rgba(251,247,240,.78); }
[data-testid="stSidebar"] {
    background: #2d211c;
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color: #f7efe5; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #fff; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.13); }

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.35rem 2.55rem;
    border-radius: 26px;
    color: #fffaf2;
    background: linear-gradient(124deg, #34231d 0%, #713622 62%, #a85832 100%);
    box-shadow: 0 24px 55px rgba(78,44,30,.18);
    margin-bottom: 1.35rem;
}
.hero::after {
    content: "";
    position: absolute;
    width: 310px;
    height: 310px;
    right: -90px;
    top: -115px;
    border: 1px solid rgba(255,255,255,.16);
    border-radius: 50%;
    box-shadow: 0 0 0 34px rgba(255,255,255,.035), 0 0 0 68px rgba(255,255,255,.025);
}
.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #f3d798;
    margin-bottom: .75rem;
}
.hero h1 {
    max-width: 760px;
    margin: 0;
    font-size: clamp(2.1rem, 5vw, 3.55rem);
    line-height: 1.01;
    letter-spacing: -.045em;
    color: #fffaf2;
}
.hero p {
    max-width: 710px;
    margin: 1rem 0 0;
    color: rgba(255,250,242,.79);
    font-size: 1rem;
    line-height: 1.65;
}
.hero-badges { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.25rem; }
.hero-badge {
    padding: .38rem .7rem;
    border-radius: 999px;
    font-size: .75rem;
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.15);
}

.section-kicker {
    margin: 1.7rem 0 .2rem;
    color: var(--terracotta);
    font-size: .75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .13em;
}
.section-title { margin: 0 0 .35rem; font-size: 1.55rem; letter-spacing: -.025em; }
.section-copy { margin: 0 0 1rem; color: var(--muted); }

[data-testid="stFileUploader"] {
    border-radius: 18px;
    background: rgba(255,255,255,.72);
}
[data-testid="stFileUploaderDropzone"] {
    min-height: 150px;
    border: 1.5px dashed #cbb79b;
    background: #fffdf9;
    border-radius: 16px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line) !important;
    border-radius: 19px !important;
    background: rgba(255,255,255,.78);
    box-shadow: 0 10px 30px rgba(75,49,33,.055);
}

.result-card {
    min-height: 245px;
    padding: 1.55rem 1.6rem;
    border-radius: 20px;
    color: white;
    background: linear-gradient(145deg, #2e6a5a, #214c42);
    box-shadow: 0 18px 35px rgba(34,84,70,.18);
}
.result-card.non-batik { background: linear-gradient(145deg, #a34c32, #71301f); }
.result-label { opacity:.75; font-size:.72rem; letter-spacing:.14em; font-weight:800; }
.result-value { font-size:2.25rem; line-height:1.05; font-weight:850; margin:.55rem 0 .4rem; }
.result-score { font-size:1rem; opacity:.9; }
.score-track { height:9px; border-radius:99px; background:rgba(255,255,255,.18); margin:1.25rem 0 .55rem; overflow:hidden; }
.score-fill { height:100%; background:#f1cf87; border-radius:99px; }
.score-axis { display:flex; justify-content:space-between; font-size:.7rem; opacity:.72; }
.model-note { margin-top:1rem; font-size:.74rem; line-height:1.45; opacity:.72; }

.metric-tile {
    padding: 1rem 1.05rem;
    border: 1px solid var(--line);
    background: #fffdf9;
    border-radius: 15px;
    min-height: 112px;
}
.metric-tile .value { font-size:1.45rem; font-weight:800; color:var(--ink); }
.metric-tile .label { color:var(--terracotta); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }
.metric-tile .desc { color:var(--muted); font-size:.75rem; line-height:1.35; margin-top:.35rem; }

.empty-state {
    text-align:center;
    padding:2.4rem 1rem 2rem;
    color:var(--muted);
}
.empty-icon {
    display:inline-grid; place-items:center;
    width:58px; height:58px;
    border-radius:18px;
    background:#f1e5d4;
    color:var(--terracotta);
    font-size:1.6rem;
    margin-bottom:.8rem;
}
.tiny-note { color:var(--muted); font-size:.78rem; line-height:1.5; }

.stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.7rem 1rem; }
.stTabs [aria-selected="true"] { background:#f4eadb; color:var(--terracotta-dark); }

@media (max-width: 700px) {
    .block-container { padding-top:1rem; }
    .hero { padding:1.7rem 1.35rem; border-radius:20px; }
    .hero p { font-size:.92rem; }
    .result-card { min-height:auto; }
}
</style>
"""


def inject_styles(streamlit_module) -> None:
    streamlit_module.markdown(APP_CSS, unsafe_allow_html=True)


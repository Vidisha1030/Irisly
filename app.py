"""
Irisly - Iris species recognition with KNN (Streamlit)
requirements.txt: streamlit, scikit-learn, numpy, pandas
Run: streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Irisly", page_icon="🌸", layout="wide")

SP_COLORS = ["#EF9F27", "#1D9E75", "#7F77DD"]  # setosa, versicolor, virginica
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
DEFAULTS = {"sl": 5.8, "sw": 3.0, "pl": 4.3, "pw": 1.3, "k": 5}

# ------------------------------------------------------------------ styling
st.markdown(
    """
<style>
.stApp { background:#FBEAF0; }
header[data-testid="stHeader"], #MainMenu, footer { display:none; }
.block-container { max-width:1100px; padding-top:1.2rem; padding-bottom:2rem; }
h1,h2,h3,p,span,label,div { color:#4B1528; }

.nav { display:flex; align-items:center; justify-content:space-between; background:#fff; border-radius:12px; padding:10px 20px; margin-bottom:14px; }
.nav .logo { display:flex; align-items:center; gap:10px; font-weight:600; font-size:17px; }
.nav .dot { width:32px; height:32px; border-radius:50%; background:#D4537E; display:flex; align-items:center; justify-content:center; font-size:16px; }
.nav .links { display:flex; align-items:center; gap:22px; font-size:14px; }
.nav .links a { color:#993556; text-decoration:none; }
.nav .links .cta { background:#D4537E; color:#fff !important; padding:7px 16px; border-radius:8px; font-weight:600; }

.hero { background:#F4C0D1; border-radius:14px; padding:30px; display:grid; grid-template-columns:1.4fr 1fr; gap:16px; align-items:center; margin-bottom:14px; }
.hero .tag { display:inline-block; background:#fff; color:#993556; font-size:12px; padding:3px 12px; border-radius:12px; margin-bottom:10px; }
.hero h1 { font-size:34px; font-weight:600; line-height:1.2; margin:0 0 8px; padding:0; }
.hero p { font-size:15px; color:#72243E; line-height:1.6; margin:0 0 16px; }
.btn { display:inline-block; padding:10px 20px; border-radius:8px; font-weight:600; font-size:14px; text-decoration:none; margin-right:10px; }
.btn.p { background:#D4537E; color:#fff !important; } .btn.s { background:#fff; color:#993556 !important; }
.chips { display:flex; gap:8px; margin-top:18px; flex-wrap:wrap; }
.chip { background:#fff; border-radius:10px; padding:7px 16px; text-align:center; }
.chip b { display:block; font-size:18px; color:#993556; } .chip span { font-size:11px; color:#72243E; }

[class*="st-key-card"] { background:#fff; border-radius:14px; padding:1.1rem 1.3rem; }
.ctitle { font-weight:600; font-size:16px; margin-bottom:4px; }
.csub { font-size:12px; color:#993556; margin-bottom:8px; }

.stSlider label p, .stRadio label p { color:#72243E; font-size:13px; }
[data-testid="stSliderThumbValue"] { color:#993556; }
button[kind="primary"] { background:#D4537E; border:none; color:#fff; border-radius:10px; font-weight:600; }
button[kind="primary"]:hover { background:#993556; color:#fff; }
button[kind="secondary"] { background:#fff; border:1px solid #F4C0D1; color:#993556; border-radius:10px; }
button[kind="secondary"]:hover { border-color:#D4537E; color:#D4537E; }
[data-testid="stExpander"] { background:#fff; border:none; border-radius:12px; margin-bottom:6px; }

.result { background:#fff; border:2px solid #D4537E; border-radius:14px; padding:18px 22px; margin-bottom:14px; }
.result .row { display:flex; align-items:center; gap:14px; }
.result .ico { width:52px; height:52px; border-radius:50%; background:#FBEAF0; display:flex; align-items:center; justify-content:center; font-size:26px; }
.result .lab { font-size:12px; color:#993556; } .result .name { font-size:26px; font-weight:600; }
.bar { height:8px; background:#FBEAF0; border-radius:4px; margin-top:4px; } .bar i { display:block; height:100%; background:#D4537E; border-radius:4px; }
.legend { display:flex; gap:14px; font-size:12px; color:#72243E; margin-top:12px; flex-wrap:wrap; }
.legend i { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; }

.steps, .species { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.step { background:#FBEAF0; border-radius:12px; padding:14px 16px; } .step .e { font-size:22px; }
.step b { display:block; margin:6px 0 2px; font-size:14px; } .step span { font-size:12px; color:#72243E; line-height:1.5; }
.sp { background:#fff; border-radius:12px; padding:14px 16px; }
.sp b { font-size:14px; } .sp i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }
.sp p { margin:4px 0 0; font-size:12px; color:#72243E; }

.cm { display:grid; grid-template-columns:80px repeat(3,1fr); gap:4px; font-size:12px; text-align:center; }
.cm .h { color:#72243E; } .cm .r { text-align:right; padding-right:6px; align-self:center; color:#72243E; }
.cm .c { padding:14px 0; border-radius:6px; }

table.data { width:100%; border-collapse:collapse; font-size:13px; }
table.data th { background:#FBEAF0; color:#72243E; text-align:left; padding:8px 10px; font-weight:600; }
table.data td { padding:8px 10px; border-bottom:1px solid #FBEAF0; }

.cta-band { background:#D4537E; border-radius:14px; padding:22px 26px; display:flex; align-items:center; justify-content:space-between; margin:14px 0; }
.cta-band b { color:#fff; font-size:18px; display:block; } .cta-band span { color:#FBEAF0; font-size:13px; }
.cta-band a { background:#fff; color:#993556 !important; padding:10px 18px; border-radius:8px; font-weight:600; text-decoration:none; font-size:14px; }
.foot { display:flex; justify-content:space-between; font-size:12px; color:#993556; padding:4px 8px; }
.foot span { color:#993556; margin-left:16px; }
</style>
""",
    unsafe_allow_html=True,
)


def html(s):
    """Render HTML (lines are joined so Markdown never treats it as a code block)."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


# ------------------------------------------------------------------ data and model
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURES)
    names = [n.capitalize() for n in iris.target_names]
    df["species"] = [names[i] for i in iris.target]
    return iris.data, iris.target, names, df


@st.cache_resource
def get_model(k):
    X, y, _, _ = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return model, float((pred == y_te).mean()), confusion_matrix(y_te, pred), X_tr, y_tr


X, y, SPECIES, df = load_data()

for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def reset():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.session_state.pop("snap", None)


def run_prediction():
    """Runs when 'Predict species' is clicked: saves the current inputs."""
    st.session_state["snap"] = {key: st.session_state[key] for key in DEFAULTS}


# Model for the selected K (used by the hero stats and the performance section)
k = st.session_state["k"]
model, acc, cm, X_tr, y_tr = get_model(k)

# Prediction only exists after the button has been clicked
snap = st.session_state.get("snap")
predicted = snap is not None
sample = neighbors = None
if predicted:
    p_model, _, _, p_Xtr, p_ytr = get_model(snap["k"])
    sample = np.array([[snap["sl"], snap["sw"], snap["pl"], snap["pw"]]])
    pred = int(p_model.predict(sample)[0])
    conf = float(p_model.predict_proba(sample)[0][pred]) * 100
    _, idx = p_model[1].kneighbors(p_model[0].transform(sample))
    neighbors = p_Xtr[idx[0]]
    votes = np.bincount(p_ytr[idx[0]], minlength=3)


# ------------------------------------------------------------------ SVG helpers
def scatter_svg(w=560, h=230):
    px = lambda v: 34 + (v - 0.8) / 6.4 * (w - 54)
    py = lambda v: h - 30 - v / 2.7 * (h - 50)
    s = f'<svg viewBox="0 0 {w} {h}" width="100%"><rect width="{w}" height="{h}" rx="8" fill="#FEF6F9"/>'
    s += f'<line x1="34" y1="{h-30}" x2="{w-10}" y2="{h-30}" stroke="#ED93B1"/><line x1="34" y1="10" x2="34" y2="{h-30}" stroke="#ED93B1"/>'
    for row, label in zip(X, y):
        s += f'<circle cx="{px(row[2]):.1f}" cy="{py(row[3]):.1f}" r="4" fill="{SP_COLORS[label]}" opacity="0.85"/>'
    if sample is not None:
        sx, sy = px(sample[0][2]), py(sample[0][3])
        for n in neighbors:
            s += f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{px(n[2]):.1f}" y2="{py(n[3]):.1f}" stroke="#993556" stroke-dasharray="3 3"/>'
            s += f'<circle cx="{px(n[2]):.1f}" cy="{py(n[3]):.1f}" r="8" fill="none" stroke="#444"/>'
        s += f'<polygon points="{sx:.1f},{sy-9:.1f} {sx+7:.1f},{sy:.1f} {sx:.1f},{sy+9:.1f} {sx-7:.1f},{sy:.1f}" fill="#D4537E"/>'
        s += f'<text x="{sx:.1f}" y="{sy-14:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="#993556">Your flower</text>'
    s += f'<text x="{w/2}" y="{h-8}" text-anchor="middle" font-size="11" fill="#72243E">Petal length (cm)</text></svg>'
    return s


def bars_svg(values, selected):
    s = '<svg viewBox="0 0 320 150" width="100%"><line x1="10" y1="115" x2="310" y2="115" stroke="#ED93B1"/>'
    for i, (kk, a) in enumerate(values.items()):
        x, hgt = 25 + i * 58, a * 85
        color = "#D4537E" if kk == selected else "#F4C0D1"
        s += f'<rect x="{x}" y="{115-hgt:.1f}" width="36" height="{hgt:.1f}" rx="3" fill="{color}"/>'
        s += f'<text x="{x+18}" y="{109-hgt:.1f}" text-anchor="middle" font-size="11" fill="#72243E">{a*100:.0f}%</text>'
        s += f'<text x="{x+18}" y="132" text-anchor="middle" font-size="11" fill="#72243E">K={kk}</text>'
    return s + "</svg>"


FLOWER = """<svg viewBox="0 0 220 220" width="100%" style="max-width:240px;justify-self:center">
<rect x="107" y="120" width="6" height="90" fill="#3B6D11"/>
<ellipse cx="88" cy="175" rx="20" ry="8" fill="#639922" transform="rotate(-25 88 175)"/>
<ellipse cx="132" cy="160" rx="20" ry="8" fill="#639922" transform="rotate(25 132 160)"/>
<g fill="#ED93B1">
<ellipse cx="110" cy="62" rx="20" ry="38"/>
<ellipse cx="110" cy="62" rx="20" ry="38" transform="rotate(60 110 100)"/>
<ellipse cx="110" cy="62" rx="20" ry="38" transform="rotate(120 110 100)"/>
<ellipse cx="110" cy="62" rx="20" ry="38" transform="rotate(180 110 100)"/>
<ellipse cx="110" cy="62" rx="20" ry="38" transform="rotate(240 110 100)"/>
<ellipse cx="110" cy="62" rx="20" ry="38" transform="rotate(300 110 100)"/></g>
<circle cx="110" cy="100" r="16" fill="#D4537E"/><circle cx="110" cy="100" r="7" fill="#FAC775"/></svg>"""

# ------------------------------------------------------------------ navbar + hero
html("""
<div class="nav">
  <div class="logo"><div class="dot">🌸</div>Irisly</div>
  <div class="links">
    <a href="#predictor" target="_self">Predict</a><a href="#how" target="_self">How it works</a>
    <a href="#performance" target="_self">Performance</a><a href="#dataset" target="_self">Dataset</a>
    <a href="#faq" target="_self">FAQ</a><a class="cta" href="#predictor" target="_self">Try it now</a>
  </div>
</div>
""")

html(f"""
<div class="hero">
  <div>
    <span class="tag">Machine learning demo</span>
    <h1>Recognize iris species in seconds</h1>
    <p>A K-nearest neighbors model trained on the classic Iris dataset. Enter four flower measurements and see which species they match.</p>
    <a class="btn p" href="#predictor" target="_self">Start predicting</a>
    <a class="btn s" href="#dataset" target="_self">View the dataset</a>
    <div class="chips">
      <div class="chip"><b>{len(X)}</b><span>samples</span></div>
      <div class="chip"><b>4</b><span>features</span></div>
      <div class="chip"><b>3</b><span>species</span></div>
      <div class="chip"><b>{acc*100:.1f}%</b><span>accuracy</span></div>
    </div>
  </div>
  {FLOWER}
</div>
""")

# ------------------------------------------------------------------ predictor
html('<div id="predictor"></div>')
left, right = st.columns(2, gap="medium")

with left:
    with st.container(key="card_inputs"):
        html('<div class="ctitle">Enter measurements</div>')
        st.slider("Sepal length (cm)", 4.0, 8.0, step=0.1, key="sl")
        st.slider("Sepal width (cm)", 2.0, 4.5, step=0.1, key="sw")
        st.slider("Petal length (cm)", 1.0, 7.0, step=0.1, key="pl")
        st.slider("Petal width (cm)", 0.1, 2.5, step=0.1, key="pw")
        st.radio("K neighbors", [3, 5, 7, 9], horizontal=True, key="k")
        b1, b2 = st.columns([3, 1])
        b1.button("Predict species", type="primary", on_click=run_prediction, use_container_width=True)
        b2.button("Reset", on_click=reset, use_container_width=True)

with right:
    if predicted:
        legend = "".join(
            f'<span><i style="background:{SP_COLORS[i]}"></i>{n} {votes[i]}</span>' for i, n in enumerate(SPECIES)
        )
        html(f"""
        <div class="result">
          <div class="row"><div class="ico">🌸</div>
            <div><div class="lab">Predicted species (K = {snap["k"]})</div><div class="name">Iris {SPECIES[pred].lower()}</div></div></div>
          <div style="display:flex;justify-content:space-between;font-size:13px;margin-top:14px">
            <span>Confidence</span><b style="color:#993556">{conf:.0f}%</b></div>
          <div class="bar"><i style="width:{conf:.0f}%"></i></div>
          <div class="legend">{legend}</div>
        </div>
        """)
    else:
        html("""
        <div class="result">
          <div class="row"><div class="ico">🌸</div>
            <div><div class="lab">Predicted species</div><div class="name">Waiting for input</div></div></div>
          <p style="font-size:13px;color:#72243E;margin:14px 0 0">Set your measurements, then press Predict species to see the result.</p>
        </div>
        """)
    with st.container(key="card_mini"):
        html('<div class="ctitle">Petal length vs width</div>')
        html(scatter_svg(400, 190))

# ------------------------------------------------------------------ how it works
html("""
<div id="how"></div>
<div style="background:#fff;border-radius:14px;padding:20px 22px;margin:14px 0">
  <div class="ctitle" style="margin-bottom:12px">How it works</div>
  <div class="steps">
    <div class="step"><div class="e">📏</div><b>1. Measure</b><span>Enter sepal and petal size in centimeters.</span></div>
    <div class="step"><div class="e">🔎</div><b>2. Find neighbors</b><span>The model finds the K closest known flowers.</span></div>
    <div class="step"><div class="e">🗳️</div><b>3. Vote</b><span>The most common species among them wins.</span></div>
  </div>
</div>
""")

# ------------------------------------------------------------------ performance
html('<div id="performance"></div>')
p1, p2 = st.columns(2, gap="medium")

with p1:
    with st.container(key="card_acc"):
        html('<div class="ctitle">Accuracy by K</div><div class="csub">Test set, 30 samples</div>')
        html(bars_svg({kk: get_model(kk)[1] for kk in (1, 3, 5, 7, 9)}, k))

with p2:
    with st.container(key="card_cm"):
        html('<div class="ctitle">Confusion matrix</div><div class="csub">Rows actual, columns predicted</div>')
        cells = '<div></div>' + "".join(f'<div class="h">{n[:5]}.</div>' for n in SPECIES)
        for i, n in enumerate(SPECIES):
            cells += f'<div class="r">{n}</div>'
            for j in range(3):
                v = int(cm[i][j])
                style = "background:#D4537E;color:#fff;font-weight:600" if i == j else (
                    "background:#F4C0D1" if v else "background:#FBEAF0")
                cells += f'<div class="c" style="{style}">{v}</div>'
        html(f'<div class="cm">{cells}</div>')

# ------------------------------------------------------------------ dataset
html('<div id="dataset"></div>')
with st.container(key="card_data"):
    d1, d2 = st.columns([4, 1])
    d1.markdown('<div class="ctitle">Dataset preview</div>', unsafe_allow_html=True)
    d2.download_button("Download CSV", df.to_csv(index=False), "iris.csv", "text/csv", use_container_width=True)
    preview = df.iloc[[0, 1, 50, 51, 100, 101]]
    rows = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in preview.itertuples(index=False)
    )
    head = "".join(f"<th>{c.replace('_', ' ').capitalize()}</th>" for c in preview.columns)
    html(f'<table class="data"><tr>{head}</tr>{rows}</table>')

html("""
<div class="species" style="margin:14px 0">
  <div class="sp"><i style="background:#EF9F27"></i><b>Setosa</b><p>Small petals, easy to separate.</p></div>
  <div class="sp"><i style="background:#1D9E75"></i><b>Versicolor</b><p>Medium petals, overlaps a little.</p></div>
  <div class="sp"><i style="background:#7F77DD"></i><b>Virginica</b><p>Largest petals and sepals.</p></div>
</div>
""")

# ------------------------------------------------------------------ FAQ
html('<div id="faq"></div>')
with st.container(key="card_faq"):
    html('<div class="ctitle" style="margin-bottom:10px">Frequently asked questions</div>')
    with st.expander("What is K in K-nearest neighbors?"):
        st.write("K is how many of the closest known flowers get a vote. A small K reacts to noise; a large K blurs the boundaries between species.")
    with st.expander("Why are the features scaled?"):
        st.write("KNN works on distances, so every measurement needs a comparable range. StandardScaler stops one feature from dominating the others.")
    with st.expander("Can I use it on other flowers?"):
        st.write("No. The model only knows setosa, versicolor, and virginica. For other flowers you would need a dataset that includes them.")

# ------------------------------------------------------------------ CTA + footer
html("""
<div class="cta-band">
  <div><b>Ready to try it?</b><span>Predict your first iris in under ten seconds.</span></div>
  <a href="#predictor" target="_self">Start predicting</a>
</div>
<div class="foot"><div>Irisly, built with Python, scikit-learn and Streamlit</div><div><span>GitHub</span><span>Dataset source</span><span>Contact</span></div></div>
""")
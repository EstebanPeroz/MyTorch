#!/usr/bin/env python3
"""Regenerate the benchmarks of docs/benchmarks.md (Python 3 standard library only).

    python3 scripts/benchmark.py --dataset dataset/full.txt --workdir /tmp/bench

What it does:
  1. shuffles the dataset (fixed seed) and cuts a held-out TEST set and a TRAIN set that never overlap;
  2. for each configuration, generates a blank network, trains it with `--train --dashboard` and reads the
     curves from the dashboard's `/metrics` endpoint (so the numbers are those of the analyzer itself);
  3. predicts the test set with `--predict` and builds the confusion matrix and per-class metrics;
  4. writes the SVG figures and `results.json` in --out (default docs/benchmarks).

The binaries must be built first (`make`). Training takes a few minutes per configuration.
"""
import argparse
import html
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black", "Stalemate"]
SHORT = ["Nothing", "Chk W", "Chk B", "Mate W", "Mate B", "Stale"]
COLORS = ["#2a6fdb", "#d9541e", "#2a9d5c", "#8a5cc2"]


def variants(base, epochs):
    """The compared configurations, derived from configs/chess.conf."""
    def derive(layers, dropout):
        cfg = json.loads(json.dumps(base))
        cfg["layers"] = [dict(l) for l in layers]
        for layer in cfg["layers"]:
            if not dropout:
                layer.pop("dropout", None)
        cfg["training"]["epochs"] = epochs
        return cfg

    def hidden(neurons):
        return [{"neurons": n, "activation": "relu", "initialization": "he_normal"} for n in neurons] + [
            base["layers"][-1]
        ]

    return [
        ("small 128-64", derive(hidden([128, 64]), False)),
        ("wide 512-256-64", derive(hidden([512, 256, 64]), False)),
        ("wide + dropout 0.2 (chess.conf)", derive(base["layers"], True)),
    ]


def split(dataset, workdir, train_size, test_size):
    lines = list(dict.fromkeys(l.strip() for l in open(dataset) if l.strip()))  # deduplicated, in order
    random.Random(1).shuffle(lines)
    test, train = lines[:test_size], lines[test_size : test_size + train_size]
    for name, rows in (("test.txt", test), ("train.txt", train)):
        with open(os.path.join(workdir, name), "w") as f:
            f.write("\n".join(rows) + "\n")
    return test


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def train(name, cfg, idx, workdir, analyzer, generator):
    conf = os.path.join(workdir, f"cfg{idx}.conf")
    with open(conf, "w") as f:
        json.dump(cfg, f)
    run([generator, conf, "1"], cwd=workdir, stdout=subprocess.DEVNULL)
    blank = os.path.join(workdir, f"cfg{idx}_1.nn")
    out = os.path.join(workdir, f"cfg{idx}_trained.nn")
    print(f"training {name} ...", flush=True)
    proc = subprocess.Popen(
        [analyzer, "--train", "--dashboard", "--save", out, blank, os.path.join(workdir, "train.txt")],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    url = None
    for line in proc.stderr:
        m = re.search(r"dashboard: (http://\S+)", line)
        if m:
            url = m.group(1)
            break
    last = None
    start = time.time()
    while proc.poll() is None and url:
        try:
            last = json.load(urllib.request.urlopen(url + "/metrics", timeout=5))
        except Exception:
            pass
        if last and last.get("status") != "running":
            break
        time.sleep(0.3)
    proc.stderr.read()
    if proc.wait() != 0 or not last:
        sys.exit(f"training of {name} failed")
    last["seconds"] = time.time() - start
    print(f"  {len(last['points'])} epochs in {last['seconds']:.0f}s", flush=True)
    return last, out


def predict(network, workdir, analyzer, test):
    res = run([analyzer, "--predict", network, os.path.join(workdir, "test.txt")], capture_output=True, text=True)
    predicted = res.stdout.splitlines()
    assert len(predicted) == len(test)
    matrix = [[0] * 6 for _ in range(6)]
    for row, pred in zip(test, predicted):
        expected = next(l for l in LABELS if row.endswith(" " + l))
        matrix[LABELS.index(expected)][LABELS.index(pred.strip())] += 1
    return matrix


def per_class(matrix):
    out = []
    for i in range(6):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(matrix[r][i] for r in range(6))
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out.append({"label": LABELS[i], "support": support, "precision": p, "recall": r, "f1": f1})
    return out


# ---------------------------------------------------------------- SVG

STYLE = (
    "text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;fill:#222}"
    ".mut{fill:#666}.grid{stroke:#ddd;stroke-width:1}.ax{stroke:#888;stroke-width:1}"
)


def svg(w, h, body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{html.escape(title)}"><style>{STYLE}</style>'
        f'<rect width="{w}" height="{h}" fill="#fff"/>'
        f'<text x="{w / 2}" y="24" text-anchor="middle" font-size="16" font-weight="600">{html.escape(title)}</text>'
        f"{body}</svg>\n"
    )


def nice_ticks(lo, hi, n=5):
    step = (hi - lo) / n
    mag = 10 ** (len(str(int(step * 1000))) - 4) if step else 1
    for m in (1, 2, 2.5, 5, 10):
        if step <= m * mag:
            step = m * mag
            break
    start = int(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.01:
        if v >= lo - step * 0.01:
            ticks.append(round(v, 6))
        v += step
    return ticks or [lo, hi]


def line_chart(title, series, ylabel, ymin=None, ymax=None, w=720, h=420, pct=False):
    """series: list of (label, color, dash, [values per epoch])."""
    ml, mr, mt, mb = 64, 20, 44, 92
    pw, ph = w - ml - mr, h - mt - mb
    vals = [v for s in series for v in s[3] if v is not None]
    lo = min(vals) if ymin is None else ymin
    hi = max(vals) if ymax is None else ymax
    pad = (hi - lo) * 0.05 or 0.01
    lo, hi = (lo if ymin is not None else lo - pad), (hi if ymax is not None else hi + pad)
    n = max(len(s[3]) for s in series)
    x = lambda i: ml + (pw * i / (n - 1 if n > 1 else 1))
    y = lambda v: mt + ph * (1 - (v - lo) / (hi - lo))
    b = []
    for t in nice_ticks(lo, hi):
        label = f"{t * 100:.0f}%" if pct else f"{t:g}"
        b.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        b.append(f'<text class="mut" x="{ml - 8}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="12">{label}</text>')
    step = max(1, n // 10)
    for i in range(0, n, step):
        b.append(f'<text class="mut" x="{x(i):.1f}" y="{mt + ph + 18}" text-anchor="middle" font-size="12">{i + 1}</text>')
    b.append(f'<line class="ax" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    b.append(f'<line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt + ph}"/>')
    b.append(f'<text class="mut" x="{ml + pw / 2}" y="{mt + ph + 38}" text-anchor="middle" font-size="12">epoch</text>')
    b.append(f'<text class="mut" transform="translate(16 {mt + ph / 2}) rotate(-90)" text-anchor="middle" font-size="12">{ylabel}</text>')
    for k, (label, color, dash, values) in enumerate(series):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values) if v is not None)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        b.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{d} points="{pts}"/>')
        lx = ml + (k % 2) * (pw / 2)
        ly = mt + ph + 56 + (k // 2) * 18
        b.append(f'<line x1="{lx}" x2="{lx + 26}" y1="{ly - 4}" y2="{ly - 4}" stroke="{color}" stroke-width="2"{d}/>')
        b.append(f'<text x="{lx + 32}" y="{ly}" font-size="12">{html.escape(label)}</text>')
    return svg(w, h, "".join(b), title)


def confusion_chart(title, matrix, w=620, h=600):
    ml, mt, cell = 130, 90, 74
    b = [f'<text class="mut" x="{ml + 3 * cell}" y="{mt - 44}" text-anchor="middle" font-size="12">predicted</text>']
    for j in range(6):
        b.append(f'<text x="{ml + j * cell + cell / 2}" y="{mt - 12}" text-anchor="middle" font-size="12">{SHORT[j]}</text>')
    b.append(f'<text class="mut" transform="translate(16 {mt + 3 * cell}) rotate(-90)" text-anchor="middle" font-size="12">expected</text>')
    for i in range(6):
        total = sum(matrix[i])
        b.append(f'<text x="{ml - 8}" y="{mt + i * cell + cell / 2 + 4}" text-anchor="end" font-size="12">{LABELS[i]} ({total})</text>')
        for j in range(6):
            share = matrix[i][j] / total if total else 0
            fill = f"rgb({int(255 - share * 215)},{int(255 - share * 150)},{int(255 - share * 20)})"
            ink = "#fff" if share > 0.55 else "#222"
            cx, cy = ml + j * cell, mt + i * cell
            b.append(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" fill="{fill}" stroke="#fff"/>')
            b.append(f'<text x="{cx + cell / 2}" y="{cy + cell / 2 - 2}" text-anchor="middle" font-size="14" font-weight="600" style="fill:{ink}">{matrix[i][j]}</text>')
            b.append(f'<text x="{cx + cell / 2}" y="{cy + cell / 2 + 15}" text-anchor="middle" font-size="11" style="fill:{ink}">{share * 100:.1f}%</text>')
    b.append(f'<text class="mut" x="{w / 2}" y="{h - 14}" text-anchor="middle" font-size="12">Cells show counts and the share of the expected row (the diagonal is the recall).</text>')
    return svg(w, h, "".join(b), title)


def bars_chart(title, groups, w=760, h=420):
    """groups: list of (series label, color, [(class label, value)])."""
    ml, mr, mt, mb = 56, 20, 44, 96
    pw, ph = w - ml - mr, h - mt - mb
    classes = [c for c, _ in groups[0][2]]
    gw = pw / len(classes)
    bw = gw * 0.7 / len(groups)
    y = lambda v: mt + ph * (1 - v)
    b = []
    for t in (0, 0.25, 0.5, 0.75, 1):
        b.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        b.append(f'<text class="mut" x="{ml - 8}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="12">{t * 100:.0f}%</text>')
    for c, cls in enumerate(classes):
        gx = ml + c * gw + gw * 0.15
        b.append(f'<text x="{ml + c * gw + gw / 2}" y="{mt + ph + 18}" text-anchor="middle" font-size="12">{html.escape(cls)}</text>')
        for k, (_, color, vals) in enumerate(groups):
            v = vals[c][1]
            b.append(f'<rect x="{gx + k * bw:.1f}" y="{y(v):.1f}" width="{bw - 2:.1f}" height="{ph * v:.1f}" fill="{color}"/>')
            b.append(f'<text x="{gx + k * bw + bw / 2 - 1:.1f}" y="{y(v) - 4:.1f}" text-anchor="middle" font-size="10">{v * 100:.0f}</text>')
    b.append(f'<line class="ax" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    for k, (label, color, _) in enumerate(groups):
        lx = ml + k * (pw / len(groups))
        b.append(f'<rect x="{lx}" y="{mt + ph + 44}" width="14" height="10" fill="{color}"/>')
        b.append(f'<text x="{lx + 20}" y="{mt + ph + 54}" font-size="12">{html.escape(label)}</text>')
    return svg(w, h, "".join(b), title)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset/full.txt"))
    ap.add_argument("--workdir", required=True, help="scratch directory for splits and trained networks")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/benchmarks"))
    ap.add_argument("--train-size", type=int, default=100000)
    ap.add_argument("--test-size", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=40, help="most epochs per configuration")
    ap.add_argument("--analyzer", default=os.path.join(ROOT, "target/release/my_torch_analyzer"))
    ap.add_argument("--generator", default=os.path.join(ROOT, "target/release/my_torch_generator"))
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    base = json.load(open(os.path.join(ROOT, "configs/chess.conf")))
    test = split(a.dataset, a.workdir, a.train_size, a.test_size)

    results = []
    for idx, (name, cfg) in enumerate(variants(base, a.epochs)):
        metrics, network = train(name, cfg, idx, a.workdir, a.analyzer, a.generator)
        matrix = predict(network, a.workdir, a.analyzer, test)
        points = metrics["points"]
        results.append({
            "name": name, "points": points, "matrix": matrix, "classes": per_class(matrix),
            "test_accuracy": sum(matrix[i][i] for i in range(6)) / len(test),
            "seconds": metrics["seconds"], "layers": metrics.get("layers"),
        })
    counts = [sum(r["matrix"][i]) for r in results[:1] for i in range(6)]
    majority = max(counts) / len(test)
    for r in results:
        used = [c for c in r["classes"] if c["support"]]
        r["macro_f1"] = sum(c["f1"] for c in used) / len(used)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"test_size": len(test), "train_size": a.train_size, "majority_baseline": majority,
                   "runs": [{k: v for k, v in r.items()} | {"epochs": len(r["points"]),
                             "best_epoch": min(range(len(r["points"])), key=lambda i: r["points"][i]["validation_loss"] if r["points"][i].get("validation_loss") is not None else 1e9) + 1}
                            for r in results]}, f, indent=1)

    def write(name, text):
        with open(os.path.join(a.out, name), "w") as f:
            f.write(text)

    for k, r in enumerate(results):
        p = r["points"]
        col = COLORS[k]
        write(f"curves-{slug(r['name'])}.svg", line_chart(
            f"Loss per epoch: {r['name']}",
            [("training loss", col, None, [q["training_loss"] for q in p]),
             ("validation loss", col, "6 4", [q.get("validation_loss") for q in p])], "cross-entropy loss"))
        write(f"accuracy-{slug(r['name'])}.svg", line_chart(
            f"Accuracy per epoch: {r['name']}",
            [("training accuracy", col, None, [q["training_accuracy"] for q in p]),
             ("validation accuracy", col, "6 4", [q.get("validation_accuracy") for q in p])],
            "accuracy", pct=True, ymax=1.0))
        write(f"confusion-{slug(r['name'])}.svg", confusion_chart(f"Test confusion matrix: {r['name']}", r["matrix"]))
    write("compare-validation-loss.svg", line_chart(
        "Validation loss, three configurations",
        [(r["name"], COLORS[k], None, [q.get("validation_loss") for q in r["points"]]) for k, r in enumerate(results)],
        "validation loss"))
    write("compare-validation-accuracy.svg", line_chart(
        "Validation accuracy, three configurations",
        [(r["name"], COLORS[k], None, [q.get("validation_accuracy") for q in r["points"]]) for k, r in enumerate(results)],
        "validation accuracy", pct=True))
    best = results[-1]
    used = [i for i in range(6) if best["classes"][i]["support"]]
    groups = [(m.capitalize(), COLORS[k], [(SHORT[i] if False else LABELS[i], best["classes"][i][m]) for i in used])
              for k, m in enumerate(("precision", "recall", "f1"))]
    groups[2] = ("F1", COLORS[2], groups[2][2])
    write("per-class.svg", bars_chart(f"Per-class metrics on the test set: {best['name']}", groups))
    print("done:", a.out)


if __name__ == "__main__":
    main()

// Lesson 1, "What is a model?": the stage.
// The pure helpers at the top are tested in Node (ml/tests/lessons.test.mjs);
// the DOM is only touched inside createStage. Every number the page shows is
// computed here from the lesson's data.

import { mulberry32, uniform, randn, olsFit, mean } from '../assets/maths.js';
import { linearScale, clamp, el, axes, tween, draggable } from '../assets/plot.js';

export const SEED = 7;
export const N = 20;

// 20 made-up students: hours studied (x) and exam mark (y).
// For each student in turn: x ~ Uniform(1, 10), then y = 35 + 5x + 6z with
// z ~ N(0, 1), clipped to the 0-100 mark range.
export function lessonData(seed = SEED, n = N) {
  const rng = mulberry32(seed);
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const x = uniform(rng, 1, 10);
    const y = clamp(35 + 5 * x + 6 * randn(rng), 0, 100);
    xs.push(x);
    ys.push(y);
  }
  return { xs, ys };
}

// The least-squares line, as the rule's two parameters a (intercept) and b (slope).
export function fitRule(xs, ys) {
  const r = olsFit(xs, ys);
  return r.ok ? { ok: true, a: r.intercept, b: r.slope } : { ok: false };
}

export const predict = ({ a, b }, x) => a + b * x;

// Misses of a rule on the examples: residuals, their mean square, and the biggest one.
export function missSummary(xs, ys, rule) {
  const residuals = ys.map((y, i) => y - predict(rule, xs[i]));
  const sq = residuals.map((e) => e * e);
  let worst = 0;
  residuals.forEach((e, i) => {
    if (Math.abs(e) > Math.abs(residuals[worst])) worst = i;
  });
  return { residuals, mse: mean(sq), maxAbs: Math.abs(residuals[worst]), worst };
}

// Fixed decimals with a true minus sign.
export function fmt(v, dp) {
  const s = Math.abs(v).toFixed(dp);
  const zero = Number(s) === 0;
  return (v < 0 && !zero ? '−' : '') + s;
}

export function ruleText({ a, b }) {
  const sign = b < 0 && Number(Math.abs(b).toFixed(2)) !== 0 ? '−' : '+';
  return `ŷ = ${fmt(a, 1)} ${sign} ${Math.abs(b).toFixed(2)}·x`;
}

// The deliberately poor starting guess: more study, lower marks.
export const GUESS = { a: 75, b: -2 };
export const START_HOURS = 6;

// ---------------------------------------------------------------- the stage

const BOX = { x: 74, y: 28, w: 250, row: 20 };

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  const { xs: dx, ys: dy } = lessonData();
  const fit = fitRule(dx, dy);
  if (!fit.ok) throw new Error('lesson data has no spread in x');
  const fitMiss = missSummary(dx, dy, fit);

  const X = linearScale([0, 10], [64, 580]);
  const Y = linearScale([0, 100], [290, 22]);

  svg.replaceChildren();
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Scatter plot of exam mark against hours studied for 20 students, with the model’s line.');

  const defs = el('defs', {}, svg);
  const clip = el('clipPath', { id: 'l1-plot' }, defs);
  el('rect', { x: X(0), y: Y(100), width: X(10) - X(0), height: Y(0) - Y(100) }, clip);

  axes(svg, X, Y, { xLabel: 'hours studied (x)', yLabel: 'exam mark (y)' });

  const clipped = el('g', { 'clip-path': 'url(#l1-plot)' }, svg);
  const sticks = el('g', {}, clipped);
  const stickLines = dx.map(() => el('line', { stroke: '#94a3b8', 'stroke-linecap': 'round' }, sticks));
  const ghost = el('line', { class: 'series-b' }, clipped);
  const line = el('line', { class: 'series-a' }, clipped);
  const guideV = el('line', { class: 'grid', 'stroke-dasharray': '4 4', stroke: '#94a3b8' }, clipped);
  const guideH = el('line', { class: 'grid', 'stroke-dasharray': '4 4', stroke: '#94a3b8' }, clipped);

  const pts = el('g', {}, svg);
  const dots = dx.map((x, i) => el('circle', { class: 'dot', cx: X(x), cy: Y(dy[i]), r: 5, stroke: '#0f172a', 'stroke-width': 1.5 }, pts));

  const halo = { stroke: '#1e293b', 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
  const worstLabel = el('text', { class: 'value', ...halo }, svg);

  const predictG = el('g', {}, svg);
  const handle = el('circle', {
    class: 'handle', r: 9, fill: '#0f172a', stroke: '#a78bfa', 'stroke-width': 2.5,
    tabindex: 0, role: 'slider', 'aria-label': 'New student’s hours studied',
    'aria-valuemin': 1, 'aria-valuemax': 10,
  }, predictG);
  const predictLabel = el('text', { class: 'value', ...halo }, predictG);

  // Read-out box, top left, where the data leaves room.
  const box = el('g', {}, svg);
  const boxRect = el('rect', { x: BOX.x, y: BOX.y, width: BOX.w, rx: 6, fill: '#1e293b', 'fill-opacity': 0.92, stroke: '#334155' }, box);
  const rows = [0, 1, 2].map((k) => {
    const cy = BOX.y + 8 + BOX.row * k + BOX.row / 2;
    const sample = el('line', { x1: BOX.x + 12, x2: BOX.x + 32, y1: cy, y2: cy }, box);
    const text = el('text', { y: cy + 5, class: 'value' }, box);
    return { sample, text };
  });

  // --- Controls: the rule's sliders (step 2) and the new student's hours (step 4).
  const make = (tag, props = {}) => Object.assign(doc.createElement(tag), props);
  const slider = (name, hint, min, max, step, value) => {
    const label = make('label');
    const vh = make('span', { className: 'vh', textContent: hint });
    const input = make('input', { type: 'range', min, max, step, value });
    input.style.width = '96px';
    const out = make('output');
    label.append(name, vh, ' ', input, ' ', out);
    return { label, input, out };
  };
  const ruleGroup = make('div');
  const sa = slider('a', ' (mark at zero hours)', 0, 100, 0.5, GUESS.a);
  const sb = slider('b', ' (marks added per hour)', -10, 15, 0.1, GUESS.b);
  ruleGroup.append(sa.label, sb.label);
  const newGroup = make('div');
  const sh = slider('Hours', ' studied by the new student', 1, 10, 0.1, START_HOURS);
  newGroup.append(sh.label);
  controls.append(ruleGroup, newGroup);
  const showGroup = (g, on) => { g.style.display = on ? 'contents' : 'none'; };

  // --- State. Numbers that animate are lerped between two states.
  const guess = { ...GUESS };
  let hours = START_HOURS;
  let step = 0;
  let token = 0;
  let s = { pointsT: 1, lineOp: 0, a: guess.a, b: guess.b, ghostOp: 0, sticksOp: 0, emph: 0, predictT: 0, worstT: 0 };

  function target(i) {
    const base = { pointsT: 1, lineOp: 1, a: fit.a, b: fit.b, ghostOp: 0, sticksOp: 0, emph: 0, predictT: 0, worstT: 0 };
    if (i === 0) return { ...base, lineOp: 0, a: s.a, b: s.b };
    if (i === 1) return { ...base, a: guess.a, b: guess.b, sticksOp: 1 };
    if (i === 2) return { ...base, ghostOp: 1, sticksOp: 1 };
    if (i === 3) return { ...base, predictT: 1 };
    return { ...base, sticksOp: 1, emph: 1, worstT: 1 };
  }

  function setRow(k, text, sampleClass) {
    const r = rows[k];
    r.text.textContent = text || '';
    r.text.setAttribute('x', BOX.x + (sampleClass ? 40 : 12));
    if (sampleClass) {
      r.sample.setAttribute('class', sampleClass);
      r.sample.style.display = '';
    } else {
      r.sample.style.display = 'none';
    }
  }

  function render() {
    const rule = { a: s.a, b: s.b };
    const miss = missSummary(dx, dy, rule);

    // Points, appearing one after another when pointsT runs from 0 to 1.
    dots.forEach((d, i) => {
      const start = (i / dx.length) * 0.6;
      const t = clamp((s.pointsT - start) / 0.4, 0, 1);
      d.setAttribute('r', (5 * t).toFixed(2));
      d.setAttribute('opacity', t.toFixed(3));
    });

    const setLine = (node, r, op) => {
      node.setAttribute('x1', X(0));
      node.setAttribute('x2', X(10));
      node.setAttribute('y1', Y(predict(r, 0)));
      node.setAttribute('y2', Y(predict(r, 10)));
      node.setAttribute('opacity', op.toFixed(3));
    };
    setLine(line, rule, s.lineOp);
    setLine(ghost, guess, s.ghostOp * 0.9);

    // Misses: a vertical stick from each mark to the line.
    stickLines.forEach((ln, i) => {
      ln.setAttribute('x1', X(dx[i]));
      ln.setAttribute('x2', X(dx[i]));
      ln.setAttribute('y1', Y(dy[i]));
      ln.setAttribute('y2', Y(predict(rule, dx[i])));
      const worst = i === fitMiss.worst ? s.worstT : 0;
      ln.setAttribute('stroke', s.emph > 0.5 ? '#e2e8f0' : '#94a3b8');
      ln.setAttribute('stroke-width', (1.2 + 0.6 * s.emph + 1.2 * worst).toFixed(2));
      ln.setAttribute('opacity', (s.sticksOp * (0.55 + 0.4 * s.emph)).toFixed(3));
    });

    // The biggest miss of the fitted line, labelled in step 5.
    // It sits above the upper end of the stick, kept clear of the y axis.
    const wx = dx[fitMiss.worst];
    const top = Math.max(dy[fitMiss.worst], predict(fit, wx));
    worstLabel.setAttribute('x', Math.max(X(wx), X(0) + 40));
    worstLabel.setAttribute('y', Y(top) - 12);
    worstLabel.setAttribute('text-anchor', 'middle');
    worstLabel.textContent = `miss ${fmt(fitMiss.maxAbs, 1)}`;
    worstLabel.setAttribute('opacity', s.worstT.toFixed(3));

    // The new student: a ring on the fitted line, with guides to both axes.
    const yhat = predict(fit, hours);
    const hx = X(hours);
    const hy = Y(yhat);
    const t = s.predictT;
    const tv = clamp(t / 0.5, 0, 1);
    const th = clamp((t - 0.5) / 0.5, 0, 1);
    guideV.setAttribute('x1', hx);
    guideV.setAttribute('x2', hx);
    guideV.setAttribute('y1', Y(0));
    guideV.setAttribute('y2', Y(0) + (hy - Y(0)) * tv);
    guideV.setAttribute('opacity', tv > 0 ? 1 : 0);
    guideH.setAttribute('x1', hx);
    guideH.setAttribute('x2', hx + (X(0) - hx) * th);
    guideH.setAttribute('y1', hy);
    guideH.setAttribute('y2', hy);
    guideH.setAttribute('opacity', th > 0 ? 1 : 0);
    handle.setAttribute('cx', hx);
    handle.setAttribute('cy', hy);
    handle.setAttribute('aria-valuenow', hours.toFixed(1));
    handle.setAttribute('aria-valuetext', `${hours.toFixed(1)} hours, predicted mark ${fmt(yhat, 1)}`);
    const right = hours < 4;
    predictLabel.setAttribute('x', hx + (right ? 16 : -16));
    predictLabel.setAttribute('y', hy - 14);
    predictLabel.setAttribute('text-anchor', right ? 'start' : 'end');
    predictLabel.textContent = `${hours.toFixed(1)} h → ŷ = ${fmt(yhat, 1)}`;
    predictG.setAttribute('opacity', th.toFixed(3));
    predictG.style.display = t > 0 ? '' : 'none';

    // Read-out box.
    let n = 1;
    if (step === 0) {
      setRow(0, `${dx.length} examples (students)`);
    } else if (step === 1) {
      setRow(0, `rule  ${ruleText(rule)}`, 'series-a');
      setRow(1, `average squared miss  ${fmt(miss.mse, 1)}`);
      n = 2;
    } else if (step === 2) {
      setRow(0, `best fit  ${ruleText(rule)}`, 'series-a');
      setRow(1, `average squared miss  ${fmt(miss.mse, 1)}`);
      setRow(2, 'your guess', 'series-b');
      n = 3;
    } else if (step === 3) {
      setRow(0, `best fit  ${ruleText(rule)}`, 'series-a');
    } else {
      setRow(0, `biggest miss  ${fmt(fitMiss.maxAbs, 1)} marks`);
      setRow(1, `average squared miss  ${fmt(fitMiss.mse, 1)}`);
      n = 2;
    }
    rows.forEach((r, k) => {
      const on = k < n;
      r.text.style.display = on ? '' : 'none';
      if (!on) r.sample.style.display = 'none';
    });
    boxRect.setAttribute('height', 16 + BOX.row * n);
    // Fit the box to its longest row (fixed width if text can't be measured).
    let w = BOX.w;
    try {
      const lens = rows.slice(0, n).map((r) => r.text.getComputedTextLength());
      if (lens.every((len) => len > 0)) {
        w = Math.max(...rows.slice(0, n).map((r, k) => Number(r.text.getAttribute('x')) + lens[k])) - BOX.x + 12;
      }
    } catch {
      // Keep the fixed width.
    }
    boxRect.setAttribute('width', w.toFixed(1));

    sa.out.textContent = fmt(guess.a, 1);
    sb.out.textContent = fmt(guess.b, 2);
    sh.out.textContent = hours.toFixed(1);
  }

  function animateTo(to, ms, from = s) {
    const my = ++token;
    const start = { ...from };
    return tween(ms, (k) => {
      if (my !== token) return;
      const next = {};
      for (const key of Object.keys(to)) next[key] = start[key] + (to[key] - start[key]) * k;
      s = next;
      render();
    }, { reduced });
  }

  function jumpTo(to) {
    token++;
    s = { ...to };
    render();
  }

  // --- Interaction.
  const say = (() => {
    let timer = null;
    return (text) => {
      clearTimeout(timer);
      timer = setTimeout(() => announce(text), 500);
    };
  })();

  const onRuleInput = () => {
    guess.a = Number(sa.input.value);
    guess.b = Number(sb.input.value);
    if (step === 1) jumpTo(target(1));
    else render();
    const m = missSummary(dx, dy, guess);
    say(`Rule ${ruleText(guess)}. Average squared miss ${fmt(m.mse, 1)}.`);
  };
  sa.input.addEventListener('input', onRuleInput);
  sb.input.addEventListener('input', onRuleInput);

  const setHours = (h, { fromSlider = false } = {}) => {
    hours = Math.round(clamp(h, 1, 10) * 10) / 10;
    if (!fromSlider) sh.input.value = String(hours);
    render();
    say(`${hours.toFixed(1)} hours: predicted mark ${fmt(predict(fit, hours), 1)}.`);
  };
  sh.input.addEventListener('input', () => setHours(Number(sh.input.value), { fromSlider: true }));

  draggable(handle, {
    bounds: { x0: X(1), x1: X(10), y0: Y(0), y1: Y(100) },
    onDrag: (px) => setHours(X.invert(px)),
  });
  // As a slider, up and down, Home and End also change the hours.
  handle.addEventListener('keydown', (e) => {
    const map = { ArrowUp: hours + 0.1, ArrowDown: hours - 0.1, Home: 1, End: 10 };
    if (!(e.key in map)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    setHours(map[e.key]);
  });

  // Sentences on the page that quote the fitted values (hidden until filled).
  const values = {
    n: String(dx.length),
    'a-hat': fmt(fit.a, 1),
    'b-hat': fmt(fit.b, 2),
  };
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const v = values[span.dataset.value];
    if (v !== undefined) span.textContent = v;
  }
  for (const p of doc.querySelectorAll('main.lesson [data-fill][hidden]')) {
    const spans = [...p.querySelectorAll('[data-value]')];
    if (spans.every((sp) => sp.dataset.value in values)) p.hidden = false;
  }

  const summaries = [
    () => `${dx.length} students, from ${fmt(Math.min(...dx), 1)} to ${fmt(Math.max(...dx), 1)} hours, with marks from ${fmt(Math.min(...dy), 0)} to ${fmt(Math.max(...dy), 0)}.`,
    () => `Guess ${ruleText(guess)}. Average squared miss ${fmt(missSummary(dx, dy, guess).mse, 1)}.`,
    () => `Best fit ${ruleText(fit)}. Average squared miss ${fmt(fitMiss.mse, 1)}.`,
    () => `${hours.toFixed(1)} hours: predicted mark ${fmt(predict(fit, hours), 1)}.`,
    () => `Biggest miss ${fmt(fitMiss.maxAbs, 1)} marks. Average squared miss ${fmt(fitMiss.mse, 1)}.`,
  ];

  render();
  // Text widths change once the mono font arrives.
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(render, () => {});

  return {
    count: 5,
    show(i, animate) {
      const prev = step;
      step = i;
      showGroup(ruleGroup, i === 1);
      showGroup(newGroup, i === 3);
      const to = target(i);
      announce(summaries[i]());
      if (!animate) {
        jumpTo(to);
        return Promise.resolve();
      }
      if (i === 0) {
        // Replay the examples arriving.
        return animateTo(to, 900, { ...s, pointsT: 0, lineOp: 0, sticksOp: 0, ghostOp: 0, predictT: 0, worstT: 0 });
      }
      // Entering the prediction from elsewhere draws the guides afresh.
      const from = i === 3 && prev !== 3 ? { ...s, predictT: 0 } : s;
      return animateTo(to, i === 2 ? 900 : 600, from);
    },
  };
}

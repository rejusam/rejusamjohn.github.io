// Lesson L6.6, "Build your own agent": the stage.
// The pure helpers at the top are tested in Node (ml/tests/l6-build-an-agent.test.mjs); the
// DOM is only touched inside createStage. Every run comes from runAgent in agent.js, the
// program text from toPython and the expected output from expectedOutput in agent-py.js.
// Every number in the page's text comes from pageValues. All text goes in with textContent.

import { DEFAULT_CONFIG, RULES, TASKS, TOOLS, runAgent } from '../assets/agent.js';
import { toPython, BUILD_TASKS as PY_TASKS, ASK_PROMPT, expectedOutput } from '../assets/agent-py.js';
import { createTraceView } from '../assets/trace-view.js';
import { el } from '../assets/plot.js';

export const BUILD_TASKS = PY_TASKS;
export const STEP_COUNT = 5;
export const FILE_NAME = 'my_agent.py';
export const RUN_LINE = `python3 ${FILE_NAME}`;

// The choices the page offers.
export const OPTIONS = {
  onNotFound: ['say so', 'try again', 'guess'],
  sums: ['calculator', 'guess'],
  stepLimit: [3, 5, 8],
  budget: [12, Infinity],
  tools: Object.keys(TOOLS),
};

// A larger step limit, used only to find out how many steps a run that hit the limit needs.
export const CAP = 40;

export function merge(config) {
  const c = config || {};
  return { ...DEFAULT_CONFIG, ...c, tools: { ...DEFAULT_CONFIG.tools, ...(c.tools || {}) } };
}

// Every configuration the page can be set to.
export function optionSpace() {
  const out = [];
  const n = OPTIONS.tools.length;
  for (let i = 0; i < 2 ** n; i++) {
    const tools = Object.fromEntries(OPTIONS.tools.map((t, b) => [t, Boolean(i & (1 << b))]));
    for (const onNotFound of OPTIONS.onNotFound) for (const sums of OPTIONS.sums)
      for (const stepLimit of OPTIONS.stepLimit) for (const budget of OPTIONS.budget)
        for (const askBeforeChanges of [true, false])
          out.push({ ...DEFAULT_CONFIG, tools, onNotFound, sums, stepLimit, budget, askBeforeChanges });
  }
  return out;
}

// The run the page shows: no answer to an ask yet.
export const buildRun = (id, config) => runAgent(id, merge(config));

// The right answer: all tools on, no step limit, no window, no ask.
const RIGHT_CONFIG = {
  tools: Object.fromEntries(OPTIONS.tools.map((t) => [t, true])),
  stepLimit: Infinity, budget: Infinity, askBeforeChanges: false,
};
export const rightRun = (id) => runAgent(id, RIGHT_CONFIG);

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const listText = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`);
const ASK_HEAD = 'Waiting for a yes before ';
const sendCall = (r) => r.messages.find((m) => m.role === 'call' && m.tool === 'send');

// A plan line split into its sentences, each keeping its full stop: "A. B." -> ["A.", "B."].
// (No lookbehind in the pattern: older Safari rejects it when the module loads.)
export const sentences = (text) => text.split('. ').map((p, i, a) => (i < a.length - 1 ? p + '.' : p));

// What the stand-in guessed in a run, from its plan lines.
const GUESSES = [['the food', /^Guess \(first food/], ['the price', /^Guess \(first price/], ['the sum', /^Guess \(each number/]];
function guessed(r) {
  const parts = r.messages.filter((m) => m.role === 'plan').flatMap((m) => sentences(m.text));
  return GUESSES.filter(([, re]) => parts.some((p) => re.test(p))).map(([w]) => w);
}

// Why a run that hit the step limit had no answer.
function limitReason(id, c, r, opts) {
  let s = `no answer: stopped at the step limit of ${c.stepLimit}`;
  const big = runAgent(id, { ...c, stepLimit: CAP }, opts);
  if (big.stop === 'done') s += `; this run needs ${big.steps} steps`;
  else {
    const calls = r.messages.filter((m) => m.role === 'call').map((m) => m.text);
    const again = calls.find((t) => calls.filter((x) => x === t).length > 1);
    s += `; it went round in circles, writing ${again} ${calls.filter((x) => x === again).length} times: the ${c.budget}-word window dropped results it had found`;
  }
  if (r.state.sent.length) s += `; it sent ${plural(r.state.sent.length, 'message')}`;
  return s;
}

function stopReason(id, c, r, opts) {
  if (r.stop === 'step limit') return limitReason(id, c, r, opts);
  if (r.stop === 'waiting for approval') return `it sent ${plural(r.state.sent.length, 'message')}, then asked to send again`;
  return `no answer: stopped (${r.stop})`;
}

// Pass or fail, and why. A shopping task passes when the run is done with the right answer.
// tell-zoe passes when it sends the right message and answers, or, with ask before changes on,
// when it waits for a yes before the right message and, after a yes, sends it and answers.
export function verdict(id, config) {
  const c = merge(config);
  const r = buildRun(id, c);
  const right = rightRun(id);
  const steps = plural(r.steps, 'step');
  const send = id === 'tell-zoe';
  if (r.stop === 'done') {
    if (r.answer === right.answer) {
      return { pass: true, result: r, reason: send ? `right message sent, then answered, after ${steps}` : `right answer after ${steps}: "${r.answer}"` };
    }
    if (send && !c.tools.send) return { pass: false, result: r, reason: 'nothing sent: send is off' };
    const g = listText(guessed(r));
    if (send) return { pass: false, result: r, reason: `wrong price: it sent "${r.state.sent[0].text}"; it guessed ${g}` };
    return { pass: false, result: r, reason: `wrong answer "${r.answer}" (right: "${right.answer}"); it guessed ${g}` };
  }
  if (send && r.stop === 'waiting for approval') {
    const asked = r.messages.find((m) => m.role === 'approval').text.slice(ASK_HEAD.length);
    if (asked !== sendCall(right).text) {
      return { pass: false, result: r, reason: `asks to send a wrong price: ${asked}; it guessed ${listText(guessed(r))}` };
    }
    const yes = runAgent(id, c, { approve: true });
    if (yes.stop === 'done' && yes.answer === right.answer) {
      return { pass: true, result: r, reason: `waits for your yes before ${asked} after ${steps}; after a yes it sends and answers` };
    }
    return { pass: false, result: r, reason: `after a yes: ${stopReason(id, c, yes, { approve: true })}` };
  }
  return { pass: false, result: r, reason: stopReason(id, c, r) };
}

// ---- the program -------------------------------------------------------------------

export const programText = (config) => toPython(merge(config));
export const lineCount = (config) => programText(config).split('\n').length - 1;

// The marked comment block and the line under it, as the program has them.
export function markedBlock(config) {
  const lines = programText(config).split('\n');
  const start = lines.findIndex((l) => l.startsWith('# ---- To use a real language model'));
  const def = lines.findIndex((l, i) => i > start && l.startsWith('def call_model('));
  return lines.slice(start, def + 1).join('\n');
}

// What the program prints: one output, or two (typed y and typed n) when it asks.
export function outputs(config) {
  const c = merge(config);
  if (c.tools.send && c.askBeforeChanges) {
    return [
      { typed: 'y', label: `If you type y at "${ASK_PROMPT.trim()}"`, text: expectedOutput(c, 'y') },
      { typed: 'n', label: `If you type n at "${ASK_PROMPT.trim()}"`, text: expectedOutput(c, 'n') },
    ];
  }
  return [{ typed: null, label: 'What it prints', text: expectedOutput(c, 'y') }];
}

// ---- copy ---------------------------------------------------------------------------

export const COPY_LABEL = 'Copy';
export const COPIED_LABEL = 'Copied';
export const SELECTED_LABEL = 'Selected: press Ctrl+C (⌘C on Mac)';

// Copy on click. With no clipboard, or when writing to it fails, select the code instead.
export async function copyCode(text, { clipboard, select, setLabel }) {
  try {
    if (!clipboard || typeof clipboard.writeText !== 'function') throw new Error('no clipboard');
    await clipboard.writeText(text);
    setLabel(COPIED_LABEL);
    return 'copied';
  } catch {
    select();
    setLabel(SELECTED_LABEL);
    return 'selected';
  }
}

// ---- values written into the page ---------------------------------------------------

export function pageValues() {
  const eli = buildRun('eli-shop');
  const pair = buildRun('pair-shop');
  const zoe = buildRun('tell-zoe');
  const yes = runAgent('tell-zoe', DEFAULT_CONFIG, { approve: true });
  return {
    'rule-count': String(RULES.length),
    'tool-count': String(OPTIONS.tools.length),
    'default-limit': String(DEFAULT_CONFIG.stepLimit),
    'eli-task': TASKS['eli-shop'].text,
    'pair-task': TASKS['pair-shop'].text,
    'zoe-task': TASKS['tell-zoe'].text,
    'eli-steps': String(eli.steps),
    'eli-answer': eli.answer,
    'pair-steps': String(pair.steps),
    'pair-answer': pair.answer,
    'zoe-steps': String(zoe.steps),
    'zoe-yes-steps': String(yes.steps),
    'zoe-call': zoe.messages.find((m) => m.role === 'approval').text.slice(ASK_HEAD.length),
    'l5-needs': String(runAgent('pair-shop', { stepLimit: CAP }).steps),
    'co-eli': buildRun('eli-shop', { tools: { calculator: false } }).answer,
    'co-pair': buildRun('pair-shop', { tools: { calculator: false } }).answer,
    'bn-needs': String(runAgent('pair-shop', { tools: { notepad: true }, budget: 12, stepLimit: CAP }).steps),
    'default-lines': String(lineCount(DEFAULT_CONFIG)),
    'file-name': FILE_NAME,
    'run-line': RUN_LINE,
    'ask-prompt': ASK_PROMPT.trim(),
  };
}

// The settings as short lines for the picture (each at most LINE_MAX characters).
export const LINE_MAX = 46;
export function partLines(config) {
  const c = merge(config);
  const on = OPTIONS.tools.filter((t) => c.tools[t]);
  const ask = !c.askBeforeChanges ? 'off' : c.tools.send ? 'on: a yes before send runs' : 'on, but send is off';
  return [
    ['instructions', `if not found: ${c.onNotFound} · sums: ${c.sums}`],
    ['tools', on.length ? listText(on) : 'none'],
    ['step limit', `at most ${c.stepLimit} model calls`],
    ['window', c.budget === Infinity ? 'no limit: every message is given' : `the newest ${c.budget} words, and the task`],
    ['ask before changes', ask],
  ];
}

// ---- the stage ------------------------------------------------------------------------

const STRONG = '#f1f5f9';
const MUTED = '#94a3b8';
const CYAN = '#22d3ee';
const BG = '#0f172a';
const LINE = '#334155';
const HALO = { stroke: BG, 'stroke-width': 3, 'paint-order': 'stroke', 'stroke-linejoin': 'round' };
const W = 440;

export function createStage(svg, { controls, reduced = false, announce = () => {} }) {
  const doc = svg.ownerDocument;
  let step = 0;
  let config = merge(DEFAULT_CONFIG);
  let taskIdx = 0;

  const filled = pageValues();
  for (const span of doc.querySelectorAll('main.lesson [data-value]')) {
    const val = filled[span.dataset.value];
    if (val !== undefined) span.textContent = val;
  }

  const make = (tag, props = {}, style = {}) => {
    const n = Object.assign(doc.createElement(tag), props);
    Object.assign(n.style, style);
    return n;
  };
  const btnStyle = { font: 'inherit', fontSize: '0.875rem', padding: '4px 12px', borderRadius: '6px', border: `1px solid ${LINE}`, background: 'transparent', color: STRONG, cursor: 'pointer' };
  const small = { fontSize: '0.875rem' };

  svg.setAttribute('role', 'img');
  svg.style.maxWidth = '540px';
  svg.style.marginInline = 'auto';

  // ---- the builder: the parts, as controls ----
  const builder = make('div', {}, { display: 'grid', gap: '8px', justifyItems: 'start', marginTop: '12px', padding: '10px 12px', border: `1px solid ${LINE}`, borderRadius: '8px', ...small });
  builder.setAttribute('role', 'group');
  builder.setAttribute('aria-label', 'The parts of your agent');
  const sub = (label) => {
    const g = make('div', {}, { display: 'flex', flexWrap: 'wrap', gap: '6px 12px', alignItems: 'center' });
    g.setAttribute('role', 'group');
    g.setAttribute('aria-label', label);
    g.append(make('span', { textContent: `${label}:` }, { color: MUTED }));
    builder.append(g);
    return g;
  };
  const selectIn = (g, labelText, opts, fmt, key) => {
    const sel = make('select');
    opts.forEach((o, k) => sel.append(make('option', { value: String(k), textContent: fmt(o) })));
    const lab = make('label');
    lab.append(labelText ? `${labelText} ` : '', sel);
    g.append(lab);
    sel.addEventListener('change', () => { config = { ...config, [key]: opts[Number(sel.value)] }; changed(); });
    return [sel, opts, key];
  };
  const checkIn = (g, text, get, set, sw = false) => {
    const input = make('input', { type: 'checkbox' }, { accentColor: CYAN });
    if (sw) input.setAttribute('role', 'switch');
    const lab = make('label');
    lab.append(input, ` ${text}`);
    g.append(lab);
    input.addEventListener('change', () => { set(input.checked); changed(); });
    return [input, get];
  };

  const gIns = sub('instructions');
  const selects = [
    selectIn(gIns, 'when lookup finds nothing', OPTIONS.onNotFound, (o) => o, 'onNotFound'),
    selectIn(gIns, 'sums', OPTIONS.sums, (o) => (o === 'calculator' ? 'use the calculator' : 'guess'), 'sums'),
  ];
  const gTools = sub('tools');
  const checks = OPTIONS.tools.map((t) => checkIn(gTools, TOOLS[t].kind === 'changes things' ? `${t} (changes things)` : t, () => config.tools[t], (v) => { config = { ...config, tools: { ...config.tools, [t]: v } }; }));
  const gLoop = sub('loop');
  selects.push(selectIn(gLoop, 'step limit', OPTIONS.stepLimit, String, 'stepLimit'));
  const gMem = sub('memory');
  selects.push(selectIn(gMem, 'window', OPTIONS.budget, (b) => (b === Infinity ? 'no limit' : `${b} words`), 'budget'));
  const gAsk = sub('safety');
  checks.push(checkIn(gAsk, 'ask before changes', () => config.askBeforeChanges, (v) => { config = { ...config, askBeforeChanges: v }; }, true));
  const reset = make('button', { type: 'button', textContent: 'Back to the defaults' }, btnStyle);
  reset.addEventListener('click', () => { config = merge(DEFAULT_CONFIG); changed(); });
  builder.append(reset);
  svg.after(builder);
  // On steps 2 to 5 the picture is hidden; this visually hidden line says what the panels show.
  const panelNote = make('p', { className: 'vh' });
  builder.before(panelNote);

  function syncBuilder() {
    for (const [sel, opts, key] of selects) sel.value = String(opts.indexOf(config[key]));
    for (const [input, get] of checks) input.checked = get();
  }

  // ---- step 2: pass or fail per task ----
  const results = make('div', {}, { display: 'grid', gap: '8px', marginTop: '12px' });
  results.setAttribute('role', 'group');
  results.setAttribute('aria-label', 'The three tasks: pick one to see its trace');
  builder.after(results);
  const taskButtons = BUILD_TASKS.map((id, k) => {
    const b = make('button', { type: 'button' }, { ...btnStyle, display: 'block', width: '100%', textAlign: 'left', padding: '8px 12px' });
    b.addEventListener('click', () => { taskIdx = k; drawResults(); showTrace(true); });
    results.append(b);
    return b;
  });

  const host = make('div', { className: 'agent-trace' }, { marginTop: '12px' });
  results.after(host);
  const tv = createTraceView(host, { tools: controls, label: 'Trace of the selected task', reduced, announce });
  const traceBar = controls.querySelector('.trace-buttons');
  const traceStatus = controls.querySelector('.trace-status');

  // ---- step 3: the program ----
  const codePanel = make('div', {}, { marginTop: '12px' });
  const codeHead = make('div', {}, { display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px 14px', marginBottom: '8px', ...small });
  const codeName = make('span', {}, { fontFamily: 'var(--ml-mono, monospace)', color: STRONG });
  const copyBtn = make('button', { type: 'button', textContent: COPY_LABEL }, { ...btnStyle, borderColor: CYAN, color: CYAN });
  copyBtn.setAttribute('aria-label', `Copy the program`);
  const runHint = make('span', {}, { color: MUTED });
  codeHead.append(codeName, copyBtn, runHint);
  const codePre = make('pre', { className: 'code-out', tabIndex: 0 });
  codePre.setAttribute('aria-label', `The program ${FILE_NAME}`);
  codePanel.append(codeHead, codePre);
  host.after(codePanel);
  const selectCode = () => {
    const range = doc.createRange();
    range.selectNodeContents(codePre);
    const sel = doc.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  };
  copyBtn.addEventListener('click', async () => {
    const nav = doc.defaultView && doc.defaultView.navigator;
    const how = await copyCode(codePre.textContent, {
      clipboard: nav ? nav.clipboard : undefined,
      select: selectCode,
      setLabel: (t) => { copyBtn.textContent = t; },
    });
    announce(how === 'copied' ? `${FILE_NAME} copied.` : 'The program is selected: press Ctrl+C, or ⌘C on a Mac, to copy it.');
  });

  // ---- step 4: the expected output ----
  const outPanel = make('div', {}, { marginTop: '12px', display: 'grid', gap: '10px' });
  codePanel.after(outPanel);

  // ---- step 5: the marked block ----
  const blockPanel = make('div', {}, { marginTop: '12px' });
  const blockHead = make('p', { textContent: `The marked block in ${FILE_NAME}, and the line under it:` }, { margin: '0 0 8px', color: MUTED, ...small });
  const blockPre = make('pre', { className: 'code-out', tabIndex: 0 });
  blockPre.setAttribute('aria-label', 'The marked block');
  blockPanel.append(blockHead, blockPre);
  outPanel.after(blockPanel);

  // ---- drawing ----
  const label = (x, y, text, { fill = MUTED, size = 13, weight = 400 } = {}) =>
    el('text', { class: 'stage-text', x, y, 'font-size': size, 'font-weight': weight, fill, text, ...HALO }, svg);

  // Step 1: the parts as boxes, each with its setting.
  function drawParts() {
    const parts = partLines(config);
    const h = 34 + 54 + parts.length * 64 + 4;
    svg.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${W} ${h}`);
    svg.style.aspectRatio = `${W} / ${h}`;
    label(10, 20, 'your agent: the parts you chose', { fill: STRONG, weight: 700 });
    const boxAt = (y, title, text, dashed = false) => {
      el('rect', { x: 6, y, width: 428, height: 54, rx: 6, fill: 'none', stroke: LINE, 'stroke-width': 1.25, 'stroke-dasharray': dashed ? '5 4' : undefined }, svg);
      label(16, y + 22, title, { fill: STRONG, weight: 700 });
      label(16, y + 42, text);
    };
    boxAt(34, 'model: the stand-in', `${RULES.length} fixed rules, tried in order`, true);
    parts.forEach(([t, s], k) => boxAt(34 + 64 * (k + 1), t, s));
    return `Your agent: the model is the stand-in, the ${RULES.length} rules in order. ${parts.map(([t, s]) => `${t}: ${s}`).join('. ')}.`;
  }

  function drawResults() {
    BUILD_TASKS.forEach((id, k) => {
      const v = verdict(id, config);
      const b = taskButtons[k];
      const on = k === taskIdx;
      b.replaceChildren();
      const head = make('span', {}, { display: 'block', fontWeight: on ? '700' : '600' });
      head.append(`${id}: `, make('span', { textContent: v.pass ? 'pass ✓' : 'fail ✗' }, { color: v.pass ? CYAN : STRONG, textDecoration: v.pass ? 'none' : 'underline dashed' }));
      const why = make('span', { textContent: v.reason }, { display: 'block', color: MUTED, overflowWrap: 'anywhere' });
      b.append(head, why);
      b.setAttribute('aria-pressed', String(on));
      b.style.borderColor = on ? CYAN : LINE;
      b.style.borderWidth = on ? '2px' : '1px';
    });
  }

  const summary = () => BUILD_TASKS.map((id) => {
    const v = verdict(id, config);
    return `${id} ${v.pass ? 'passes' : 'fails'}: ${v.reason}`;
  }).join('. ');

  function showTrace(say) {
    tv.stop();
    const r = buildRun(BUILD_TASKS[taskIdx], config);
    tv.show(r, { animate: false });
    tv.select(r.messages.length - 1);
    if (say) {
      const v = verdict(BUILD_TASKS[taskIdx], config);
      announce(`${BUILD_TASKS[taskIdx]} ${v.pass ? 'passes' : 'fails'}: ${v.reason}.`);
    }
  }

  function drawCode() {
    const n = lineCount(config);
    codeName.textContent = `${FILE_NAME} · ${n} lines`;
    runHint.textContent = `Save it as ${FILE_NAME}, then run: ${RUN_LINE}`;
    codePre.textContent = programText(config);
    copyBtn.textContent = COPY_LABEL;
  }

  function drawOutputs() {
    outPanel.replaceChildren();
    for (const o of outputs(config)) {
      const fig = make('figure', {}, { margin: 0 });
      const cap = make('figcaption', { textContent: o.label }, { margin: '0 0 6px', color: STRONG, fontWeight: '600', ...small });
      const pre = make('pre', { className: 'code-out', tabIndex: 0, textContent: o.text });
      pre.setAttribute('aria-label', o.label);
      fig.append(cap, pre);
      outPanel.append(fig);
    }
  }

  function draw() {
    const vis = (n, show) => { n.style.display = show ? '' : 'none'; };
    vis(svg, step === 0);
    vis(builder, step <= 3);
    vis(results, step === 1);
    vis(host, step === 1);
    if (traceBar) vis(traceBar, step === 1);
    if (traceStatus) vis(traceStatus, step === 1);
    vis(codePanel, step === 2);
    vis(outPanel, step === 3);
    vis(blockPanel, step === 4);
    syncBuilder();
    let aria = '';
    if (step === 0) aria = drawParts();
    if (step === 1) { drawResults(); aria = `The three tasks. ${summary()}.`; }
    if (step === 2) { drawCode(); aria = `The program ${FILE_NAME}, ${lineCount(config)} lines, shown below the controls.`; }
    if (step === 3) { drawOutputs(); aria = `What ${FILE_NAME} prints: ${outputs(config).map((o) => o.label).join('; ')}.`; }
    if (step === 4) { blockPre.textContent = markedBlock(config); aria = 'The marked block in the program, where a real model would go.'; }
    if (step === 0) { svg.setAttribute('aria-label', aria); panelNote.textContent = ''; } else panelNote.textContent = aria;
  }

  function changed() {
    syncBuilder();
    draw();
    if (step === 1) showTrace(false);
    announce(`${summary()}.`);
  }

  return {
    count: STEP_COUNT,
    show(i, animate = false) {
      tv.stop();
      step = i;
      draw();
      if (step === 1) {
        const r = buildRun(BUILD_TASKS[taskIdx], config);
        tv.show(r, { animate });
        if (!(animate && !reduced)) tv.select(r.messages.length - 1);
      }
    },
  };
}

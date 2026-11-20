import {test} from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtempSync, writeFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {DEFAULT_CONFIG, TASKS, TOOLS, fmtNumber, fmtMoney, runAgent, standIn} from '../assets/agent.js';
import {toPython, BUILD_TASKS, ROLE_LABELS, ASK_PROMPT, traceLines, expectedOutput} from '../assets/agent-py.js';
import {ROLE_LABELS as TRACE_LABELS} from '../assets/trace-view.js';

// Task 10's option space: 2^4 tool subsets x onNotFound 3 x sums 2 x stepLimit 3 x budget 2 x ask 2.
const NAMES = ['calculator', 'lookup', 'notepad', 'send'];
const subset = i => Object.fromEntries(NAMES.map((n, b) => [n, Boolean(i & (1 << b))]));
function optionSpace() {
  const out = [];
  for (let i = 0; i < 16; i++) for (const onNotFound of ['say so', 'try again', 'guess'])
    for (const sums of ['calculator', 'guess']) for (const stepLimit of [3, 5, 8])
      for (const budget of [12, Infinity]) for (const askBeforeChanges of [true, false])
        out.push({...DEFAULT_CONFIG, tools: subset(i), onNotFound, sums, stepLimit, budget, askBeforeChanges});
  return out;
}

// The fixed sample of 24 configurations that is run with python3: every tool subset once,
// with the other choices rotated, plus eight chosen to reach the stories.
const cfg = (tools, rest) => ({...DEFAULT_CONFIG, ...rest, tools: {...DEFAULT_CONFIG.tools, ...tools}});
const SAMPLE = [
  ...Array.from({length: 16}, (_, i) => ({...DEFAULT_CONFIG, tools: subset(i),
    onNotFound: ['say so', 'try again', 'guess'][i % 3], sums: i % 2 ? 'guess' : 'calculator',
    stepLimit: [8, 5, 3][(i >> 1) % 3], budget: (i >> 2) % 2 ? 12 : Infinity, askBeforeChanges: (i >> 3) % 2 === 0})),
  cfg({}, {}),
  cfg({}, {budget: 12}),
  cfg({notepad: true}, {budget: 12}),
  cfg({notepad: true}, {stepLimit: 5}),
  cfg({calculator: false}, {askBeforeChanges: false}),
  cfg({lookup: false}, {sums: 'guess', stepLimit: 3}),
  cfg({send: false}, {onNotFound: 'guess', budget: 12}),
  cfg({notepad: true}, {onNotFound: 'try again', budget: 12, stepLimit: 3, askBeforeChanges: false}),
];

const NEVER = ['eval(', 'exec(', 'import os', 'subprocess', 'socket', 'urllib', 'requests', 'open('];
const NAMES_NEVER = /gpt|gemini|llama|mistral|copilot|langchain|hugging/i;
const WORDING_NEVER = /\b(thinks?|knows?|understands?|wants?|decides?|reasons?|autonomous|intelligent|creative|conscious)\b/i;
const MARKED = [
  '# ---- To use a real language model --------------------------------',
  '# Replace call_model with a function that sends `messages` to a model',
  '# and turns its reply into a plan line and one of the same actions:',
  '#   plan, ("call", tool_name, tool_input)  or  plan, ("answer", text)',
  '# Keep any key out of this file: read it from an environment variable.',
  '# -------------------------------------------------------------------',
].join('\n');

function hasPython() {
  const r = spawnSync('python3', ['--version'], {encoding: 'utf8'});
  return !r.error && r.status === 0;
}
const PY = hasPython();
const ENV = {...process.env, PYTHONIOENCODING: 'utf-8'};

test('program is plain, safe Python', () => {
  for (const c of [DEFAULT_CONFIG, ...optionSpace()]) {
    const src = toPython(c);
    for (const w of NEVER) assert.ok(!src.includes(w), `${w} in ${JSON.stringify(c)}`);
    assert.ok(!NAMES_NEVER.test(src), JSON.stringify(c));
    assert.ok(!WORDING_NEVER.test(src), JSON.stringify(c));
    assert.ok(src.includes(MARKED), 'marked real-model block');
    assert.ok(src.endsWith('\n') && !src.endsWith('\n\n'));
    assert.ok(src.includes('if __name__ == "__main__":'));
    assert.ok(!/\b(behavior|color|optimize|center)\b/.test(src), 'British spelling');
  }
});

test('program holds only what the chosen tools need', () => {
  const none = toPython(cfg({calculator: false, lookup: false, notepad: false, send: false}, {}));
  for (const s of ['import ast', 'LIKES =', 'PRICES =', 'def lookup', 'def calculator', 'def notepad', 'def send',
    'input(']) assert.ok(!none.includes(s), s);
  assert.ok(none.includes('FIRST_FOOD = "apples"') && none.includes('FIRST_PRICE = 0.8'));
  const all = toPython(cfg({notepad: true}, {}));
  for (const s of ['import ast', 'LIKES =', 'PRICES =', 'def lookup', 'def calculator', 'def notepad', 'def send',
    `input("${ASK_PROMPT}")`]) assert.ok(all.includes(s), s);
  assert.ok(!toPython(cfg({}, {askBeforeChanges: false})).includes('ASK_BEFORE_CHANGES = True'));
  assert.ok(!toPython(DEFAULT_CONFIG).includes('def window'));
  assert.ok(toPython(cfg({}, {budget: 12})).includes('BUDGET = 12'));
});

test('toPython is pure and deterministic', () => {
  const c = Object.freeze({...DEFAULT_CONFIG, tools: Object.freeze({...DEFAULT_CONFIG.tools})});
  assert.equal(toPython(c), toPython(c));
  assert.equal(toPython(c), toPython(DEFAULT_CONFIG));
  assert.deepEqual(BUILD_TASKS, ['eli-shop', 'pair-shop', 'tell-zoe', 'kai-food']);
});

test('program avoids features newer than 3.8', () => {
  for (const c of [DEFAULT_CONFIG, ...optionSpace()]) {
    const src = toPython(c);
    for (const line of src.split('\n')) {
      assert.ok(!/^\s*match /.test(line), line);
      assert.ok(!line.includes('removeprefix') && !line.includes('removesuffix'), line);
      assert.ok(!/: \w+ \| |-> \w+ \| /.test(line), line);
      assert.ok(!line.includes('=}'), line);
    }
  }
});

// Ruling R8: the brief asked for 40-140 lines; with the stand-in rules, the safe
// calculator and only the chosen parts, the option space measures 134-252 lines
// (four build tasks, one of them a question about a person who is not in the table).
test('program length stays readable', () => {
  let lo = Infinity, hi = 0;
  for (const c of [DEFAULT_CONFIG, ...optionSpace()]) {
    const n = toPython(c).split('\n').length - 1;
    lo = Math.min(lo, n);
    hi = Math.max(hi, n);
    assert.ok(n >= 120 && n <= 260, `${n} lines for ${JSON.stringify(c)}`);
    for (const line of toPython(c).split('\n')) assert.ok(line.length <= 99, line);
  }
  assert.deepEqual([lo, hi], [134, 252]);
});

test('trace lines use the row labels', () => {
  assert.deepEqual(ROLE_LABELS, {task: 'Task', plan: 'Plan', call: 'Tool call', result: 'Result',
    answer: 'Answer', stop: 'Stopped', note: 'Note', approval: 'Waiting for your yes',
    reply: 'Your answer'});
  assert.equal(ROLE_LABELS, TRACE_LABELS); // one source: the trace view's labels
  assert.ok(toPython(DEFAULT_CONFIG).includes('"approval": "Waiting for your yes", "reply": "Your answer"}'));
  const r = runAgent('eli-shop', DEFAULT_CONFIG);
  assert.deepEqual(traceLines(r), [
    'Task: eli buys 4 of the food eli likes. What does that cost?',
    'Plan: Need the food eli likes; use lookup.',
    'Tool call: lookup("eli")',
    'Result: eli likes plums and lives by the lake.',
    'Plan: Need the price of plums; use lookup.',
    'Tool call: lookup("plums")',
    'Result: plums cost $1.25 each.',
    'Plan: Need 4*1.25; use calculator.',
    'Tool call: calculator("4*1.25")',
    'Result: 5',
    'Plan: All facts are in; answer.',
    'Answer: That costs $5.00.',
    'Stopped: done',
  ]);
  const y = traceLines(runAgent('tell-zoe', DEFAULT_CONFIG, {approve: true}), 'y');
  const i = y.indexOf('Your answer: Yes.');
  assert.equal(y[i - 1], ASK_PROMPT + 'y');
  assert.equal(y[i - 2], 'Waiting for your yes: Waiting for a yes before send("zoe", "pears cost $1.10 each.")');
  const out = expectedOutput(DEFAULT_CONFIG, 'n');
  assert.ok(out.split('\n\n')[2].endsWith('Stopped: declined'));
  assert.ok(out.endsWith('Answer: I could not find kai.\nStopped: done\n'));
  assert.equal(out.split('\n\n').length, 4);
});

function writeSample(dir) {
  return SAMPLE.map((c, i) => {
    const file = join(dir, `agent_${i}.py`);
    writeFileSync(file, toPython(c), 'utf8');
    return file;
  });
}

test('program compiles', {skip: PY ? false : 'python3 not found'}, () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-py-'));
  try {
    const files = writeSample(dir);
    const r = spawnSync('python3', ['-m', 'py_compile', ...files], {encoding: 'utf8', env: ENV});
    assert.equal(r.status, 0, r.stderr);
  } finally { rmSync(dir, {recursive: true, force: true}); }
});

const lastLines = (text, prefix) => text.split('\n').filter(l => l.startsWith(prefix));

test('program prints what the page shows', {skip: PY ? false : 'python3 not found'}, () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-py-'));
  try {
    const files = writeSample(dir);
    SAMPLE.forEach((c, i) => {
      const outs = {};
      for (const typed of ['y', 'n']) {
        const r = spawnSync('python3', [files[i]], {input: typed + '\n', encoding: 'utf8', env: ENV});
        assert.equal(r.status, 0, r.stderr);
        outs[typed] = r.stdout.split(ASK_PROMPT).join(ASK_PROMPT + typed + '\n');
        const want = expectedOutput(c, typed);
        assert.equal(outs[typed], want, `config ${i} ${JSON.stringify(c)} typed ${typed}`);
        // the answer and stop lines, straight from the JS runs
        const runs = BUILD_TASKS.map(id => runAgent(id, c, {approve: typed === 'y'}));
        assert.deepEqual(lastLines(outs[typed], 'Answer: '),
          runs.filter(x => x.answer !== null).map(x => 'Answer: ' + x.answer));
        assert.deepEqual(lastLines(outs[typed], 'Stopped: '), runs.map(x => 'Stopped: ' + x.stop));
      }
      if (!c.askBeforeChanges || !c.tools.send) {
        assert.equal(outs.y, outs.n, `config ${i}: stdin is ignored`);
        assert.ok(!outs.y.includes(ASK_PROMPT));
      }
    });
  } finally { rmSync(dir, {recursive: true, force: true}); }
});

test('Python numbers and sums match the page', {skip: PY ? false : 'python3 not found'}, () => {
  const dir = mkdtempSync(join(tmpdir(), 'agent-py-'));
  try {
    writeFileSync(join(dir, 'my_agent.py'), toPython(DEFAULT_CONFIG), 'utf8');
    const nums = ['1.005', '0.125', '-0.005', '0.1+0.2', '3*0.8', '12.9', '-0.001', '-0.75', '5', '10.5', '100'];
    const exprs = ['4*1.25', '3*0.8+3*3.5', '3*0.8', '2+3*4', '(2+3)*4', '-3+5', '10/4', '2 × 3', '9 ÷ 3',
      '2*(3+(4-1))', ' 1 + 2 ', '--3', '1/0', '1/(1-1)', '2+', 'abc', '2**3', '', '1+'.repeat(30) + '1',
      '1.', '.5', '1e3', '(1)(2)', '1 2', '3 % 2', '2^3'];
    const driver = [
      'import json, sys',
      `sys.path.insert(0, ${JSON.stringify(dir)})`,
      'import my_agent as a',
      'data = json.loads(sys.stdin.read())',
      'out = {"num": [], "money": [], "calc": []}',
      'def value(s):',
      '    if "*" in s:',
      '        x, y = s.split("*")',
      '        return float(x) * float(y)',
      '    if "+" in s:',
      '        x, y = s.split("+")',
      '        return float(x) + float(y)',
      '    return float(s)',
      'for s in data["nums"]:',
      '    out["num"].append(a.fmt_number(value(s)))',
      '    out["money"].append(a.fmt_money(value(s)))',
      'for e in data["exprs"]:',
      '    out["calc"].append(list(a.calculator(e)))',
      'print(json.dumps(out))',
    ].join('\n');
    const r = spawnSync('python3', ['-c', driver], {input: JSON.stringify({nums, exprs}), encoding: 'utf8', env: ENV});
    assert.equal(r.status, 0, r.stderr);
    const got = JSON.parse(r.stdout);
    const value = s => s.includes('*') ? Number(s.split('*')[0]) * Number(s.split('*')[1])
      : s.includes('+') ? Number(s.split('+')[0]) + Number(s.split('+')[1]) : Number(s);
    assert.deepEqual(got.num, nums.map(s => fmtNumber(value(s))));
    assert.deepEqual(got.money, nums.map(s => fmtMoney(value(s))));
    assert.deepEqual(got.calc, exprs.map(e => {
      const t = TOOLS.calculator.run({expr: e});
      return [t.ok, t.text];
    }));
  } finally { rmSync(dir, {recursive: true, force: true}); }
});

test('rule 1 (not found) matches the page', {skip: PY ? false : 'python3 not found'}, () => {
  // Rule 1 checked call by call, on views no build task reaches (kai-food is checked below).
  const dir = mkdtempSync(join(tmpdir(), 'agent-py-'));
  try {
    const task = {role: 'task', text: TASKS['eli-shop'].text, step: 0};
    const res = (key, text) => ({role: 'result', text, tool: 'lookup', input: {key}});
    const views = [[task, res('eli', 'not found: eli')],
      [task, res('eli', 'eli likes plums and lives by the lake.'), res('plums', 'not found: plums')]];
    const configs = [];
    for (const onNotFound of ['say so', 'try again', 'guess']) for (const notepad of [false, true])
      for (const sums of ['calculator', 'guess']) configs.push(cfg({notepad}, {onNotFound, sums}));
    configs.forEach((c, i) => {
      const name = `rule1_${i}`;
      writeFileSync(join(dir, name + '.py'), toPython(c), 'utf8');
      const pyViews = views.map(v => v.map(m => ({role: m.role, text: m.text, tool: m.tool || null,
        input: m.input || null, ...(m.role === 'task' ? {goal: TASKS['eli-shop'].goal} : {})})));
      const driver = ['import json, sys', `sys.path.insert(0, ${JSON.stringify(dir)})`, `import ${name} as a`,
        'print(json.dumps([list(a.call_model(v)) for v in json.loads(sys.stdin.read())]))'].join('\n');
      const r = spawnSync('python3', ['-c', driver], {input: JSON.stringify(pyViews), encoding: 'utf8', env: ENV});
      assert.equal(r.status, 0, r.stderr);
      const got = JSON.parse(r.stdout);
      views.forEach((v, j) => {
        const js = standIn({task: TASKS['eli-shop'], visible: v, notes: []}, c);
        const action = js.action.type === 'answer' ? ['answer', js.action.text] : ['call', js.action.tool, js.action.input];
        assert.deepEqual(got[j], [js.plan, action], `${JSON.stringify(c)} view ${j}`);
      });
    });
  } finally { rmSync(dir, {recursive: true, force: true}); }
});

test('kai-food: the program and the page agree for every "not found" setting', {skip: PY ? false : 'python3 not found'}, () => {
  // kai is not in the lookup table: each onNotFound option, with lookup on and off, the notepad
  // on and off and both windows. The program's kai-food block must equal the page's run.
  const dir = mkdtempSync(join(tmpdir(), 'agent-py-'));
  try {
    let i = 0;
    for (const onNotFound of ['say so', 'try again', 'guess']) for (const lookup of [true, false])
      for (const notepad of [false, true]) for (const budget of [Infinity, 12]) {
        const c = cfg({lookup, notepad}, {onNotFound, budget, askBeforeChanges: false});
        const file = join(dir, `kai_${i++}.py`);
        writeFileSync(file, toPython(c), 'utf8');
        const r = spawnSync('python3', [file], {input: '', encoding: 'utf8', env: ENV});
        assert.equal(r.status, 0, r.stderr);
        assert.equal(r.stdout, expectedOutput(c, 'y'), JSON.stringify(c));
        const block = r.stdout.trimEnd().split('\n\n')[3];
        const run = runAgent('kai-food', c);
        assert.equal(block, traceLines(run).join('\n'));
        const last = block.split('\n').slice(-2);
        if (lookup && onNotFound === 'say so') assert.deepEqual(last, ['Answer: I could not find kai.', 'Stopped: done']);
        else if (lookup && onNotFound === 'try again') assert.deepEqual(last, ['Tool call: lookup("kai")', 'Stopped: repeated call']);
        else assert.deepEqual(last, ['Answer: kai likes apples.', 'Stopped: done']);
      }
  } finally { rmSync(dir, {recursive: true, force: true}); }
});

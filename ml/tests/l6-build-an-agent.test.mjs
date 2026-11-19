import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DEFAULT_CONFIG, RULES, TASKS, TOOLS, runAgent, fmtMoney, PRICES } from '../assets/agent.js';
import { toPython, BUILD_TASKS, ASK_PROMPT, traceLines, expectedOutput } from '../assets/agent-py.js';
import * as S from '../build-an-agent/stage.js';

const html = readFileSync(new URL('../build-an-agent/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../build-an-agent/stage.js', import.meta.url), 'utf8');

const NAMES = ['calculator', 'lookup', 'notepad', 'send'];
const ALL_ON = { calculator: true, lookup: true, notepad: true, send: true };
const cfg = (tools = {}, rest = {}) => ({ ...DEFAULT_CONFIG, ...rest, tools: { ...DEFAULT_CONFIG.tools, ...tools } });

// The right answers, worked out here on their own: all tools on, no step limit, no window, no ask.
const RIGHT = Object.fromEntries(BUILD_TASKS.map((id) => [id,
  runAgent(id, { tools: ALL_ON, stepLimit: Infinity, budget: Infinity, askBeforeChanges: false })]));

// The pass rule, written out again from the brief so the stage's verdict is checked against it.
function expectPass(id, c) {
  const r = runAgent(id, c);
  if (r.stop === 'done') return r.answer === RIGHT[id].answer;
  if (id !== 'tell-zoe' || r.stop !== 'waiting for approval' || !(c.askBeforeChanges && c.tools.send)) return false;
  const rightCall = RIGHT[id].messages.find((m) => m.role === 'call' && m.tool === 'send').text;
  const ask = r.messages.find((m) => m.role === 'approval');
  const yes = runAgent(id, c, { approve: true });
  return ask.text === `Waiting for a yes before ${rightCall}` && yes.stop === 'done' && yes.answer === RIGHT[id].answer;
}

test('the build runs match runAgent, and eli-shop costs $5.00', () => {
  assert.deepEqual(S.BUILD_TASKS, ['eli-shop', 'pair-shop', 'tell-zoe']);
  assert.equal(S.STEP_COUNT, 5);
  for (const id of BUILD_TASKS) {
    assert.deepEqual(S.buildRun(id, DEFAULT_CONFIG), runAgent(id, DEFAULT_CONFIG));
    assert.deepEqual(S.rightRun(id), RIGHT[id]);
  }
  assert.equal(S.rightRun('eli-shop').answer, 'That costs $5.00.'); // the one literal
  assert.equal(RIGHT['pair-shop'].answer, `That costs ${fmtMoney(3 * PRICES.apples + 3 * PRICES.grapes)}.`);
  assert.equal(RIGHT['tell-zoe'].answer, `Told zoe: pears cost ${fmtMoney(PRICES.pears)} each.`);
});

test('the option space is the one the page offers', () => {
  assert.deepEqual(S.OPTIONS.onNotFound, ['say so', 'try again', 'guess']);
  assert.deepEqual(S.OPTIONS.sums, ['calculator', 'guess']);
  assert.deepEqual(S.OPTIONS.stepLimit, [3, 5, 8]);
  assert.deepEqual(S.OPTIONS.budget, [12, Infinity]);
  assert.deepEqual(S.OPTIONS.tools, NAMES);
  const space = S.optionSpace();
  assert.equal(space.length, 16 * 3 * 2 * 3 * 2 * 2);
  assert.equal(new Set(space.map((c) => JSON.stringify(c))).size, space.length);
  assert.ok(space.some((c) => JSON.stringify(c) === JSON.stringify({ ...DEFAULT_CONFIG })), 'defaults are in the space');
});

test('pass/fail equals runAgent for every configuration', () => {
  const tally = { pass: 0, fail: 0 };
  for (const c of S.optionSpace()) {
    for (const id of BUILD_TASKS) {
      const v = S.verdict(id, c);
      assert.deepEqual(v.result, runAgent(id, c));
      assert.equal(v.pass, expectPass(id, c), `${id} ${JSON.stringify(c)}`);
      assert.ok(typeof v.reason === 'string' && v.reason.length > 0);
      assert.ok(!/undefined|NaN|null/.test(v.reason), v.reason);
      tally[v.pass ? 'pass' : 'fail'] += 1;
    }
  }
  assert.ok(tally.pass > 0 && tally.fail > 0);
});

test('every fail reason says why, from the run', () => {
  for (const c of S.optionSpace()) {
    for (const id of BUILD_TASKS) {
      const v = S.verdict(id, c);
      const r = v.result;
      if (v.pass) {
        assert.match(v.reason, r.stop === 'done' ? /^right (answer|message)/ : /^waits for your yes before send\(/);
        assert.ok(v.reason.includes(`${r.steps} step`));
        continue;
      }
      if (r.stop === 'step limit') {
        assert.ok(v.reason.startsWith(`no answer: stopped at the step limit of ${c.stepLimit}`), v.reason);
        const big = runAgent(id, { ...c, stepLimit: S.CAP });
        if (big.stop === 'done') {
          assert.ok(v.reason.includes(`this run needs ${big.steps} steps`), v.reason);
          assert.ok(big.steps > c.stepLimit);
        } else {
          // Going round in circles happens only with the 12-word window and no notepad.
          assert.equal(big.stop, 'step limit');
          assert.equal(c.budget, 12);
          assert.equal(c.tools.notepad, false);
          assert.ok(v.reason.includes('went round in circles'), v.reason);
          assert.ok(v.reason.includes(`the ${c.budget}-word window`), v.reason);
          const m = v.reason.match(/writing (\S+\)) (\d+) times/);
          assert.ok(m, v.reason);
          assert.equal(r.messages.filter((x) => x.role === 'call' && x.text === m[1]).length, Number(m[2]));
          assert.ok(Number(m[2]) >= 2);
          // The circle is real: the first result of that call is outside the final window.
          const first = r.messages.findIndex((x) => x.role === 'call' && x.text === m[1]);
          assert.equal(r.messages[first + 1].role, 'result');
          assert.ok(r.messages[first + 1].unseen, `${id} ${JSON.stringify(c)}`);
        }
        if (r.state.sent.length) assert.ok(v.reason.includes(`sent ${r.state.sent.length} message`), v.reason);
      } else if (r.stop === 'done') {
        if (id === 'tell-zoe' && !c.tools.send) assert.equal(v.reason, 'nothing sent: send is off');
        else {
          assert.match(v.reason, /^wrong (answer|price)/);
          assert.match(v.reason, /it guessed/);
          for (const [word, re] of [['the food', /^Guess \(first food/], ['the price', /^Guess \(first price/], ['the sum', /Guess \(each number/]]) {
            const used = r.messages.some((x) => x.role === 'plan' && x.text.split(/(?<=\.) /).some((p) => re.test(p)));
            assert.equal(v.reason.includes(word), used, `${word} ${v.reason}`);
          }
        }
      } else if (r.stop === 'waiting for approval') {
        assert.ok(c.askBeforeChanges && c.tools.send);
        const yes = runAgent(id, c, { approve: true });
        if (/^asks to send a wrong price/.test(v.reason)) assert.notEqual(r.messages.find((x) => x.role === 'approval').text, `Waiting for a yes before ${RIGHT[id].messages.find((x) => x.tool === 'send').text}`);
        else assert.ok(v.reason.startsWith('after a yes: ') && yes.answer !== RIGHT[id].answer, v.reason);
      } else {
        assert.fail(`unexpected stop ${r.stop}`);
      }
    }
  }
});

test('stories told in step 2', () => {
  // Defaults: all three pass.
  const d = BUILD_TASKS.map((id) => S.verdict(id, DEFAULT_CONFIG));
  assert.deepEqual(d.map((v) => v.pass), [true, true, true]);
  assert.equal(d[0].result.steps, 4);
  assert.equal(d[1].result.steps, 6);
  assert.equal(d[2].result.stop, 'waiting for approval');
  assert.equal(d[2].result.steps, 2);
  assert.equal(runAgent('tell-zoe', DEFAULT_CONFIG, { approve: true }).steps, 3);
  // Step limit 5: only pair-shop fails; it needs 6 steps.
  const l5 = BUILD_TASKS.map((id) => S.verdict(id, cfg({}, { stepLimit: 5 })));
  assert.deepEqual(l5.map((v) => v.pass), [true, false, true]);
  assert.match(l5[1].reason, /needs 6 steps/);
  // Calculator off: wrong sums.
  const co = BUILD_TASKS.map((id) => S.verdict(id, cfg({ calculator: false })));
  assert.deepEqual(co.map((v) => v.pass), [false, false, true]);
  assert.equal(co[0].result.answer, 'That costs $4.00.');
  assert.equal(co[1].result.answer, 'That costs $14.40.');
  // A 12-word window: all three fail.
  const b = BUILD_TASKS.map((id) => S.verdict(id, cfg({}, { budget: 12 })));
  assert.deepEqual(b.map((v) => v.pass), [false, false, false]);
  assert.match(b[0].reason, /went round in circles/);
  assert.match(b[1].reason, /went round in circles/);
  assert.match(b[2].reason, /^after a yes: /);
  // A 12-word window and the notepad: eli-shop and tell-zoe pass; pair-shop needs 10 steps.
  const bn = BUILD_TASKS.map((id) => S.verdict(id, cfg({ notepad: true }, { budget: 12 })));
  assert.deepEqual(bn.map((v) => v.pass), [true, false, true]);
  assert.match(bn[1].reason, /needs 10 steps/);
  // The notepad alone (no window limit): pair-shop needs 10 steps, more than 8.
  const np = S.verdict('pair-shop', cfg({ notepad: true }));
  assert.equal(np.pass, false);
  assert.match(np.reason, new RegExp(`step limit of ${DEFAULT_CONFIG.stepLimit}; this run needs 10 steps`));
  // Step 4: after a no the run is declined; a yes covers the first question only.
  const def = S.outputs(DEFAULT_CONFIG);
  assert.ok(def[1].text.endsWith('Stopped: declined\n'));
  assert.ok(def[0].text.endsWith('Stopped: done\n'));
  const w = S.outputs(cfg({}, { budget: 12 }))[0].text.trimEnd().split('\n\n')[2].split('\n');
  assert.equal(w.filter((l) => l.startsWith('Waiting for your yes: Waiting for a yes before')).length, 2);
  assert.equal(w.filter((l) => l.startsWith(ASK_PROMPT)).length, 1);
  assert.equal(w.at(-1), 'Stopped: waiting for approval');
  // The step counts in The maths: 2k + 2, and 4k + 2 with the notepad.
  for (const [id, k] of [['eli-shop', 1], ['pair-shop', 2]]) {
    assert.equal(runAgent(id, cfg({}, { stepLimit: Infinity })).steps, 2 * k + 2);
    assert.equal(runAgent(id, cfg({ notepad: true }, { stepLimit: Infinity })).steps, 4 * k + 2);
  }
});

test('code shown equals toPython(config)', () => {
  for (const c of S.optionSpace()) {
    assert.equal(S.programText(c), toPython(c));
    assert.equal(S.lineCount(c), toPython(c).split('\n').length - 1);
  }
  assert.equal(S.lineCount(DEFAULT_CONFIG), 206);
  // Switching a default tool off drops its code, so the program gets shorter.
  for (const t of ['calculator', 'lookup', 'send']) assert.ok(S.lineCount(cfg({ [t]: false })) < S.lineCount(DEFAULT_CONFIG), t);
  assert.ok(S.lineCount(cfg({ notepad: true })) > S.lineCount(DEFAULT_CONFIG));
  assert.equal(S.FILE_NAME, 'my_agent.py');
  assert.equal(S.RUN_LINE, 'python3 my_agent.py');
  // The marked block in step 5 is the program's own text.
  for (const c of [DEFAULT_CONFIG, cfg({ calculator: false, lookup: false, send: false })]) {
    const block = S.markedBlock(c);
    assert.ok(toPython(c).includes(block));
    assert.match(block, /^# ---- To use a real language model/);
    assert.match(block, /def call_model\(messages\):$/);
  }
  // The program is safe: standard library imports only, no files, no network.
  for (const c of S.optionSpace()) {
    const p = S.programText(c);
    const imports = p.split('\n').filter((l) => /^(import|from) /.test(l)).map((l) => l.split(' ')[1]);
    for (const m of imports) assert.ok(['ast', 'math', 'operator', 're'].includes(m), m);
    for (const w of ['open(', 'socket', 'urllib', 'http', 'eval(', 'exec(', 'subprocess', 'import os']) assert.ok(!p.includes(w), w);
    if (c.tools.send) assert.match(p, /SENT\.append\(\(to, text\)\)/);
  }
});

test('expected output lines equal the parity runs', () => {
  for (const c of S.optionSpace()) {
    const outs = S.outputs(c);
    const both = c.tools.send && c.askBeforeChanges;
    assert.equal(outs.length, both ? 2 : 1);
    if (both) {
      assert.deepEqual(outs.map((o) => o.typed), ['y', 'n']);
      assert.equal(outs[0].text, expectedOutput(c, 'y'));
      assert.equal(outs[1].text, expectedOutput(c, 'n'));
      assert.notEqual(outs[0].text, outs[1].text);
      assert.ok(outs[0].text.includes(`${ASK_PROMPT}y\n`) && outs[1].text.includes(`${ASK_PROMPT}n\n`));
      assert.match(outs[0].label, /type y/);
      assert.match(outs[1].label, /type n/);
    } else {
      assert.equal(outs[0].text, expectedOutput(c, 'y'));
      assert.equal(expectedOutput(c, 'y'), expectedOutput(c, 'n'));
      assert.ok(!outs[0].text.includes(ASK_PROMPT));
    }
    // Each task's block is the trace of the run the page shows (or its yes/no run).
    const blocks = outs[0].text.trimEnd().split('\n\n');
    assert.equal(blocks.length, 3);
    assert.deepEqual(blocks.slice(0, 2), ['eli-shop', 'pair-shop'].map((id) => traceLines(runAgent(id, c)).join('\n')));
  }
});

const hasPy = (bin) => { const r = spawnSync(bin, ['--version'], { encoding: 'utf8' }); return !r.error && r.status === 0; };

test('the program the page shows prints the page\'s expected output', { skip: hasPy('python3') ? false : 'python3 not found' }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'build-agent-'));
  const env = { ...process.env, PYTHONIOENCODING: 'utf-8' };
  const sample = [DEFAULT_CONFIG, cfg({}, { stepLimit: 5 }), cfg({}, { budget: 12 }), cfg({ notepad: true }, { budget: 12 }),
    cfg({ calculator: false }, { askBeforeChanges: false }), cfg({ send: false, lookup: false }, { onNotFound: 'guess', sums: 'guess', stepLimit: 3 })];
  try {
    sample.forEach((c, i) => {
      const file = join(dir, `${S.FILE_NAME.replace('.py', '')}_${i}.py`);
      writeFileSync(file, S.programText(c), 'utf8');
      for (const o of S.outputs(c)) {
        const typed = o.typed || 'y';
        const r = spawnSync('python3', [file], { input: `${typed}\n`, encoding: 'utf8', env });
        assert.equal(r.status, 0, r.stderr);
        assert.equal(r.stdout.split(ASK_PROMPT).join(`${ASK_PROMPT}${typed}\n`), o.text, `config ${i} typed ${typed}`);
      }
    });
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('copy falls back to selection', async () => {
  assert.equal(S.COPY_LABEL, 'Copy');
  assert.equal(S.COPIED_LABEL, 'Copied');
  assert.equal(S.SELECTED_LABEL, 'Selected: press Ctrl+C (⌘C on Mac)');
  const run = async (clipboard) => {
    const seen = { selected: 0, label: S.COPY_LABEL, written: null };
    const out = await S.copyCode('print(1)\n', {
      clipboard,
      select: () => { seen.selected += 1; },
      setLabel: (t) => { seen.label = t; },
    });
    return { out, ...seen };
  };
  // Works: the text goes to the clipboard and nothing is selected.
  let written = null;
  const ok = await run({ writeText: async (t) => { written = t; } });
  assert.deepEqual([ok.out, ok.selected, ok.label, written], ['copied', 0, S.COPIED_LABEL, 'print(1)\n']);
  // No clipboard at all.
  const none = await run(undefined);
  assert.deepEqual([none.out, none.selected, none.label], ['selected', 1, S.SELECTED_LABEL]);
  // A clipboard without writeText.
  const bare = await run({});
  assert.deepEqual([bare.out, bare.selected, bare.label], ['selected', 1, S.SELECTED_LABEL]);
  // writeText rejects (no permission).
  const no = await run({ writeText: () => Promise.reject(new Error('denied')) });
  assert.deepEqual([no.out, no.selected, no.label], ['selected', 1, S.SELECTED_LABEL]);
  // writeText throws straight away.
  const boom = await run({ writeText: () => { throw new Error('boom'); } });
  assert.deepEqual([boom.out, boom.selected, boom.label], ['selected', 1, S.SELECTED_LABEL]);
  // The clipboard is used on click only: nothing in the stage calls it at load.
  assert.equal((src.match(/clipboard/g) || []).length > 0, true);
  assert.ok(!/readText/.test(src));
});

test('rules list matches RULES', () => {
  const block = html.match(/<ul class="rules-list">([^]*?)<\/ul>/);
  assert.ok(block, 'rules list present');
  const items = [...block[1].matchAll(/<li><strong>(R\d)<\/strong> ([^<]*)<\/li>/g)];
  const unescape = (s) => s.replace(/&quot;/g, '"').replace(/&amp;/g, '&');
  assert.deepEqual(items.map((m) => [m[1], unescape(m[2])]), RULES.map((r) => [r.id, r.text]));
});

test('page values are computed and match the page', () => {
  const v = S.pageValues();
  const eli = runAgent('eli-shop');
  const pair = runAgent('pair-shop');
  const zoe = runAgent('tell-zoe');
  const yes = runAgent('tell-zoe', {}, { approve: true });
  const expect = {
    'rule-count': String(RULES.length),
    'tool-count': String(Object.keys(TOOLS).length),
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
    'zoe-call': zoe.messages.find((m) => m.role === 'approval').text.replace('Waiting for a yes before ', ''),
    'l5-needs': String(runAgent('pair-shop', { stepLimit: S.CAP }).steps),
    'co-eli': runAgent('eli-shop', { tools: { calculator: false } }).answer,
    'co-pair': runAgent('pair-shop', { tools: { calculator: false } }).answer,
    'bn-needs': String(runAgent('pair-shop', { tools: { notepad: true }, budget: 12, stepLimit: S.CAP }).steps),
    'default-lines': String(toPython(DEFAULT_CONFIG).split('\n').length - 1),
    'file-name': 'my_agent.py',
    'run-line': 'python3 my_agent.py',
    'ask-prompt': ASK_PROMPT.trim(),
  };
  assert.deepEqual(v, expect);
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  const unesc = (t) => t.replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&#39;/g, "'");
  for (const [, key, text] of spans) {
    assert.ok(key in v, `unknown key ${key}`);
    assert.equal(unesc(text), v[key], key);
  }
  for (const key of Object.keys(v)) assert.ok(spans.some(([, k]) => k === key), `unused key ${key}`);
});

test('the page has the lesson parts and no free-text input', () => {
  // The row labels listed in step 4 are the trace view's, the person's answer included.
  assert.match(html, /Task, Plan, Tool call, Result, Waiting for your yes, Your answer, Answer, and Stopped/);
  assert.match(html, /<title>Build your own agent · ML Explorer<\/title>/);
  assert.match(html, /Level 6 · Agents/);
  assert.match(html, /href="\/ml\/agent-mistakes\/"/);
  assert.match(html, /<a class="next-lesson" href="\/ml\/">Back to the path →<\/a>/);
  assert.match(html, /Level 7 \(In practice\) is coming next/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.equal((html.match(/<li class="step"/g) || []).length, S.STEP_COUNT);
  const step1 = html.split('id="step-1"')[1].split('id="step-2"')[0];
  assert.match(step1, /stand-in/);
  assert.match(step1, /fixed rules/);
  assert.match(step1, /language model, which can write a wrong or malformed call/);
  const step3 = html.split('id="step-3"')[1].split('id="step-4"')[0];
  assert.match(step3, /standard library/);
  assert.match(step3, /no network/);
  const step5 = html.split('id="step-5"')[1].split('</li>')[0];
  assert.match(step5, /environment variable/);
  // call_model returns a plan line and an action, as the program's loop unpacks it.
  assert.match(step5, /returns a plan line and one action/);
  assert.match(S.markedBlock(DEFAULT_CONFIG), /a plan line and one of the same actions/);
  assert.match(S.programText(DEFAULT_CONFIG), /plan, action = call_model\(/);
  assert.ok(!/<input[^>]*type="(text|search|number)"/.test(html + src));
  assert.ok(!/<textarea/.test(html + src));
  assert.ok(!/innerHTML|eval\(|new Function|data:text|createObjectURL/.test(src));
  assert.ok(!/\son[a-z]+=/.test(html));
  const forms = [...html.matchAll(/<form class="q" id="(q\d)"[^>]*>([\s\S]*?)<\/form>/g)];
  assert.equal(forms.length, 2);
  for (const [, id, body] of forms) {
    const opts = [...body.matchAll(/type="radio" name="q\d" value="([^"]+)"/g)].map((m) => m[1]);
    assert.ok(opts.length >= 3, id);
    for (const o of opts) assert.match(body, new RegExp(`class="why" data-for="${o}"`), `${id} ${o}`);
  }
  const text = html.replace(/<[^>]+>/g, ' ').replace(/Intelligent Agents/g, '');
  for (const w of ['thinks', 'knows', 'understands', 'wants', 'decides', 'autonomous', 'intelligent', 'remembers', 'reasons', 'creative', 'conscious']) {
    assert.ok(!new RegExp(`\\b${w}\\b`, 'i').test(text), w);
  }
  assert.ok(!/30.60 lines|30 to 60/.test(text));
});

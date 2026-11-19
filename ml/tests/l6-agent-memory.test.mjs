import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DEFAULT_CONFIG, PRICES, TASKS, runAgent, contextWindow, countWords, calc, fmtMoney, lookupText } from '../assets/agent.js';
import { WORLD } from '../assets/world.js';
import * as S from '../agent-memory/stage.js';

const html = readFileSync(new URL('../agent-memory/index.html', import.meta.url), 'utf8');
const src = readFileSync(new URL('../agent-memory/stage.js', import.meta.url), 'utf8');

const calls = (r) => r.messages.filter((m) => m.role === 'call');
const planIndex = (r, s) => r.messages.findIndex((m) => m.role === 'plan' && m.step === s);
const cfg = (budget, notepad) => ({ stepLimit: 10, budget, tools: { notepad } });

test('the lesson runs match runAgent, and the pair task costs $12.90', () => {
  const full = runAgent('pair-shop', { stepLimit: 10 });
  const g = TASKS['pair-shop'].goal;
  const expr = g.people.map((p) => `${g.qty}*${PRICES[WORLD.likes[p]]}`).join('+');
  assert.equal(full.answer, `That costs ${fmtMoney(calc(expr).value)}.`);
  assert.equal(full.answer, 'That costs $12.90.'); // the one literal
  assert.equal(S.TASK, 'pair-shop');
  assert.equal(S.STEP_LIMIT, 10);
  assert.deepEqual(S.fullRun(), full);
  for (const b of [...S.BUDGETS, Infinity]) {
    for (const np of [false, true]) assert.deepEqual(S.memoryRun(b, np), runAgent('pair-shop', cfg(b, np)));
  }
  for (const c of S.COMPARE_RUNS) assert.deepEqual(c.run(), runAgent('pair-shop', cfg(c.budget, c.notepad)));
});

test('window story at 12 words', () => {
  assert.equal(S.SMALL_BUDGET, 12);
  const r = S.memoryRun(12, false);
  assert.equal(r.stop, 'step limit');
  assert.equal(r.steps, S.STEP_LIMIT);
  assert.equal(r.answer, null);
  // The calls alternate: lookup ana, lookup apples, lookup ana, ...; sam is never looked up.
  const texts = calls(r).map((m) => m.text);
  const ana = `lookup("ana")`;
  const apples = `lookup("${WORLD.likes.ana}")`;
  texts.forEach((t, k) => assert.equal(t, k % 2 === 0 ? ana : apples));
  assert.ok(!texts.some((t) => t.includes('sam')));
  // No call is the same as the one just before it, so the repeat check never fires.
  for (let k = 1; k < texts.length; k++) assert.notEqual(texts[k], texts[k - 1]);
  // Step 2's call is given ana's result; step 3's call is not, so it looks ana up again.
  const anaResult = r.messages.findIndex((m) => m.role === 'result' && m.text === lookupText('ana'));
  assert.equal(anaResult, 3);
  assert.ok(r.messages[planIndex(r, 2)].seen.includes(anaResult));
  assert.ok(!r.messages[planIndex(r, 3)].seen.includes(anaResult));
  assert.equal(calls(r).find((m) => m.step === 3).text, ana);
  assert.equal(S.againStep(r), 3);
  // Step 3's window: the three newest messages fill the budget exactly; ana's result would overflow it.
  const w = S.windowParts(r.messages, planIndex(r, 3), 12);
  assert.deepEqual(w.kept.map((k) => k.i), [6, 5, 4]);
  assert.equal(w.total, 12);
  assert.equal(w.next.i, anaResult);
  assert.equal(w.next.words, countWords(lookupText('ana')));
  assert.equal(S.sumText(w), '4 + 1 + 7 = 12');
  // Counts of each lookup.
  assert.equal(texts.filter((t) => t === ana).length, 5);
  assert.equal(texts.filter((t) => t === apples).length, 5);
  // At the default step limit it stops at the limit too.
  assert.equal(runAgent('pair-shop', { budget: 12 }).stop, 'step limit');
  // Every budget from 8 to 52 goes in circles until the step limit; 6 is stopped by the repeat check
  // after 2 steps; 54 and up are done in the same 6 steps as with no limit.
  for (const b of S.BUDGETS) {
    const x = S.memoryRun(b, false);
    if (b === 6) { assert.equal(x.stop, 'repeated call'); assert.equal(x.steps, 2); }
    else if (b <= 52) { assert.equal(x.stop, 'step limit'); assert.equal(x.answer, null); }
    else { assert.equal(x.stop, 'done'); assert.equal(x.steps, S.fullRun().steps); assert.equal(x.answer, S.fullRun().answer); }
  }
  assert.equal(S.minBudgetDone(false), 54);
  // At 6 words ana's result alone (8 words) is over the budget, so step 2's call is given only the task.
  const tiny = S.memoryRun(6, false);
  assert.deepEqual(tiny.messages[planIndex(tiny, 2)].seen, [0]);
  assert.ok(countWords(lookupText('ana')) > 6);
});

test('notepad story', () => {
  const full = S.fullRun();
  assert.equal(full.stop, 'done');
  assert.equal(full.steps, 6);
  const r = S.memoryRun(12, true);
  assert.equal(r.stop, 'done');
  assert.equal(r.steps, 10);
  assert.equal(r.answer, full.answer);
  // Four notes: the four lookup results, in order.
  const looked = r.messages.filter((m) => m.role === 'result' && m.tool === 'lookup').map((m) => m.text);
  assert.equal(r.state.notes.length, 4);
  assert.deepEqual(r.state.notes, looked);
  assert.deepEqual(S.notesUpTo(r.messages, r.messages.length - 1), r.state.notes);
  // Notes grow one at a time, as each notepad result row appears.
  const counts = r.messages.map((_, i) => S.notesUpTo(r.messages, i).length);
  for (let i = 1; i < counts.length; i++) {
    const m = r.messages[i];
    assert.equal(counts[i] - counts[i - 1], m.role === 'result' && m.tool === 'notepad' ? 1 : 0);
  }
  // Each lookup is written down in the next step; no fact is looked up twice.
  const lookups = calls(r).filter((m) => m.tool === 'lookup').map((m) => m.text);
  assert.equal(new Set(lookups).size, lookups.length);
  assert.equal(calls(r).filter((m) => m.tool === 'notepad').length, 4);
  // The notepad costs one step per note: 10 - 6 = 4, also with no budget.
  assert.equal(r.steps - full.steps, r.state.notes.length);
  const noLimit = S.memoryRun(Infinity, true);
  assert.equal(noLimit.stop, 'done');
  assert.equal(noLimit.steps, 10);
  // That is why this lesson uses a step limit of 10: at 8 or 9 the notepad run stops at the limit.
  assert.equal(DEFAULT_CONFIG.stepLimit, 8);
  assert.equal(runAgent('pair-shop', { budget: 12, tools: { notepad: true } }).stop, 'step limit');
  assert.equal(runAgent('pair-shop', { budget: 12, stepLimit: 9, tools: { notepad: true } }).stop, 'step limit');
  // With the notepad, every budget from 8 up is done in 10 steps; at 6 the repeat check stops it.
  for (const b of S.BUDGETS) {
    const x = S.memoryRun(b, true);
    if (b === 6) { assert.equal(x.stop, 'repeated call'); assert.equal(x.steps, 2); }
    else { assert.equal(x.stop, 'done'); assert.equal(x.steps, 10); assert.equal(x.answer, full.answer); }
  }
  assert.equal(S.minBudgetDone(true), 8);
  // The notes are given to the stand-in apart from the window: in the notepad run, step 9's call is
  // given none of the lookup results, yet it writes the calculator call.
  const p9 = planIndex(r, 9);
  const lookupRows = r.messages.map((m, i) => (m.role === 'result' && m.tool === 'lookup' ? i : -1)).filter((i) => i >= 0);
  for (const i of lookupRows) assert.ok(!r.messages[p9].seen.includes(i));
  assert.equal(calls(r).find((m) => m.step === 9).tool, 'calculator');
});

test('word counts on the page equal countWords', () => {
  assert.match(src, /import \{[^}]*countWords[^}]*\} from '\.\.\/assets\/agent\.js'/);
  assert.equal(S.taskWords(), countWords(TASKS['pair-shop'].text));
  for (const [b, np] of [[12, false], [12, true], [Infinity, false], [40, false], [6, false]]) {
    const r = S.memoryRun(b, np);
    r.messages.forEach((m, i) => {
      if (m.role !== 'plan') return;
      const w = S.windowParts(r.messages, i, b);
      for (const k of w.kept) assert.equal(k.words, countWords(r.messages[k.i].text));
      assert.equal(w.total, w.kept.reduce((s, k) => s + countWords(r.messages[k.i].text), 0));
      if (w.next) assert.equal(w.next.words, countWords(r.messages[w.next.i].text));
      // The picture's lines carry exactly these numbers.
      const lines = S.windowLines(w, b);
      if (w.kept.length > 1 && lines.sum) assert.equal(lines.sum, `${w.kept.map((k) => k.words).join(' + ')} = ${w.total}`);
      assert.ok(lines.kept.includes(String(w.total)));
      if (w.next && Number.isFinite(b)) assert.ok(lines.next.includes(`${w.total} + ${w.next.words} = ${w.total + w.next.words}`));
    });
  }
});

test('window always fits the budget for every slider value', () => {
  assert.deepEqual(S.BUDGETS, Array.from({ length: 28 }, (_, k) => 6 + 2 * k));
  assert.equal(S.BUDGETS[0], S.BUDGET_MIN);
  assert.equal(S.BUDGETS.at(-1), S.BUDGET_MAX);
  for (const b of S.BUDGETS) {
    for (const np of [false, true]) {
      const r = S.memoryRun(b, np);
      r.messages.forEach((m, i) => {
        if (m.role !== 'plan') return;
        const w = S.windowParts(r.messages, i, b);
        assert.ok(w.total <= b, `budget ${b}, row ${i}`);
        assert.ok(m.seen.includes(0), 'the task is always kept');
        // The kept messages are the newest ones before this call, with no gaps.
        assert.deepEqual(w.kept.map((k) => k.i), Array.from({ length: w.kept.length }, (_, k) => i - 1 - k));
        assert.deepEqual([0, ...w.kept.map((k) => k.i).reverse()], m.seen);
        // The next older message is the first that does not fit.
        if (w.next) { assert.equal(w.next.i, i - 1 - w.kept.length); assert.ok(w.total + w.next.words > b); }
        else assert.equal(i - 1 - w.kept.length, 0);
        // Same as contextWindow over the messages before the call.
        const cw = contextWindow(r.messages.slice(0, i), b);
        assert.deepEqual(cw.map((x, k) => (x ? k : -1)).filter((k) => k >= 0), m.seen);
        // The sum line fits the picture at phone width.
        const lines = S.windowLines(w, b);
        for (const t of Object.values(lines)) if (t) assert.ok(t.length <= S.LINE_MAX, `${t} (${t.length})`);
      });
    }
  }
});

test('each row shows the window of the model call in its step', () => {
  const r = S.memoryRun(12, false);
  for (let i = 0; i < r.messages.length; i++) {
    const v = S.viewOf(r.messages, i);
    const s = Math.max(1, r.messages[i].step);
    assert.equal(v.step, s);
    assert.equal(v.planIdx, planIndex(r, s));
    const flags = S.unseenFlags(r.messages, v.planIdx);
    flags.forEach((f, k) => assert.equal(f, k < v.planIdx && !r.messages[v.planIdx].seen.includes(k)));
    assert.equal(flags[i], false, 'the selected row is never greyed');
  }
  // The stop row shows the last call's window.
  assert.equal(S.viewOf(r.messages, r.messages.length - 1).step, r.steps);
  // With no limit nothing is ever greyed.
  const f = S.fullRun();
  for (let i = 0; i < f.messages.length; i++) assert.ok(!S.unseenFlags(f.messages, S.viewOf(f.messages, i).planIdx).some(Boolean));
  // Step 3 opens on step 3's plan line, which greys ana's first lookup (rows 1 to 3).
  const flags3 = S.unseenFlags(r.messages, planIndex(r, 3));
  assert.deepEqual(flags3.map((x, k) => (x ? k : -1)).filter((k) => k >= 0), [1, 2, 3]);
});

test('page values are computed and match the page', () => {
  const v = S.pageValues();
  const full = runAgent('pair-shop', { stepLimit: 10 });
  const small = runAgent('pair-shop', cfg(12, false));
  const note = runAgent('pair-shop', cfg(12, true));
  const expect = {
    'task': TASKS['pair-shop'].text,
    'task-words': String(countWords(TASKS['pair-shop'].text)),
    'limit': '10',
    'default-limit': String(DEFAULT_CONFIG.stepLimit),
    'full-steps': String(full.steps),
    'full-answer': full.answer,
    'small': '12',
    'b-min': '6',
    'b-max': '60',
    'min-done': '54',
    'circle-from': '8',
    'circle-to': '52',
    'tiny-steps': '2',
    'ana-call': 'lookup("ana")',
    'apples-call': `lookup("${WORLD.likes.ana}")`,
    'ana-result': lookupText('ana'),
    'ana-words': String(countWords(lookupText('ana'))),
    'again-step': '3',
    'again-sum': '4 + 1 + 7 = 12',
    'again-over': `12 + ${countWords(lookupText('ana'))} = ${12 + countWords(lookupText('ana'))}`,
    'small-steps': String(small.steps),
    'ana-lookups': '5',
    'apples-lookups': '5',
    'note-steps': String(note.steps),
    'note-count': String(note.state.notes.length),
    'note-extra': String(note.steps - full.steps),
    'note-min': '8',
    'note-calc-step': '9',
  };
  assert.deepEqual(v, expect);
  const spans = [...html.matchAll(/<span data-value="([^"]+)">([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 30, `${spans.length} spans`);
  const unesc = (t) => t.replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&#39;/g, "'");
  for (const [, key, text] of spans) {
    assert.ok(key in v, `unknown key ${key}`);
    assert.equal(unesc(text), v[key], key);
  }
  for (const key of Object.keys(v)) assert.ok(spans.some(([, k]) => k === key), `unused key ${key}`);
});

test('the page has the lesson parts and no free-text input', () => {
  assert.match(html, /<title>Memory · ML Explorer<\/title>/);
  assert.match(html, /Level 6 · Agents/);
  assert.match(html, /href="\/ml\/agent-loop\/"/);
  assert.match(html, /href="\/ml\/agent-mistakes\/"/);
  assert.match(html, /href="\/ml\/tokens\/"/);
  assert.match(html, /href="\/ml\/llms\/"/);
  assert.match(html, /<p class="coffee-line">If this helped, you can <button type="button" data-coffee>buy me a coffee<\/button>\.<\/p>/);
  assert.equal((html.match(/<li class="step"/g) || []).length, S.STEP_COUNT);
  assert.ok(!/<input[^>]*type="(text|search|number)"/.test(html + src));
  assert.ok(!/<textarea/.test(html));
  assert.ok(!/innerHTML|eval\(|new Function/.test(src));
  assert.ok(!/\son[a-z]+=/.test(html));
  // Every quiz option has an explanation.
  for (const [, id, body] of html.matchAll(/<form class="q" id="(q\d)"[^>]*>([\s\S]*?)<\/form>/g)) {
    const opts = [...body.matchAll(/type="radio" name="q\d" value="([^"]+)"/g)].map((m) => m[1]);
    assert.ok(opts.length >= 3, id);
    for (const o of opts) assert.match(body, new RegExp(`class="why" data-for="${o}"`), `${id} ${o}`);
  }
  for (const w of ['thinks', 'knows', 'understands', 'wants', 'decides', 'autonomous', 'intelligent', 'remembers']) {
    assert.ok(!new RegExp(`\\b${w}\\b`, 'i').test(html.replace(/<[^>]+>/g, ' ').replace(/Intelligent Agents/g, '')), w);
  }
});

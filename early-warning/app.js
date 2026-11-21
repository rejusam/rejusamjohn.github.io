// The early-warning page: advisor lists from the exported files, and live scoring of one
// student through the proxy. Pure logic is in whatif.js and view.js (tested with node --test).
import {FIELDS, applyWhatIf, cutoffDay, isOriginal, problems, requestBody, rowToStudent, whatIfNotes} from './whatif.js';
import {compareRisk, curlSnippet, failMessage, findStudent, formatRisk, pythonSnippet} from './view.js';

const PROXY = 'https://ssew-proxy.rejusamjohn.workers.dev';
const SCORE_URL = `${PROXY}/v1/score`;
const PAGE_SIZE = 50;
const BELOW_LIST = 3;

const MEANING = {
  id_student: 'Student id (not a model input)',
  code_module: 'Module',
  clicks_total: 'Clicks on the course site so far',
  clicks_last_week: 'Clicks in the last full week',
  active_days_total: 'Days with any clicks',
  weeks_active: 'Weeks with any clicks',
  clicks_before_start: 'Clicks before the module started',
  n_due: 'Assessments due so far',
  n_submitted: 'Assessments submitted',
  n_missed: 'Assessments missed (due minus submitted)',
  n_late: 'Assessments submitted late',
  mean_score: 'Mean assessment score (empty: none yet)',
  num_of_prev_attempts: 'Earlier attempts at this module',
  studied_credits: 'Credits being studied',
  date_registration: 'Registration day (0 = module start)',
};

const $ = id => document.getElementById(id);
const files = new Map(); // "8-AAA" -> module file, fetched once per visit
const answers = new Map(); // request JSON -> {status, body}, so a repeat does not call the proxy

const state = {week: 8, module: 'AAA', page: 0, found: null, meta: null, record: null, call: 0};

// ---------- Site navigation (phone menu) ----------

function initSiteNav() {
  const root = document.documentElement;
  root.classList.add('js');
  const toggle = document.querySelector('.site-toggle');
  const links = $('site-links');
  const setOpen = open => {
    root.classList.toggle('site-open', open);
    toggle.setAttribute('aria-expanded', String(open));
  };
  toggle.addEventListener('click', () => setOpen(!root.classList.contains('site-open')));
  links.addEventListener('click', e => { if (e.target.closest('a')) setOpen(false); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && root.classList.contains('site-open')) { setOpen(false); toggle.focus(); }
  });
}

// ---------- Small DOM helpers ----------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child);
  return node;
}

function reasonsList(reasons) {
  if (!reasons || reasons.length === 0) return el('span', {class: 'note'}, '–');
  return el('ul', {class: 'reasons'}, ...reasons.map(r => el('li', {}, r)));
}

function riskCell(risk) {
  const bar = el('span', {class: 'risk-bar', 'aria-hidden': 'true'});
  bar.style.width = `${Math.round(risk * 40)}px`;
  return el('td', {class: 'num'}, formatRisk(risk), bar);
}

function openButton(position, id) {
  return el('button', {type: 'button', class: 'small', 'aria-label': `Open student ${id}`, dataset: {open: String(position)}}, 'Open');
}

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- Data ----------

async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

async function moduleFile(week, module) {
  const key = `${week}-${module}`;
  if (!files.has(key)) files.set(key, await getJSON(`data/w${week}-${module}.json`));
  return files.get(key);
}

// ---------- Advisor view ----------

function row(file, position, extra) {
  const r = file.rows[position];
  const id = r[0];
  const risk = r[file.columns.indexOf('risk')];
  const tr = el('tr', {}, el('td', {class: 'num'}, String(position + 1)), el('td', {}, String(id)), riskCell(risk));
  tr.append(el('td', {}, extra(r)), el('td', {}, openButton(position, id)));
  if (state.found === position) tr.className = 'is-found';
  return tr;
}

function renderLists(file) {
  const reasonsAt = file.columns.indexOf('reasons');
  const withReasons = r => reasonsList(r[reasonsAt]);
  const listed = [];
  for (let i = 0; i < file.listed; i++) listed.push(row(file, i, withReasons));
  $('list-body').replaceChildren(...listed);
  const below = [];
  for (let i = file.listed; i < Math.min(file.listed + BELOW_LIST, file.rows.length); i++) below.push(row(file, i, withReasons));
  $('below-body').replaceChildren(...below);

  const summary = $('list-summary');
  summary.replaceChildren(
    'Module ', el('strong', {}, file.module), ', week ', el('strong', {}, String(file.week)), ': ',
    el('strong', {}, file.cohort.toLocaleString('en-NZ')), ' students in the cohort, ',
    el('strong', {}, String(file.listed)), ' on the list.'
  );
  renderCohort(file);
}

function renderCohort(file) {
  const pages = Math.ceil(file.rows.length / PAGE_SIZE);
  state.page = Math.min(state.page, pages - 1);
  const start = state.page * PAGE_SIZE;
  const rows = [];
  for (let i = start; i < Math.min(start + PAGE_SIZE, file.rows.length); i++) {
    rows.push(row(file, i, () => (i < file.listed ? 'Yes' : '')));
  }
  $('cohort-body').replaceChildren(...rows);
  $('page-info').textContent = `Ranks ${start + 1}–${start + rows.length} of ${file.rows.length}`;
  $('page-prev').disabled = state.page === 0;
  $('page-next').disabled = state.page >= pages - 1;
}

async function showLists() {
  $('search-message').textContent = '';
  try {
    renderLists(await moduleFile(state.week, state.module));
  } catch (e) {
    $('list-summary').textContent = `The list could not be loaded (${e.message}).`;
  }
}

async function search() {
  let file;
  try {
    file = await moduleFile(state.week, state.module);
  } catch (e) {
    $('search-message').textContent = `The cohort could not be loaded (${e.message}). Try again.`;
    return;
  }
  const text = $('search').value;
  const hit = findStudent(file, text);
  if (!hit) {
    state.found = null;
    $('search-message').textContent = text.trim() === ''
      ? 'Type a student id.'
      : `Student ${text.trim()} is not in the week-${state.week} cohort for ${state.module}. They may be in another module, have registered after day ${cutoffDay(state.week)}, or have withdrawn by then.`;
    renderCohort(file);
    return;
  }
  state.found = hit.position;
  state.page = Math.floor(hit.position / PAGE_SIZE);
  $('search-message').textContent = '';
  renderLists(file);
  openRecord(file, hit.position);
}

// ---------- Student record ----------

function whatIfChange() {
  const read = id => ($(id).value === '' ? undefined : Number($(id).value));
  return {
    n_submitted: read('wi-submitted'),
    n_late: read('wi-late'),
    clicks_last_week: read('wi-clicks'),
    mean_score: read('wi-score'),
  };
}

function setWhatIfInputs(student) {
  $('wi-submitted').value = student.n_submitted;
  $('wi-submitted').max = student.n_due;
  $('wi-late').value = student.n_late;
  $('wi-late').max = student.n_submitted;
  // Nothing due yet: there is nothing to submit, so these two cannot change.
  $('wi-submitted').disabled = student.n_due === 0;
  $('wi-late').disabled = student.n_due === 0;
  $('wi-clicks').value = student.clicks_last_week;
  $('wi-score').value = student.mean_score ?? '';
  // No submissions and no score: there is nothing to score until work is submitted.
  $('wi-score').disabled = student.n_submitted === 0 && student.mean_score === null;
  $('wi-score').title = $('wi-score').disabled ? 'No work submitted yet: raise Assessments submitted first.' : '';
}

function renderFeatures(student, original) {
  const rows = FIELDS.map(f => {
    const value = student[f] === null ? 'empty' : String(student[f]);
    const changed = student[f] !== original[f];
    return el('tr', {}, el('td', {}, el('code', {}, f)), el('td', {}, MEANING[f]),
      el('td', {class: changed ? 'num changed' : 'num'}, value));
  });
  $('feature-body').replaceChildren(...rows);
}

function renderSnippets(body) {
  $('http-line').textContent = `POST ${SCORE_URL}`;
  $('http-body').textContent = JSON.stringify(body, null, 2);
  $('snippet-curl').textContent = curlSnippet(SCORE_URL, body);
  $('snippet-python').textContent = pythonSnippet(SCORE_URL, body);
}

function currentStudent() {
  const {original} = state.record;
  return applyWhatIf(original, whatIfChange());
}

function onWhatIf() {
  const change = whatIfChange();
  const student = applyWhatIf(state.record.original, change);
  setWhatIfInputs(student);
  renderFeatures(student, state.record.original);
  renderSnippets(requestBody(state.record.week, student));
  const changed = FIELDS.some(f => student[f] !== state.record.original[f]);
  const notes = whatIfNotes(state.record.week, change, student);
  if (changed) notes.push('Changed. Press Re-score to send it.');
  $('whatif-note').textContent = notes.join(' ');
}

function openRecord(file, position) {
  const r = file.rows[position];
  const original = rowToStudent(file.columns, r);
  state.record = {week: file.week, module: file.module, original, exported: r[file.columns.indexOf('risk')], position, listed: file.listed};

  $('record').hidden = false;
  $('record-title').textContent = `Student ${original.id_student}`;
  $('record-context').textContent = `Module ${file.module}, week ${file.week}. Features use data up to day ${cutoffDay(file.week)} of the module.`;
  $('risk-exported').textContent = formatRisk(state.record.exported);
  $('rank-exported').textContent = position < file.listed
    ? `Rank ${position + 1} of ${file.rows.length}: on the list`
    : `Rank ${position + 1} of ${file.rows.length}: not on the list (cut after rank ${file.listed})`;
  setWhatIfInputs(original);
  renderFeatures(original, original);
  renderSnippets(requestBody(file.week, original));
  $('whatif-note').textContent = whatIfNotes(file.week, {}, original).join(' ');
  $('record').scrollIntoView({behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start'});
  $('record-title').focus({preventScroll: true});
  score(original, true);
}

function showLive(status, body, student, first) {
  const note = $('risk-live-note');
  note.className = 'sub';
  if (status === 200 && body && body.results && body.results[0]) {
    const result = body.results[0];
    $('risk-live').textContent = formatRisk(result.risk);
    $('live-reasons').replaceChildren(...(result.reasons.length ? result.reasons.map(x => el('li', {}, x)) : [el('li', {}, 'No reason raised this score.')]));
    if (first) {
      if (compareRisk(state.record.exported, result.risk) === 'same') {
        note.textContent = `Same as the list (${result.risk.toFixed(4)}).`;
        note.classList.add('status-ok');
      } else {
        note.textContent = `Different from the list: live ${result.risk.toFixed(4)}, list ${state.record.exported.toFixed(4)}. The model may have been released again after the lists were made (${state.meta.generated_at_utc}); the export needs re-running.`;
        note.classList.add('status-warn');
      }
    } else {
      const move = result.risk - state.record.exported;
      note.textContent = `What-if: ${move >= 0 ? 'up' : 'down'} ${Math.abs(move).toFixed(4)} from the list's risk.`;
      note.classList.add(move > 0 ? 'status-warn' : 'status-ok');
    }
    return;
  }
  $('risk-live').textContent = '–';
  $('live-reasons').replaceChildren(el('li', {}, 'No live answer. The risk in the list still stands.'));
  note.classList.add('status-bad');
  note.textContent = failMessage(status, body);
}

async function score(student, first = false) {
  const week = state.record.week;
  const body = requestBody(week, student);
  const issues = problems(week, student);
  if (issues.length) {
    $('risk-live-note').textContent = `Not sent: ${issues.join('; ')}.`;
    return;
  }
  const key = JSON.stringify(body);
  const call = ++state.call;
  renderSnippets(body);
  $('rescore').disabled = true;
  $('risk-live').textContent = '…';
  let answer = answers.get(key);
  const fromMemory = Boolean(answer);
  let ms = 0;
  if (!fromMemory) {
    const started = performance.now();
    try {
      const response = await fetch(SCORE_URL, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: key,
      });
      let parsed = null;
      try { parsed = await response.json(); } catch { parsed = null; }
      answer = {status: response.status, body: parsed};
      if (response.status === 200) answers.set(key, answer);
    } catch (e) {
      answer = {status: 0, body: {error: e.message}};
    }
    ms = Math.round(performance.now() - started);
  }
  if (call !== state.call) return; // a newer call has started; its answer is the one to show
  $('http-source').textContent = fromMemory ? "from this visit's memory (no call made)" : 'live call to the proxy';
  $('http-status').textContent = answer.status ? String(answer.status) : 'no response';
  $('http-time').textContent = fromMemory ? '0 ms (memory)' : `${ms} ms`;
  $('http-response').textContent = answer.body ? JSON.stringify(answer.body, null, 2) : '(no JSON body)';
  showLive(answer.status, answer.body, student, first);
  $('rescore').disabled = false;
}

// ---------- Wiring ----------

async function init() {
  initSiteNav();
  try {
    state.meta = await getJSON('data/meta.json');
    const h = state.meta.health;
    $('meta-line').textContent = `Lists scored ${state.meta.generated_at_utc} through the live API (model released at commit ${h.released_at_commit.slice(0, 7)}).`;
  } catch (e) {
    $('meta-line').textContent = `The list files could not be loaded (${e.message}).`;
    return;
  }

  for (const button of document.querySelectorAll('[data-week]')) {
    button.addEventListener('click', () => {
      state.week = Number(button.dataset.week);
      state.page = 0;
      state.found = null;
      for (const b of document.querySelectorAll('[data-week]')) b.setAttribute('aria-pressed', String(b === button));
      $('record').hidden = true;
      showLists();
    });
  }
  $('module').addEventListener('change', e => {
    state.module = e.target.value;
    state.page = 0;
    state.found = null;
    $('record').hidden = true;
    showLists();
  });
  $('page-prev').addEventListener('click', async () => { state.page -= 1; renderCohort(await moduleFile(state.week, state.module)); });
  $('page-next').addEventListener('click', async () => { state.page += 1; renderCohort(await moduleFile(state.week, state.module)); });
  $('search-go').addEventListener('click', search);
  $('search').addEventListener('keydown', e => { if (e.key === 'Enter') search(); });

  $('advisor').addEventListener('click', async e => {
    const button = e.target.closest('[data-open]');
    if (!button) return;
    openRecord(await moduleFile(state.week, state.module), Number(button.dataset.open));
  });
  $('record-close').addEventListener('click', () => { $('record').hidden = true; });
  for (const id of ['wi-submitted', 'wi-late', 'wi-clicks', 'wi-score']) $(id).addEventListener('change', onWhatIf);
  $('rescore').addEventListener('click', () => {
    const student = currentStudent();
    score(student, isOriginal(student, state.record.original));
  });
  $('reset').addEventListener('click', () => {
    const {original} = state.record;
    setWhatIfInputs(original);
    renderFeatures(original, original);
    $('whatif-note').textContent = whatIfNotes(state.record.week, {}, original).join(' ');
    score(original, true); // from this visit's memory if it was scored already
  });
  for (const button of document.querySelectorAll('[data-copy]')) {
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText($(button.dataset.copy).textContent);
        $('copy-note').textContent = 'Copied.';
      } catch {
        $('copy-note').textContent = 'Copy did not work here; select the text instead.';
      }
    });
  }
  showLists();
}

init();

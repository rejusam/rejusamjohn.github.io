import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {FIELDS, applyWhatIf, cutoffDay, isOriginal, problems, requestBody, rowToStudent, whatIfNotes} from '../whatif.js';

const fixture = JSON.parse(readFileSync(new URL('./fixture-w8.json', import.meta.url)));
const students = fixture.rows.map(row => rowToStudent(fixture.columns, row));

// Small seeded generator so the random checks are the same on every run.
function random(seed) {
  let s = seed;
  return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
}
const between = (rand, lo, hi) => Math.floor(lo + rand() * (hi - lo + 1));

test('FIELDS are the 15 API fields in model order', () => {
  assert.equal(FIELDS.length, 15);
  assert.deepEqual(FIELDS, fixture.columns.slice(0, 15));
});

test('rowToStudent keeps exactly the request fields', () => {
  assert.deepEqual(Object.keys(students[0]), FIELDS);
});

test('cutoffDay matches features.py (7 * week - 1)', () => {
  assert.deepEqual([2, 4, 8, 12].map(cutoffDay), [13, 27, 55, 83]);
});

test('every exported student passes the API rules', () => {
  for (const s of students) assert.deepEqual(problems(8, s), [], `student ${s.id_student}`);
});

test('problems finds each broken rule', () => {
  const s = students[0];
  assert.match(problems(8, {...s, n_missed: s.n_missed + 1}).join(), /n_missed/);
  assert.match(problems(8, {...s, n_late: s.n_submitted + 1}).join(), /n_late/);
  assert.match(problems(8, {...s, clicks_last_week: s.clicks_total + 1}).join(), /clicks_total/);
  assert.match(problems(8, {...s, weeks_active: 9}).join(), /weeks_active/);
  assert.match(problems(8, {...s, date_registration: 56}).join(), /cutoff/);
  assert.match(problems(8, {...s, mean_score: 101}).join(), /mean_score/);
  assert.match(problems(8, {...s, clicks_total: -1}).join(), /clicks_total/);
  assert.match(problems(8, {...s, clicks_total: 1.5}).join(), /clicks_total/);
  assert.match(problems(8, {...s, code_module: 'ZZZ'}).join(), /code_module/);
  assert.match(problems(8, {...s, gender: 'F'}).join(), /gender/);
  assert.match(problems(3, s).join(), /week/);
  assert.deepEqual(problems(8, {...s, mean_score: null, date_registration: null}), []);
});

test('id_student has its own bound, above the count bound (OULAD ids reach 2,716,795)', () => {
  assert.deepEqual(problems(8, {...students[0], id_student: 2716795}), []);
  assert.match(problems(8, {...students[0], id_student: 100000001}).join(), /id_student/);
  assert.match(problems(8, {...students[0], clicks_total: 1000001, clicks_last_week: 0, clicks_before_start: 0}).join(), /clicks_total/);
});

test('500 random what-if changes always give a valid row', () => {
  const rand = random(7);
  for (let i = 0; i < 500; i++) {
    const original = students[i % students.length];
    const change = {
      n_submitted: between(rand, -5, original.n_due + 5),
      n_late: between(rand, -5, original.n_due + 5),
      clicks_last_week: between(rand, -10, 500),
    };
    if (rand() < 0.5) change.mean_score = between(rand, -10, 110);
    const changed = applyWhatIf(original, change);
    assert.deepEqual(problems(8, changed), [], JSON.stringify({original, change}));
  }
});

test('what-if keeps the counts in step', () => {
  const s = students.find(x => x.n_due >= 2 && x.n_submitted === x.n_due);
  const changed = applyWhatIf(s, {n_submitted: s.n_due - 1});
  assert.equal(changed.n_missed, 1);
  const more = applyWhatIf(s, {clicks_last_week: s.clicks_last_week + 40});
  assert.equal(more.clicks_total, s.clicks_total + 40);
  assert.equal(applyWhatIf(s, {n_submitted: 99}).n_submitted, s.n_due);
});

test('no score yet: raising submitted gives 50, back to 0 gives null', () => {
  const s = students.find(x => x.mean_score === null && x.n_due >= 2);
  assert.ok(s, 'fixture has a student with no score and two assessments due');
  const raised = applyWhatIf(s, {n_submitted: 2});
  assert.equal(raised.mean_score, 50);
  assert.deepEqual(problems(8, raised), []);
  const back = applyWhatIf(s, {n_submitted: 0, mean_score: 70});
  assert.equal(back.mean_score, null);
  assert.equal(applyWhatIf(s, {n_submitted: 1, mean_score: 70}).mean_score, 70);
});

test('a score from work handed in early is kept when submitted is not changed', () => {
  // The feature query counts n_submitted over assessments already due, but mean_score over every
  // known submission, so n_submitted 0 with a score is a real row.
  const s = {...students[0], n_submitted: 0, n_missed: students[0].n_due, n_late: 0, mean_score: 70};
  assert.deepEqual(problems(8, s), []);
  assert.equal(applyWhatIf(s, {clicks_last_week: s.clicks_last_week + 5}).mean_score, 70);
  assert.equal(applyWhatIf(s, {mean_score: 60}).mean_score, 60);
  const raised = applyWhatIf(s, {n_submitted: 1});
  assert.equal(raised.mean_score, 70);
});

test('a score of null with work submitted stays null unless submitted changes', () => {
  const s = {...students.find(x => x.n_submitted > 0), mean_score: null};
  assert.equal(applyWhatIf(s, {clicks_last_week: 3}).mean_score, null);
});

test('isOriginal tells a what-if row from the real one', () => {
  assert.equal(isOriginal(applyWhatIf(students[0], {}), students[0]), true);
  assert.equal(isOriginal(applyWhatIf(students[0], {clicks_last_week: students[0].clicks_last_week + 1}), students[0]), false);
});

test('a what-if does not change the original row', () => {
  const s = {...students[0]};
  applyWhatIf(s, {n_submitted: 0, clicks_last_week: 0});
  assert.deepEqual(s, students[0]);
});

test('requestBody wraps one student', () => {
  assert.deepEqual(requestBody(8, students[0]), {week: 8, students: [students[0]]});
});

test('whatIfNotes explains why nothing about assessments can change when none is due', () => {
  const s = {...students[0], n_due: 0, n_submitted: 0, n_missed: 0, n_late: 0, mean_score: null};
  const applied = applyWhatIf(s, {n_submitted: 2});
  assert.equal(applied.n_submitted, 0);
  const notes = whatIfNotes(2, {n_submitted: 2}, applied);
  assert.equal(notes.length, 1);
  assert.match(notes[0], /No assessment is due by week 2/);
  assert.match(notes[0], /clicks last week/);
});

test('whatIfNotes says when a typed value was capped', () => {
  const s = students.find(x => x.n_due >= 2);
  const applied = applyWhatIf(s, {n_submitted: s.n_due + 5, n_late: s.n_due + 9});
  const notes = whatIfNotes(8, {n_submitted: s.n_due + 5, n_late: s.n_due + 9}, applied);
  assert.ok(notes.some(n => n === `Assessments submitted is at most ${s.n_due}: the number due by week 8.`), notes);
  assert.ok(notes.some(n => n === `Submitted late is at most ${applied.n_submitted}, the number submitted.`), notes);
});

test('whatIfNotes is empty when nothing was capped', () => {
  const s = students.find(x => x.n_due >= 2);
  assert.deepEqual(whatIfNotes(8, {n_submitted: 1, n_late: 0, clicks_last_week: 3}, applyWhatIf(s, {n_submitted: 1, n_late: 0, clicks_last_week: 3})), []);
});

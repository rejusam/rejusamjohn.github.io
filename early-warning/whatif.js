// What-if changes to one student's feature row, and the scoring API's input rules.
// problems() mirrors every rule in the API (src/early_warning/api.py in
// github.com/rejusam/student-success-early-warning), so a what-if never gets a 422.

export const FIELDS = [
  'id_student', 'code_module', 'clicks_total', 'clicks_last_week', 'active_days_total',
  'weeks_active', 'clicks_before_start', 'n_due', 'n_submitted', 'n_missed', 'n_late',
  'mean_score', 'num_of_prev_attempts', 'studied_credits', 'date_registration',
];
const MODULES = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'];
const WEEKS = [2, 4, 8, 12];
const MAX_COUNT = 1000000;
const MAX_ID = 100000000;
const COUNTS = FIELDS.filter(f => !['id_student', 'code_module', 'mean_score', 'date_registration'].includes(f));
const START_SCORE = 50; // a first score when a what-if submits work for a student with none yet

// Last day of data used for a prediction at the start of `week` (features.py).
export const cutoffDay = week => 7 * week - 1;

export function rowToStudent(columns, row) {
  return Object.fromEntries(FIELDS.map(f => [f, row[columns.indexOf(f)]]));
}

export const requestBody = (week, student) => ({week, students: [student]});

export function problems(week, s) {
  const out = [];
  if (!WEEKS.includes(week)) out.push('week must be 2, 4, 8 or 12');
  for (const key of Object.keys(s)) if (!FIELDS.includes(key)) out.push(`unknown field ${key}`);
  for (const f of COUNTS) {
    if (!Number.isInteger(s[f]) || s[f] < 0 || s[f] > MAX_COUNT) {
      out.push(`${f} must be a whole number from 0 to ${MAX_COUNT}`);
    }
  }
  if (!Number.isInteger(s.id_student) || s.id_student < 0 || s.id_student > MAX_ID) {
    out.push(`id_student must be a whole number from 0 to ${MAX_ID}`);
  }
  if (!MODULES.includes(s.code_module)) out.push('code_module must be an OULAD module');
  if (s.mean_score !== null && !(typeof s.mean_score === 'number' && s.mean_score >= 0 && s.mean_score <= 100)) {
    out.push('mean_score must be empty or from 0 to 100');
  }
  if (s.date_registration !== null && !(Number.isInteger(s.date_registration) && s.date_registration >= -1000)) {
    out.push('date_registration must be empty or a whole number from -1000');
  }
  if (out.length) return out; // the rules below assume the fields are well formed
  if (s.n_missed !== s.n_due - s.n_submitted) out.push('n_missed must equal n_due - n_submitted');
  if (s.n_late > s.n_submitted) out.push('n_late cannot be more than n_submitted');
  if (Math.max(s.clicks_last_week, s.clicks_before_start) > s.clicks_total) {
    out.push('clicks_last_week and clicks_before_start are part of clicks_total');
  }
  if (s.date_registration !== null && s.date_registration > cutoffDay(week)) {
    out.push('registered after the cutoff day of the week');
  }
  if (s.weeks_active > week) out.push('weeks_active cannot be more than the weeks that have passed');
  return out;
}

const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, Math.round(value)));

// A new row from the original and the changed fields (n_submitted, n_late, clicks_last_week,
// mean_score). Fields that depend on them follow, so every rule in problems() still holds.
export function applyWhatIf(original, change) {
  const s = {...original};
  s.n_submitted = clamp(change.n_submitted ?? original.n_submitted, 0, original.n_due);
  s.n_missed = s.n_due - s.n_submitted;
  s.n_late = clamp(change.n_late ?? original.n_late, 0, s.n_submitted);

  const otherClicks = original.clicks_total - original.clicks_last_week;
  s.clicks_last_week = clamp(change.clicks_last_week ?? original.clicks_last_week, 0, MAX_COUNT - otherClicks);
  s.clicks_total = otherClicks + s.clicks_last_week;

  // The score changes only with the user's say-so. n_submitted counts work for assessments
  // already due, but mean_score covers every submission, so a student can have n_submitted 0
  // and a score (work handed in early); a what-if on clicks must not wipe that score.
  const given = change.mean_score === undefined || change.mean_score === null
    ? undefined
    : Math.min(100, Math.max(0, change.mean_score));
  const noScoreAtZero = original.mean_score === null || s.n_submitted !== original.n_submitted;
  if (s.n_submitted === 0 && noScoreAtZero) s.mean_score = null;
  else if (s.n_submitted === original.n_submitted) s.mean_score = given ?? original.mean_score;
  else s.mean_score = given ?? original.mean_score ?? START_SCORE;
  return s;
}

export const isOriginal = (student, original) => FIELDS.every(f => student[f] === original[f]);

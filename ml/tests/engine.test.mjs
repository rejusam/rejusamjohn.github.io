import test from 'node:test';
import assert from 'node:assert/strict';
import { stepFromHash, nextIndex, shouldHandleKey } from '../assets/engine.js';

test('stepFromHash clamps out of range', () => {
  assert.equal(stepFromHash('#step-4', 6), 3);
  assert.equal(stepFromHash('#step-9', 6), 0);
  assert.equal(stepFromHash('', 6), 0);
  assert.equal(stepFromHash('#foo', 6), 0);
});

test('stepFromHash rejects zero, negatives and junk', () => {
  assert.equal(stepFromHash('#step-0', 6), 0);
  assert.equal(stepFromHash('#step--2', 6), 0);
  assert.equal(stepFromHash('#step-2x', 6), 0);
  assert.equal(stepFromHash('#step-6', 6), 5);
  assert.equal(stepFromHash(undefined, 6), 0);
});

test('nextIndex clamps', () => {
  assert.equal(nextIndex(5, 1, 6), 5);
  assert.equal(nextIndex(0, -1, 6), 0);
  assert.equal(nextIndex(2, 1, 6), 3);
});

test('arrows still step from buttons and summaries; Space does not', () => {
  assert.equal(shouldHandleKey({ tagName: 'BUTTON' }, 'ArrowRight'), true);
  assert.equal(shouldHandleKey({ tagName: 'BUTTON' }, 'ArrowLeft'), true);
  assert.equal(shouldHandleKey({ tagName: 'BUTTON' }, ' '), false);
  assert.equal(shouldHandleKey({ tagName: 'SUMMARY' }, ' '), false);
  assert.equal(shouldHandleKey({ tagName: 'SUMMARY' }, 'ArrowRight'), true);
  assert.equal(shouldHandleKey({ tagName: 'INPUT', type: 'range' }, 'ArrowRight'), false);
  assert.equal(shouldHandleKey({ tagName: 'SELECT' }, 'ArrowLeft'), false);
  assert.equal(shouldHandleKey({ tagName: 'DIV', isContentEditable: true }, 'ArrowLeft'), false);
  assert.equal(shouldHandleKey({ tagName: 'BODY' }, ' '), true);
});

test('shouldHandleKey ignores form controls', () => {
  assert.equal(shouldHandleKey({ tagName: 'INPUT', type: 'range' }), false);
  assert.equal(shouldHandleKey({ tagName: 'BUTTON' }), false);
  assert.equal(shouldHandleKey({ tagName: 'SELECT' }), false);
  assert.equal(shouldHandleKey({ tagName: 'TEXTAREA' }), false);
  assert.equal(shouldHandleKey({ tagName: 'DIV', isContentEditable: true }), false);
  assert.equal(shouldHandleKey({ tagName: 'BODY' }), true);
});

import { PATH, TOTAL_LIVE, findLesson, nextLesson, lessonHref } from '../assets/path.js';

test('path lists seven levels and twenty-eight live lessons', () => {
  assert.deepEqual(PATH.map((l) => l.level), ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']);
  const live = PATH.flatMap((l) => l.lessons);
  assert.equal(live.length, TOTAL_LIVE);
  assert.equal(TOTAL_LIVE, 28);
  assert.equal(new Set(live.map((l) => l.slug)).size, TOTAL_LIVE);
  for (const l of PATH.slice(5)) {
    assert.equal(l.soon, true);
    assert.deepEqual(l.lessons, []);
  }
});

test('findLesson and nextLesson walk the path', () => {
  const found = findLesson('gradient-descent');
  assert.equal(found.level.level, 'L2');
  assert.equal(found.lesson.title, 'Gradient descent');
  assert.equal(findLesson('nope'), null);
  assert.equal(nextLesson('what-is-a-model').slug, 'data-as-numbers');
  assert.equal(nextLesson('slopes').slug, 'linear-regression');
  assert.equal(nextLesson('overfitting').slug, 'k-nearest-neighbours');
  assert.equal(nextLesson('pca').slug, 'neurons');
  assert.equal(nextLesson('rnns'), null);
  assert.equal(nextLesson('nope'), null);
  assert.equal(lessonHref('slopes'), '/ml/slopes/');
});

import {test} from 'node:test';
import assert from 'node:assert/strict';
import {
  mulberry32, randn, uniform, mean, median, variance, sd, mse, mae, euclidean,
  centralDiff, olsFit, polyFit, polyEval, gdRun, mseGrad2D, standardise,
  trainTestSplit, eig2Sym
} from '../assets/maths.js';

test('mulberry32 is deterministic', () => { const a = mulberry32(42), b = mulberry32(42); assert.equal(a(), b()); });
test('population and sample variance', () => {
  const xs = [2,4,4,4,5,5,7,9];
  assert.equal(variance(xs, {sample:false}), 4); assert.equal(sd(xs, {sample:false}), 2);
  assert.ok(Math.abs(variance(xs) - 32/7) < 1e-12);
});
test('median odd and even', () => { assert.equal(median([3,1,2]), 2); assert.equal(median([4,1,3,2]), 2.5); });
test('mse and mae', () => { assert.equal(mse([1,2,3],[1,2,5]), 4/3); assert.equal(mae([1,2,3],[1,2,5]), 2/3); });
test('euclidean', () => assert.equal(euclidean([0,0],[3,4]), 5));
test('centralDiff matches analytic', () => assert.ok(Math.abs(centralDiff(x => x**3, 2) - 12) < 1e-6));
test('olsFit hand example', () => {
  const r = olsFit([1,2,3,4],[6,5,7,10]);
  assert.ok(r.ok); assert.ok(Math.abs(r.slope-1.4)<1e-12); assert.ok(Math.abs(r.intercept-3.5)<1e-12);
});
test('olsFit returns ok:false when x has no spread', () => assert.deepEqual(olsFit([2,2,2],[1,2,3]), {ok:false, reason:'no-x-spread'}));
test('lstsq recovers exact degree-5 polynomial', () => {
  const xs = Array.from({length:15}, (_,i) => -1 + 2*i/14), c = [0.5,-1,2,0.3,-0.7,1.1];
  const r = polyFit(xs, xs.map(x => polyEval(c,x)), 5);
  r.coef.forEach((v,i) => assert.ok(Math.abs(v-c[i]) < 1e-8));
});
test('ridge shrinks coefficient norm as lambda grows', () => {
  const rng = mulberry32(3), xs = Array.from({length:15}, () => uniform(rng,-1,1)), ys = xs.map(x => Math.sin(Math.PI*x)+0.25*randn(rng));
  const n = l => Math.hypot(...polyFit(xs, ys, 12, l).coef.slice(1));
  assert.ok(n(1e-6) > n(1e-2) && n(1e-2) > n(1));
});
test('train MSE is non-increasing in degree (lambda 0)', () => {
  const rng = mulberry32(3), xs = Array.from({length:15}, () => uniform(rng,-1,1)), ys = xs.map(x => Math.sin(Math.PI*x)+0.25*randn(rng));
  let prev = Infinity;
  for (let d = 0; d <= 12; d++) { const f = polyFit(xs, ys, d); const m = mse(ys, xs.map(x => polyEval(f.coef,x))); assert.ok(m <= prev + 1e-9); prev = m; }
});
test('gd converges to OLS', () => {
  const xs=[1,2,3,4], ys=[6,5,7,10], r = gdRun({grad: mseGrad2D(xs,ys), init:[0,0], eta:0.05, steps:20000});
  assert.equal(r.status,'converged'); const [b,w] = r.path.at(-1);
  assert.ok(Math.abs(w-1.4)<1e-6 && Math.abs(b-3.5)<1e-6);
});
test('gdRun flags divergence and never returns non-finite values', () => {
  const r = gdRun({grad: w => [2*w[0]], init:[1], eta:1.5, steps:5000});
  assert.equal(r.status,'diverged'); assert.ok(r.path.flat().every(Number.isFinite));
});
test('standardise gives mean 0, population sd 1', () => { const {z} = standardise([10,20,30,40]); assert.ok(Math.abs(mean(z))<1e-12 && Math.abs(sd(z,{sample:false})-1)<1e-12); });
test('trainTestSplit is a disjoint cover', () => {
  const {train,test:te} = trainTestSplit(40, 0.3, mulberry32(1));
  assert.equal(te.length, 12); assert.deepEqual([...train,...te].sort((a,b)=>a-b), [...Array(40).keys()]);
});
test('eig2Sym', () => assert.deepEqual(eig2Sym([[2,0],[0,2]]), [2,2]));

// Two-class data for the tree lessons: points uniform on [0, 10]², labelled by
// a rule made of boxes, then each label flipped with probability flip.
// Pure; used by the decision-tree and random-forest lessons.
import {mulberry32} from './maths.js';

// The rule before noise: class 1 if (x₁ > 5 and x₂ > 3) or (x₁ < 3 and x₂ > 7).
export const treeRule = ([a, b]) => ((a > 5 && b > 3) || (a < 3 && b > 7) ? 1 : 0);

// Draw order, one point at a time: x₁ = 10u, x₂ = 10u, then one more u;
// the label is flipped when that u < flip. Three draws per point, always.
export function treeData({n, seed, flip = 0.1}) {
  const rng = mulberry32(seed);
  const X = [], y = [], flipped = [];
  for (let i = 0; i < n; i++) {
    const x = [10 * rng(), 10 * rng()];
    let c = treeRule(x);
    if (rng() < flip) {
      c = 1 - c;
      flipped.push(i);
    }
    X.push(x);
    y.push(c);
  }
  return {X, y, flipped};
}

// The full learning path. L0 to L2 are live; L3 to L6 show the road ahead
// without links. Lesson pages live at /ml/<slug>/.

export const PATH = [
  {
    level: 'L0',
    name: 'Start here',
    lessons: [{ slug: 'what-is-a-model', title: 'What is a model?' }],
  },
  {
    level: 'L1',
    name: 'Data & intuition',
    lessons: [
      { slug: 'data-as-numbers', title: 'Data as numbers' },
      { slug: 'centre-and-spread', title: 'Centre and spread' },
      { slug: 'measuring-error', title: 'Measuring error' },
      { slug: 'slopes', title: 'Slopes and derivatives' },
    ],
  },
  {
    level: 'L2',
    name: 'Learning from data',
    lessons: [
      { slug: 'linear-regression', title: 'Linear regression' },
      { slug: 'gradient-descent', title: 'Gradient descent' },
      { slug: 'loss-surface', title: 'Many features and the loss surface' },
      { slug: 'train-test', title: 'Train, validate, test' },
      { slug: 'overfitting', title: 'Overfitting and regularisation' },
    ],
  },
  { level: 'L3', name: 'Classic ML', lessons: [], soon: true, topics: 'Logistic regression, k-NN, trees, ensembles, k-means, PCA' },
  { level: 'L4', name: 'Deep learning', lessons: [], soon: true, topics: 'Neurons, backprop, optimisers, CNNs, RNNs' },
  { level: 'L5', name: 'Modern AI', lessons: [], soon: true, topics: 'Attention, transformers, LLMs, embeddings' },
  { level: 'L6', name: 'In practice', lessons: [], soon: true, topics: 'Evaluation, calibration, interpretability, fairness' },
];

export const TOTAL_LIVE = 10;

export const lessonHref = (slug) => `/ml/${slug}/`;

// Finds a lesson and its level by slug, or null.
export function findLesson(slug) {
  for (const level of PATH) {
    const i = level.lessons.findIndex((l) => l.slug === slug);
    if (i !== -1) return { level, lesson: level.lessons[i] };
  }
  return null;
}

// The lesson after this one along the path, or null at the end.
export function nextLesson(slug) {
  const all = PATH.flatMap((l) => l.lessons);
  const i = all.findIndex((l) => l.slug === slug);
  return i === -1 || i === all.length - 1 ? null : all[i + 1];
}

// The first lesson along the path that is not done, or null when all are.
// isDone(slug) says whether a lesson is done.
export function firstNotDone(isDone) {
  for (const level of PATH) {
    for (const lesson of level.lessons) if (!isDone(lesson.slug)) return lesson;
  }
  return null;
}

// The landing page's one start button. With no lesson done it starts at L0;
// part way through it continues at the first lesson not done; with every
// lesson done it offers L0 again. isDone(slug) says whether a lesson is done.
export function startLink(isDone) {
  const all = PATH.flatMap((level) => level.lessons);
  const first = all[0];
  const atStart = { label: `Start at L0: ${first.title}`, href: lessonHref(first.slug) };
  if (!all.some((l) => isDone(l.slug))) return atStart;
  const next = firstNotDone(isDone);
  if (!next) return { label: `Start again at L0: ${first.title}`, href: lessonHref(first.slug) };
  return { label: `Continue: ${next.title}`, href: lessonHref(next.slug) };
}

// Canonical display names for the 5 base models — single source of truth so
// the same model never renders as "xgboost" in one place and "XGBoost" in another.
export const MODEL_LABEL = {
  logistic_regression: 'Logistic Regression',
  random_forest: 'Random Forest',
  xgboost: 'XGBoost',
  adaboost: 'AdaBoost',
  knn: 'KNN',
}

const SUFFIX_LABEL = { a: '(Team A)', b: '(Team B)', diff: '(Δ A−B)' }
const UNIT_WORD = { cm: 'cm', kg: 'kg' }

// "movement_agility_b" -> "Movement Agility (Team B)"; "weight_kg_diff" -> "Weight (kg) (Δ A−B)"
export function humanizeFeature(name) {
  const parts = name.split('_')
  const suffix = SUFFIX_LABEL[parts[parts.length - 1]]
  const words = suffix ? parts.slice(0, -1) : parts
  const title = words
    .map((w) => (UNIT_WORD[w] ? `(${UNIT_WORD[w]})` : w[0].toUpperCase() + w.slice(1)))
    .join(' ')
  return suffix ? `${title} ${suffix}` : title
}

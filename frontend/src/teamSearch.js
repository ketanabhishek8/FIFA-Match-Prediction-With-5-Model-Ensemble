/**
 * Ranked name search for the Team A/B comboboxes.
 *
 * The /teams list is ~190 national sides, so the user is nearly always typing a
 * name they already know rather than browsing. Matching therefore has to be
 * forgiving in the three ways people actually miss:
 *   - punctuation/diacritics ("cote d'ivoire", "curacao", "sao tome"),
 *   - the name they call the team vs. the one the dataset uses ("USA",
 *     "Holland", "Ireland", "Czechia" -- see ALIASES),
 *   - plain typos ("Brasil", "Germny").
 * Results are ranked, not merely filtered, so the closest name is first and can
 * be committed with Enter (or by tabbing away) without ever being scrolled to.
 */

const DIACRITIC = /\p{Diacritic}/gu
const ALNUM = /[a-z0-9]/

// Words that carry no signal when building initials ("bosnia and herzegovina"
// is "bh" to anyone typing it, not "bah").
const FILLER = new Set(['and', 'of', 'the', 'de', 'da', 'du', 'el', 'al'])

/**
 * Colloquial or dataset-alternate names -> the team name the API serves.
 * Keys that aren't in the served list simply never match.
 */
const ALIASES = {
  'United States': ['usa', 'united states of america'],
  'United Arab Emirates': ['uae', 'emirates'],
  Netherlands: ['holland'],
  'Republic of Ireland': ['ireland', 'eire'],
  'Czech Republic': ['czechia'],
  'Ivory Coast': ["cote d'ivoire", 'cote divoire'],
  Turkey: ['turkiye', 'türkiye'],
  'South Korea': ['korea republic', 'rok'],
  'North Korea': ['korea dpr', 'dprk'],
  China: ['china pr'],
  Taiwan: ['chinese taipei'],
  'DR Congo': ['congo dr', 'democratic republic of the congo', 'zaire'],
  Eswatini: ['swaziland'],
  'Cape Verde': ['cape verde islands', 'cabo verde'],
  Brunei: ['brunei darussalam'],
  'Bosnia and Herzegovina': ['bosnia'],
  'North Macedonia': ['macedonia'],
  Curaçao: ['curacao'],
  'São Tomé and Príncipe': ['sao tome'],
  'Guinea-Bissau': ['guinea bissau'],
  Russia: ['russian federation'],
  'Saudi Arabia': ['ksa'],
}

// Lower is better. Ordering these as one scale is what lets the list be sorted
// by "closeness" rather than by the order the API happened to send teams in.
const TIER = {
  EXACT: 0,
  PREFIX: 1,
  WORD_PREFIX: 2,
  SUBSTRING: 3,
  INITIALS: 4,
  SUBSEQUENCE: 5,
  TYPO: 6,
}

/**
 * Fold to lowercase ASCII words, keeping `map[i]` = the index in `value` that
 * produced normalized character `i`. The map is what lets a match found in the
 * folded string be highlighted in the original ("Curaçao", not "curacao").
 */
export function normalizeWithMap(value) {
  let norm = ''
  const map = []
  let pendingSpace = false
  for (let i = 0; i < value.length; i++) {
    const folded = value[i].normalize('NFD').replace(DIACRITIC, '').toLowerCase()
    for (const ch of folded) {
      if (!ALNUM.test(ch)) {
        pendingSpace = true
        continue
      }
      if (pendingSpace && norm) {
        norm += ' '
        map.push(i)
      }
      pendingSpace = false
      norm += ch
      map.push(i)
    }
  }
  return { norm, map }
}

export const normalize = (value) => normalizeWithMap(value).norm

/** Initials, both with and without filler words: "bosnia and herzegovina" -> bah, bh. */
function initialsOf(words) {
  const all = words.map((w) => w[0]).join('')
  const content = words.filter((w) => !FILLER.has(w)).map((w) => w[0]).join('')
  return content && content !== all ? [all, content] : [all]
}

/**
 * Does some word of `text` spell `q` with letters dropped? ("grmny" -> "germany")
 * Anchored to a word start on purpose: unanchored, three letters find a path
 * through most names on the list ("uni" through "Burundi", "Lithuania"), which
 * buries the real matches under coincidences.
 */
function isSubsequence(q, text) {
  for (let start = 0; start < text.length; start = text.indexOf(' ', start) + 1) {
    if (text[start] === q[0]) {
      let i = 1
      for (let j = start + 1; j < text.length && i < q.length; j++) {
        if (text[j] === q[i]) i++
      }
      if (i === q.length) return true
    }
    if (text.indexOf(' ', start) < 0) break
  }
  return false
}

/** Levenshtein distance, abandoned as soon as it is known to exceed `max`. */
function withinEdits(a, b, max) {
  if (Math.abs(a.length - b.length) > max) return false
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j)
  for (let i = 1; i <= a.length; i++) {
    const row = [i]
    let best = i
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
      best = Math.min(best, row[j])
    }
    if (best > max) return false
    prev = row
  }
  return prev[b.length] <= max
}

// Typo tolerance has to scale with length: one edit in a 4-letter query is a
// different word ("mali"/"malawi"), one in a 10-letter query is a slip.
function maxEdits(length) {
  if (length < 5) return 0
  return length <= 8 ? 1 : 2
}

/** Best tier at which normalized `name` matches normalized query `q`, or null. */
function scoreName(name, q) {
  if (name === q) return { tier: TIER.EXACT, start: 0 }
  if (name.startsWith(q)) return { tier: TIER.PREFIX, start: 0 }

  const at = name.indexOf(q)
  if (at > 0) {
    return { tier: name[at - 1] === ' ' ? TIER.WORD_PREFIX : TIER.SUBSTRING, start: at }
  }

  const words = name.split(' ')
  if (initialsOf(words).some((i) => i.startsWith(q))) return { tier: TIER.INITIALS, start: -1 }
  if (q.length >= 3 && isSubsequence(q, name)) return { tier: TIER.SUBSEQUENCE, start: -1 }

  const max = maxEdits(q.length)
  if (max > 0 && (withinEdits(q, name, max) || words.some((w) => withinEdits(q, w, max)))) {
    return { tier: TIER.TYPO, start: -1 }
  }
  return null
}

/**
 * Rank `teams` (objects with a `team` field) against `query`.
 * Returns the same objects plus `highlight`: a [start, end) slice of the
 * original name to emphasise, or null when the match wasn't a literal one.
 * An empty query returns every team, unranked.
 */
export function searchTeams(teams, query, { limit = 50 } = {}) {
  const q = normalize(query)
  if (!q) return teams.map((t) => ({ ...t, highlight: null }))

  const scored = []
  for (const t of teams) {
    const { norm, map } = normalizeWithMap(t.team)
    let best = scoreName(norm, q)

    // Only consult aliases when the real name didn't already match well --
    // "China" should outrank whatever "china pr" would score it at.
    if (!best || best.tier > TIER.WORD_PREFIX) {
      for (const alias of ALIASES[t.team] ?? []) {
        const hit = scoreName(normalize(alias), q)
        if (hit && (!best || hit.tier < best.tier)) best = { tier: hit.tier, start: -1 }
      }
    }
    if (!best) continue

    const highlight = best.start >= 0
      ? [map[best.start], map[best.start + q.length - 1] + 1]
      : null
    scored.push({ ...t, highlight, _tier: best.tier, _start: best.start })
  }

  // Non-literal matches (initials, subsequence, typo) have no position to sort
  // on, so they fall to the end of their own tier rather than jumping ahead.
  const position = (start) => (start < 0 ? Number.MAX_SAFE_INTEGER : start)
  scored.sort((a, b) =>
    a._tier - b._tier ||
    position(a._start) - position(b._start) ||
    a.team.length - b.team.length ||
    a.team.localeCompare(b.team))

  return scored.slice(0, limit).map(({ _tier, _start, ...rest }) => rest)
}

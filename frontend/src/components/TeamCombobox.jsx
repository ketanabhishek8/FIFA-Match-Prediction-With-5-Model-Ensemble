import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { searchTeams } from '../teamSearch'

/**
 * Type-to-search team picker.
 *
 * `value` is always one of the served team names -- the API rejects anything
 * else -- so free text is only ever a *query*. It is resolved back to a real
 * team on Enter, on click, and on blur (best match wins); if nothing matches,
 * the field snaps back to the last committed value rather than leaving the form
 * holding a name the backend will 400 on.
 */
export default function TeamCombobox({ label, teams, value, onChange, excluded, excludedNote }) {
  // null = "showing the committed value"; a string = the user is searching.
  const [query, setQuery] = useState(null)
  const [open, setOpen] = useState(false)
  // Tracked by name, not index, so the highlight survives the list reordering
  // under it as the query changes. null = "whatever the best match is now".
  const [activeTeam, setActiveTeam] = useState(null)
  const listRef = useRef(null)
  // Focusing selects the committed name so the next keystroke replaces it. A
  // mouse-up would normally collapse that selection to the click point, so the
  // click that *caused* the focus has its default suppressed.
  const selectOnClick = useRef(false)
  const id = useId()
  const listId = `${id}-list`

  const results = useMemo(() => searchTeams(teams, query ?? ''), [teams, query])

  const named = activeTeam ? results.findIndex((r) => r.team === activeTeam) : -1
  const active = named >= 0 && results[named].team !== excluded
    ? named
    : results.findIndex((r) => r.team !== excluded)
  const activeResult = active >= 0 ? results[active] : null

  useEffect(() => {
    listRef.current?.querySelector('[data-active="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [active, open])

  function commit(team) {
    if (team) onChange(team)
    setQuery(null)
    setActiveTeam(null)
    setOpen(false)
  }

  function step(delta) {
    if (!open) { setOpen(true); return }
    for (let i = active + delta; i >= 0 && i < results.length; i += delta) {
      if (results[i].team !== excluded) { setActiveTeam(results[i].team); return }
    }
  }

  function handleKeyDown(e) {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        step(1)
        break
      case 'ArrowUp':
        e.preventDefault()
        step(-1)
        break
      case 'Enter':
        // Only swallow Enter while a suggestion is on offer -- otherwise it
        // should submit the form as usual.
        if (open && activeResult) {
          e.preventDefault()
          commit(activeResult.team)
        }
        break
      case 'Escape':
        setQuery(null)
        setActiveTeam(null)
        setOpen(false)
        break
      case 'Tab':
        if (open && query !== null && activeResult) commit(activeResult.team)
        break
      default:
    }
  }

  function handleBlur() {
    // Resolve whatever is typed to a real team, or restore the committed one.
    commit(query !== null && query.trim() && activeResult ? activeResult.team : undefined)
  }

  return (
    <div className="field combobox">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        className="combo-input"
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={open && activeResult ? `${id}-opt-${active}` : undefined}
        autoComplete="off"
        spellCheck="false"
        placeholder={teams.length ? `Search ${teams.length} teams…` : 'Loading teams…'}
        value={query ?? value}
        onChange={(e) => { setQuery(e.target.value); setActiveTeam(null); setOpen(true) }}
        onFocus={(e) => { setOpen(true); setActiveTeam(value); e.target.select() }}
        onMouseDown={(e) => { selectOnClick.current = document.activeElement !== e.currentTarget }}
        onMouseUp={(e) => {
          if (!selectOnClick.current) return
          selectOnClick.current = false
          e.preventDefault()
        }}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
      />

      {/* teams.length guards the startup window, where an open list could only
          ever say "no match" for a query the user hasn't typed yet. */}
      {open && teams.length > 0 && (
        // Mouse-down inside the list must not blur the input first: blur
        // commits, and committing before the click lands would close the list
        // out from under the pointer.
        <ul
          className="combo-list"
          id={listId}
          role="listbox"
          ref={listRef}
          onMouseDown={(e) => e.preventDefault()}
        >
          {results.map((r, i) => {
            const disabled = r.team === excluded
            return (
              <li
                key={r.team}
                id={`${id}-opt-${i}`}
                className="combo-option"
                role="option"
                aria-selected={r.team === value}
                aria-disabled={disabled || undefined}
                data-active={i === active}
                onMouseEnter={() => !disabled && setActiveTeam(r.team)}
                onClick={() => !disabled && commit(r.team)}
              >
                <span className="combo-name">
                  {r.highlight
                    ? <>
                        {r.team.slice(0, r.highlight[0])}
                        <mark>{r.team.slice(r.highlight[0], r.highlight[1])}</mark>
                        {r.team.slice(r.highlight[1])}
                      </>
                    : r.team}
                </span>
                <span className="combo-meta">
                  {disabled
                    ? excludedNote
                    : r.years.length > 1
                      ? `${r.years[0]}–${r.years[r.years.length - 1]}`
                      : r.years[0]}
                </span>
              </li>
            )
          })}
          {!results.length && (
            <li className="combo-empty">No team matches “{query}”.</li>
          )}
        </ul>
      )}
    </div>
  )
}


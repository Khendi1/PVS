import { useState, useEffect, useRef, useCallback, memo } from 'react'
import { setParam, resetParam } from '../api.js'
import LfoPanel from './LfoPanel.jsx'

// Fire at most once per animation frame — aligns sends with display refresh rate
function rafThrottle(fn) {
  let rafId = null
  let latest = null
  return (...args) => {
    latest = args
    if (rafId === null) {
      rafId = requestAnimationFrame(() => {
        fn(...latest)
        rafId = null
        latest = null
      })
    }
  }
}

function fmt(v) {
  const n = parseFloat(v)
  if (isNaN(n)) return v
  return Number.isInteger(n) ? String(n) : n.toFixed(3)
}

function ParamDropdown({ param }) {
  const [localVal, setLocalVal] = useState(param.value)

  useEffect(() => { setLocalVal(param.value) }, [param.value])

  function handleChange(e) {
    const v = e.target.value
    setLocalVal(v)
    setParam(param.name, v).catch(() => {})
  }

  const label = param.name.replace(/^[A-Z0-9_]+\./, '').replace(/_/g, ' ')
  const options = param.options || []

  return (
    <div className="param-row dropdown" title={param.name}>
      <span className="param-label">{label}</span>
      <select
        className="param-select"
        value={localVal}
        onChange={handleChange}
      >
        {options.map(opt => (
          <option key={opt} value={opt}>
            {opt.replace(/_/g, ' ')}
          </option>
        ))}
      </select>
    </div>
  )
}

function ParamSlider({ param }) {
  const isDropdown = param.options && param.options.length > 0 &&
    (param.type.includes('DROPDOWN') || param.type.includes('RADIO') || typeof param.value === 'string')

  // showLfo must be declared before any early return to satisfy React's hook rules
  const [showLfo, setShowLfo] = useState(false)

  if (isDropdown) return <ParamDropdown param={param} />

  const range = param.max - param.min
  const step = range > 0 ? range / 200 : 0.01

  const [localVal, setLocalVal] = useState(param.value)
  const [inputVal, setInputVal] = useState(fmt(param.value))
  const dragging = useRef(false)
  const inputFocused = useRef(false)

  // Sync from external poll (MIDI/OSC) — only when not actively dragging/typing
  useEffect(() => {
    if (!dragging.current && !inputFocused.current) {
      setLocalVal(param.value)
      setInputVal(fmt(param.value))
    }
  }, [param.value])

  const sendThrottled = useCallback(
    rafThrottle((name, val) => setParam(name, val).catch(() => {})),
    [param.name]
  )

  function handleSlider(e) {
    const v = parseFloat(e.target.value)
    setLocalVal(v)
    setInputVal(fmt(v))
    sendThrottled(param.name, v)
  }

  function handleInputChange(e) { setInputVal(e.target.value) }

  function handleInputCommit() {
    inputFocused.current = false
    const v = parseFloat(inputVal)
    if (isNaN(v)) { setInputVal(fmt(localVal)); return }
    const clamped = Math.min(param.max, Math.max(param.min, v))
    setLocalVal(clamped)
    setInputVal(fmt(clamped))
    setParam(param.name, clamped).catch(() => {})
  }

  async function handleReset() {
    try {
      const res = await resetParam(param.name)
      setLocalVal(res.value)
      setInputVal(fmt(res.value))
    } catch (e) { /* ignore */ }
  }

  const label = param.name.replace(/^[A-Z0-9_]+\./, '').replace(/_/g, ' ')

  return (
    <div className="param-col">
      <div className="param-row" title={param.name}>
        <span className="param-label">{label}</span>
        <input
          type="range"
          min={param.min}
          max={param.max}
          step={step}
          value={localVal}
          onMouseDown={() => { dragging.current = true }}
          onMouseUp={() => { dragging.current = false }}
          onTouchStart={() => { dragging.current = true }}
          onTouchEnd={() => { dragging.current = false }}
          onChange={handleSlider}
        />
        <input
          className="param-value-input"
          type="text"
          value={inputVal}
          onFocus={() => { inputFocused.current = true }}
          onBlur={handleInputCommit}
          onChange={handleInputChange}
          onKeyDown={e => { if (e.key === 'Enter') handleInputCommit() }}
        />
        <button className="reset-btn" onClick={handleReset} title={`Reset to ${fmt(param.default)}`}>R</button>
        <button className="lfo-btn" onClick={() => setShowLfo(v => !v)} title="LFO">~</button>
      </div>
      {showLfo && <LfoPanel param={param} onClose={() => setShowLfo(false)} />}
    </div>
  )
}

export default memo(ParamSlider, (prev, next) =>
  prev.param.name === next.param.name &&
  prev.param.value === next.param.value &&
  prev.param.options === next.param.options
)

import { useState, useEffect, useRef, useCallback } from 'react'
import { setParam, resetParam } from '../api.js'

function debounce(fn, ms) {
  let t
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms) }
}

function fmt(v) {
  const n = parseFloat(v)
  if (isNaN(n)) return v
  return Number.isInteger(n) ? String(n) : n.toFixed(3)
}

export default function ParamSlider({ param, onExternalSync }) {
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

  const sendDebounced = useCallback(
    debounce((name, val) => setParam(name, val).catch(() => {}), 150),
    [param.name]
  )

  function handleSlider(e) {
    const v = parseFloat(e.target.value)
    setLocalVal(v)
    setInputVal(fmt(v))
    sendDebounced(param.name, v)
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

  const label = param.name.replace(/_/g, ' ')

  return (
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
    </div>
  )
}

import { useState, useRef, useEffect, useCallback } from 'react'
import { setParam } from '../api.js'

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

export default function XYPad({ params }) {
  // Only numeric (non-dropdown) params are useful for XY control
  const numericParams = params.filter(
    p => !(p.options && p.options.length > 0) && typeof p.value === 'number'
  )

  const [xIdx, setXIdx] = useState(0)
  const [yIdx, setYIdx] = useState(1)
  const [pos, setPos] = useState({ x: 0.5, y: 0.5 }) // normalized 0–1
  const padRef = useRef(null)
  const pressedRef = useRef(false)

  const xParam = numericParams[xIdx] || null
  const yParam = numericParams[yIdx] || null

  // Sync dot position when x/y param changes externally
  useEffect(() => {
    if (!xParam || !yParam) return
    const nx = (xParam.value - xParam.min) / (xParam.max - xParam.min || 1)
    const ny = (yParam.value - yParam.min) / (yParam.max - yParam.min || 1)
    setPos({ x: Math.max(0, Math.min(1, nx)), y: Math.max(0, Math.min(1, ny)) })
  }, [xParam?.name, yParam?.name, xParam?.value, yParam?.value])

  const sendBoth = useCallback(
    rafThrottle((nx, ny) => {
      if (xParam) {
        const vx = xParam.min + nx * (xParam.max - xParam.min)
        setParam(xParam.name, vx).catch(() => {})
      }
      if (yParam) {
        const vy = yParam.min + (1 - ny) * (yParam.max - yParam.min)
        setParam(yParam.name, vy).catch(() => {})
      }
    }),
    [xParam, yParam]
  )

  function getPosFromEvent(e) {
    const pad = padRef.current
    if (!pad) return null
    const rect = pad.getBoundingClientRect()
    const clientX = e.touches ? e.touches[0].clientX : e.clientX
    const clientY = e.touches ? e.touches[0].clientY : e.clientY
    const nx = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const ny = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height))
    return { x: nx, y: ny }
  }

  function handlePointerDown(e) {
    pressedRef.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    const p = getPosFromEvent(e)
    if (p) { setPos(p); sendBoth(p.x, p.y) }
  }

  function handlePointerMove(e) {
    if (!pressedRef.current) return
    const p = getPosFromEvent(e)
    if (p) { setPos(p); sendBoth(p.x, p.y) }
  }

  function handlePointerUp() {
    pressedRef.current = false
  }

  function handleTouchMove(e) {
    e.preventDefault()
    if (!pressedRef.current) return
    const p = getPosFromEvent(e)
    if (p) { setPos(p); sendBoth(p.x, p.y) }
  }

  function fmtVal(param, normalized) {
    if (!param) return '—'
    const v = param.min + normalized * (param.max - param.min)
    return v.toFixed(3)
  }

  function fmtValY(param, normalized) {
    if (!param) return '—'
    const v = param.min + (1 - normalized) * (param.max - param.min)
    return v.toFixed(3)
  }

  if (numericParams.length < 2) {
    return (
      <div style={{ padding: 16, color: 'var(--text2)', fontSize: 12 }}>
        Not enough numeric params loaded yet.
      </div>
    )
  }

  const dotX = pos.x * 100
  const dotY = pos.y * 100

  return (
    <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 8, height: '100%', boxSizing: 'border-box' }}>
      <div className="xy-controls">
        <label style={{ fontSize: 11, color: 'var(--text2)' }}>
          X&nbsp;
          <select
            className="param-select"
            value={xIdx}
            onChange={e => setXIdx(Number(e.target.value))}
          >
            {numericParams.map((p, i) => (
              <option key={p.name} value={i}>{p.name}</option>
            ))}
          </select>
        </label>
        <label style={{ fontSize: 11, color: 'var(--text2)', marginLeft: 8 }}>
          Y&nbsp;
          <select
            className="param-select"
            value={yIdx}
            onChange={e => setYIdx(Number(e.target.value))}
          >
            {numericParams.map((p, i) => (
              <option key={p.name} value={i}>{p.name}</option>
            ))}
          </select>
        </label>
      </div>

      <div
        ref={padRef}
        className="xy-pad"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onTouchMove={handleTouchMove}
        style={{ touchAction: 'none', position: 'relative' }}
      >
        {/* Crosshair lines */}
        <div style={{
          position: 'absolute',
          left: `${dotX}%`, top: 0, bottom: 0,
          width: 1, background: 'rgba(255,255,255,0.15)',
          pointerEvents: 'none',
        }} />
        <div style={{
          position: 'absolute',
          top: `${dotY}%`, left: 0, right: 0,
          height: 1, background: 'rgba(255,255,255,0.15)',
          pointerEvents: 'none',
        }} />

        {/* White dot */}
        <div style={{
          position: 'absolute',
          left: `${dotX}%`, top: `${dotY}%`,
          transform: 'translate(-50%, -50%)',
          width: 10, height: 10,
          borderRadius: '50%',
          background: 'white',
          boxShadow: '0 0 6px rgba(255,255,255,0.8)',
          pointerEvents: 'none',
        }} />

        {/* Value readout corners */}
        <div style={{ position: 'absolute', bottom: 4, left: 6, fontSize: 10, color: 'rgba(255,255,255,0.5)', pointerEvents: 'none' }}>
          {xParam ? xParam.name.replace(/^[A-Z0-9_]+\./, '') : ''}: {fmtVal(xParam, pos.x)}
        </div>
        <div style={{ position: 'absolute', bottom: 4, right: 6, fontSize: 10, color: 'rgba(255,255,255,0.5)', pointerEvents: 'none', textAlign: 'right' }}>
          {yParam ? yParam.name.replace(/^[A-Z0-9_]+\./, '') : ''}: {fmtValY(yParam, pos.y)}
        </div>
      </div>
    </div>
  )
}

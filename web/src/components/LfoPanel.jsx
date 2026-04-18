import { useState, useEffect, useRef } from 'react'
import { fetchLfo, createLfo, updateLfo, deleteLfo } from '../api.js'

const SHAPES = ['NONE', 'SINE', 'SQUARE', 'TRIANGLE', 'SAWTOOTH', 'PERLIN']

function LfoSlider({ label, value, min, max, step, onChange }) {
  return (
    <div className="lfo-row">
      <span className="lfo-label">{label}</span>
      <input
        type="range" min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="lfo-range"
      />
      <span className="lfo-val">{typeof value === 'number' ? value.toFixed(3) : value}</span>
    </div>
  )
}

export default function LfoPanel({ param, onClose }) {
  const [shape, setShape] = useState('SINE')
  const [frequency, setFrequency] = useState(0.5)
  const [amplitude, setAmplitude] = useState((param.max - param.min) / 2)
  const [phase, setPhase] = useState(0)
  const [seed, setSeed] = useState(0)
  const [active, setActive] = useState(false)
  const [loading, setLoading] = useState(true)

  // Tap-tempo state
  const tapTimesRef = useRef([])
  const tapResetTimerRef = useRef(null)
  const [tapBpm, setTapBpm] = useState(null)

  useEffect(() => {
    fetchLfo(param.name).then(lfo => {
      if (lfo) {
        setShape(lfo.shape)
        setFrequency(lfo.frequency)
        setAmplitude(lfo.amplitude)
        setPhase(lfo.phase)
        setSeed(lfo.seed)
        setActive(true)
      }
      setLoading(false)
    })
  }, [param.name])

  async function handleApply(updates = {}) {
    const config = {
      shape: updates.shape ?? shape,
      frequency: updates.frequency ?? frequency,
      amplitude: updates.amplitude ?? amplitude,
      phase: updates.phase ?? phase,
      seed: updates.seed ?? seed,
    }
    try {
      if (active) {
        await updateLfo(param.name, config)
      } else {
        await createLfo(param.name, config)
        setActive(true)
      }
    } catch (e) { /* ignore */ }
  }

  async function handleRemove() {
    try {
      await deleteLfo(param.name)
      setActive(false)
      setShape('SINE')
      onClose()
    } catch (e) { /* ignore */ }
  }

  function handleTap() {
    const now = Date.now()

    // Reset if last tap was >3 seconds ago
    if (tapTimesRef.current.length > 0) {
      const last = tapTimesRef.current[tapTimesRef.current.length - 1]
      if (now - last > 3000) {
        tapTimesRef.current = []
        setTapBpm(null)
      }
    }

    tapTimesRef.current = [...tapTimesRef.current, now]

    // Clear the auto-reset timer and set a fresh one
    if (tapResetTimerRef.current) clearTimeout(tapResetTimerRef.current)
    tapResetTimerRef.current = setTimeout(() => {
      tapTimesRef.current = []
      setTapBpm(null)
    }, 3000)

    const taps = tapTimesRef.current
    if (taps.length < 2) return

    // Compute average interval between successive taps
    let totalMs = 0
    for (let i = 1; i < taps.length; i++) {
      totalMs += taps[i] - taps[i - 1]
    }
    const avgMs = totalMs / (taps.length - 1)
    const hz = 1000 / avgMs
    const bpm = Math.round(hz * 60)

    setTapBpm(bpm)
    setFrequency(parseFloat(hz.toFixed(3)))
    handleApply({ frequency: hz })
  }

  function update(field, value, setter) {
    setter(value)
    handleApply({ [field]: value })
  }

  if (loading) return <div className="lfo-panel">loading…</div>

  return (
    <div className="lfo-panel">
      <div className="lfo-row">
        <span className="lfo-label">shape</span>
        <select
          className="param-select"
          value={shape}
          onChange={e => update('shape', e.target.value, setShape)}
        >
          {SHAPES.map(s => <option key={s} value={s}>{s.toLowerCase()}</option>)}
        </select>
        {active && (
          <button className="lfo-remove-btn" onClick={handleRemove} title="Remove LFO">✕</button>
        )}
        {!active && (
          <button className="lfo-apply-btn" onClick={() => handleApply()}>add</button>
        )}
      </div>
      {(active || true) && shape !== 'NONE' && (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <LfoSlider label="freq" value={frequency} min={0} max={2} step={0.001}
                onChange={v => update('frequency', v, setFrequency)} />
            </div>
            <button className="tap-btn" onClick={handleTap} title="Tap tempo">tap</button>
            {tapBpm !== null && (
              <span className="tap-bpm">{tapBpm} BPM</span>
            )}
          </div>
          <LfoSlider label="amp" value={amplitude} min={0} max={param.max - param.min} step={(param.max - param.min) / 200}
            onChange={v => update('amplitude', v, setAmplitude)} />
          <LfoSlider label="phase" value={phase} min={0} max={360} step={1}
            onChange={v => update('phase', v, setPhase)} />
          <LfoSlider label="seed" value={seed} min={param.min} max={param.max} step={(param.max - param.min) / 200}
            onChange={v => update('seed', v, setSeed)} />
        </>
      )}
    </div>
  )
}

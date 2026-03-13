import { useState, useEffect } from 'react'
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
          <LfoSlider label="freq" value={frequency} min={0} max={2} step={0.001}
            onChange={v => update('frequency', v, setFrequency)} />
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

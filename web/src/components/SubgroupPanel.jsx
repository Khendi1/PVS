import { useState } from 'react'
import ParamSlider from './ParamSlider.jsx'

export default function SubgroupPanel({ params }) {
  // Group by subgroup
  const subgroups = {}
  for (const p of params) {
    const sg = p.subgroup || ''
    if (!subgroups[sg]) subgroups[sg] = []
    subgroups[sg].push(p)
  }
  const sgNames = Object.keys(subgroups)

  const [activeSg, setActiveSg] = useState(sgNames[0] || '')
  const current = activeSg && subgroups[activeSg] ? activeSg : sgNames[0]
  const visibleParams = subgroups[current] || []

  return (
    <div className="subgroup-panel">
      {sgNames.length > 1 && (
        <div className="subgroup-select-wrap">
          <select
            className="subgroup-select"
            value={current}
            onChange={e => setActiveSg(e.target.value)}
          >
            {sgNames.map(sg => (
              <option key={sg} value={sg}>
                {sg ? sg.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : '(General)'}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className="param-list">
        {visibleParams.map(p => (
          <ParamSlider key={p.name} param={p} />
        ))}
      </div>
    </div>
  )
}

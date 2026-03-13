import { useState } from 'react'

export default function TabPanel({ tabs }) {
  const [active, setActive] = useState(0)
  if (!tabs || tabs.length === 0) return null

  return (
    <div className="tab-panel">
      <div className="tab-bar">
        {tabs.map((tab, i) => (
          <button
            key={tab.label}
            className={`tab-btn${active === i ? ' active' : ''}`}
            onClick={() => setActive(i)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="tab-content">
        {tabs[active]?.content}
      </div>
    </div>
  )
}

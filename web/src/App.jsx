import { useState, useEffect, useCallback } from 'react'
import { fetchParams } from './api.js'
import TabPanel from './components/TabPanel.jsx'
import SubgroupPanel from './components/SubgroupPanel.jsx'
import VideoPreview from './components/VideoPreview.jsx'
import MixerPanel from './components/MixerPanel.jsx'
import MidiTab from './components/MidiTab.jsx'

// Map param.group strings to quadrant buckets.
// The API returns group as the string form of the Groups enum,
// e.g. "Groups.SRC_1_EFFECTS", "Groups.POST_EFFECTS", "Groups.MIXER", etc.
function classifyGroup(group) {
  const g = group.toUpperCase()
  if (g.includes('SRC_1') || g.includes('SRC1')) return 'src1'
  if (g.includes('SRC_2') || g.includes('SRC2')) return 'src2'
  if (g.includes('POST'))   return 'post'
  if (g.includes('MIXER'))  return 'mixer'
  if (g.includes('SETTING') || g.includes('USER')) return 'settings'
  if (g.includes('AUDIO') || g.includes('REACT')) return 'audio'
  if (g.includes('OBS'))    return 'obs'
  return 'other'
}

function groupByGroup(params) {
  const byGroup = {}
  for (const p of params) {
    const g = p.group
    if (!byGroup[g]) byGroup[g] = []
    byGroup[g].push(p)
  }
  return byGroup
}

export default function App() {
  const [params, setParams] = useState([])   // flat array of param objects
  const [status, setStatus] = useState({ text: 'connecting...', cls: '' })

  // ── Fetch / poll ───────────────────────────────────────────────────────────
  const loadParams = useCallback(async () => {
    try {
      const data = await fetchParams()
      setParams(data)
      setStatus({ text: `${data.length} params`, cls: 'ok' })
    } catch (e) {
      setStatus({ text: 'API unavailable', cls: 'err' })
    }
  }, [])

  useEffect(() => {
    loadParams()
    const id = setInterval(loadParams, 2000)
    return () => clearInterval(id)
  }, [loadParams])

  // ── Classify params into buckets ───────────────────────────────────────────
  const byGroup = groupByGroup(params)

  // Collect unique group names per bucket
  const srcGroups = {}
  const postGroups = {}
  const mixerParams = []
  const settingsGroups = {}
  const otherGroups = {}

  for (const [gName, gParams] of Object.entries(byGroup)) {
    const bucket = classifyGroup(gName)
    if (bucket === 'src1' || bucket === 'src2') {
      srcGroups[gName] = gParams
    } else if (bucket === 'post') {
      postGroups[gName] = gParams
    } else if (bucket === 'mixer') {
      mixerParams.push(...gParams)
    } else if (bucket === 'settings' || bucket === 'audio' || bucket === 'obs') {
      settingsGroups[gName] = gParams
    } else {
      otherGroups[gName] = gParams
    }
  }

  // Tab label from group string: "Groups.SRC_1_EFFECTS" → "Src 1 Effects"
  function tabLabel(gName) {
    return gName
      .replace(/^Groups\./i, '')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
  }

  // ── Tab definitions ────────────────────────────────────────────────────────

  // Top-left: source tabs
  const srcTabs = Object.entries(srcGroups).map(([gName, gParams]) => ({
    label: tabLabel(gName),
    content: <SubgroupPanel params={gParams} />,
  }))

  // Bottom-left: post effects + any unclassified
  const postTabs = [
    ...Object.entries(postGroups).map(([gName, gParams]) => ({
      label: tabLabel(gName),
      content: <SubgroupPanel params={gParams} />,
    })),
    ...Object.entries(otherGroups).map(([gName, gParams]) => ({
      label: tabLabel(gName),
      content: <SubgroupPanel params={gParams} />,
    })),
  ]

  // Bottom-right: Mixer + Settings/Audio/OBS + MIDI tab
  const mixerTabs = [
    {
      label: 'Mixer',
      content: <MixerPanel params={mixerParams} onPatchLoad={loadParams} />,
    },
    ...Object.entries(settingsGroups).map(([gName, gParams]) => ({
      label: tabLabel(gName),
      content: <SubgroupPanel params={gParams} />,
    })),
    {
      label: 'MIDI',
      content: <MidiTab params={params} />,
    },
  ]

  return (
    <>
      <div className="app-grid">
        {/* Top-left: Source tabs */}
        <div className="quadrant">
          <TabPanel tabs={srcTabs} />
        </div>

        {/* Top-right: Video preview */}
        <div className="quadrant">
          <VideoPreview />
        </div>

        {/* Bottom-left: Post effects */}
        <div className="quadrant">
          <TabPanel tabs={postTabs} />
        </div>

        {/* Bottom-right: Mixer / Settings / MIDI */}
        <div className="quadrant">
          <TabPanel tabs={mixerTabs} />
        </div>
      </div>

      <div className={`status-bar ${status.cls}`}>{status.text}</div>
    </>
  )
}

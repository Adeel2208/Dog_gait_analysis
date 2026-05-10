import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Cell,
} from 'recharts'
import { AlertTriangle, CheckCircle, Activity, TrendingUp } from 'lucide-react'

const SEV_COLOR = {
  'Normal':       'var(--green)',
  'Mild':         '#d29922',
  'Moderate':     'var(--orange)',
  'Severe':       'var(--red)',
  'Analysing...': 'var(--muted)',
}

const LEG_FULL = {
  LF: 'Left Front', RF: 'Right Front',
  LR: 'Left Rear',  RR: 'Right Rear',
}

function Card({ title, icon, children, accent }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: `1px solid ${accent || 'var(--border)'}`,
      borderRadius: 'var(--radius)',
      padding: '20px 22px',
    }}>
      {title && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
          {icon}
          <span style={{ fontWeight: 700, fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--muted)' }}>
            {title}
          </span>
        </div>
      )}
      {children}
    </div>
  )
}

function StatRow({ label, value, unit, highlight }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '8px 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ color: 'var(--muted)', fontSize: 13 }}>{label}</span>
      <span style={{ fontWeight: 700, fontSize: 14, color: highlight || 'var(--text)' }}>
        {value}{unit && <span style={{ color: 'var(--muted)', fontWeight: 400, fontSize: 12 }}> {unit}</span>}
      </span>
    </div>
  )
}

function LegDot({ leg, affected, severity }) {
  const isAffected = leg === affected
  const color = isAffected ? SEV_COLOR[severity] || 'var(--red)' : 'var(--green)'
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
      <div style={{
        width: 44, height: 44, borderRadius: '50%',
        background: color,
        border: `3px solid ${isAffected ? 'white' : 'transparent'}`,
        boxShadow: isAffected ? `0 0 16px ${color}` : 'none',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 11, fontWeight: 700, color: '#000',
        transition: 'all 0.3s',
      }}>
        {leg}
      </div>
      <span style={{ fontSize: 11, color: 'var(--muted)' }}>{LEG_FULL[leg]}</span>
      {isAffected && (
        <span style={{ fontSize: 10, color, fontWeight: 700 }}>⚠ Affected</span>
      )}
    </div>
  )
}

export default function GaitReport({ summary }) {
  if (!summary) return null

  const g = summary.gait_assessment || {}
  const sevColor = SEV_COLOR[g.severity] || 'var(--muted)'
  const isLimping = g.is_limping

  // Step height bar data
  const stepData = Object.entries(g.step_heights || {}).map(([leg, val]) => ({
    leg: LEG_FULL[leg] || leg,
    shortLeg: leg,
    value: Math.round(val),
    fill: leg === g.affected_leg ? sevColor : 'var(--accent)',
  }))

  // Symmetry index radar data
  const siData = [
    { metric: 'Step SI (Front)', value: g.si_step?.['LF/RF'] || 0 },
    { metric: 'Step SI (Rear)',  value: g.si_step?.['LR/RR'] || 0 },
    { metric: 'Stance SI (F)',   value: g.si_stance?.['LF/RF'] || 0 },
    { metric: 'Stance SI (R)',   value: g.si_stance?.['LR/RR'] || 0 },
    { metric: 'Angle SI (F)',    value: g.si_angle?.['LF/RF'] || 0 },
    { metric: 'Angle SI (R)',    value: g.si_angle?.['LR/RR'] || 0 },
  ]

  // Knee angle bar data
  const angleData = Object.entries(g.knee_angles || {}).map(([leg, val]) => ({
    leg: LEG_FULL[leg] || leg,
    shortLeg: leg,
    value: Math.round(val),
    fill: leg === g.affected_leg ? sevColor : '#58a6ff88',
  }))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* ── Verdict banner ── */}
      <div style={{
        background: `${sevColor}18`,
        border: `1.5px solid ${sevColor}`,
        borderRadius: 'var(--radius)',
        padding: '20px 24px',
        display: 'flex', alignItems: 'center', gap: 16,
      }}>
        {isLimping
          ? <AlertTriangle size={36} color={sevColor} />
          : <CheckCircle size={36} color="var(--green)" />}
        <div>
          <div style={{ fontSize: 22, fontWeight: 800, color: sevColor }}>
            {g.label || 'Unknown'}
          </div>
          {g.affected_leg && (
            <div style={{ color: 'var(--muted)', fontSize: 14, marginTop: 2 }}>
              Primary affected limb: <strong style={{ color: 'var(--text)' }}>{LEG_FULL[g.affected_leg]}</strong>
            </div>
          )}
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Severity</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: sevColor }}>{g.severity}</div>
        </div>
      </div>

      {/* ── Leg diagram ── */}
      <Card title="Limb Status" icon={<Activity size={14} color="var(--muted)" />}>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 16, justifyItems: 'center',
        }}>
          {['LF', 'RF', 'LR', 'RR'].map(leg => (
            <LegDot key={leg} leg={leg} affected={g.affected_leg} severity={g.severity} />
          ))}
        </div>
        <div style={{
          display: 'flex', gap: 16, marginTop: 16, justifyContent: 'center',
          fontSize: 12, color: 'var(--muted)',
        }}>
          <span><span style={{ color: 'var(--green)' }}>●</span> Normal</span>
          <span><span style={{ color: sevColor }}>●</span> Affected</span>
        </div>
      </Card>

      {/* ── Two-column metrics ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

        {/* Step heights */}
        <Card title="Step Height per Leg" icon={<TrendingUp size={14} color="var(--muted)" />}>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={stepData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <XAxis dataKey="shortLeg" tick={{ fill: 'var(--muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }}
                labelStyle={{ color: 'var(--text)' }}
                formatter={v => [`${v} px`, 'Lift']}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {stepData.map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6, textAlign: 'center' }}>
            Lower lift = reduced weight-bearing
          </p>
        </Card>

        {/* Knee angles */}
        <Card title="Knee Joint Angle (ROM)" icon={<Activity size={14} color="var(--muted)" />}>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={angleData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <XAxis dataKey="shortLeg" tick={{ fill: 'var(--muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 180]} tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8 }}
                labelStyle={{ color: 'var(--text)' }}
                formatter={v => [`${v}°`, 'Angle']}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {angleData.map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6, textAlign: 'center' }}>
            Reduced angle = guarding / reduced ROM
          </p>
        </Card>
      </div>

      {/* ── Symmetry Index radar ── */}
      <Card title="Symmetry Index (SI) — 0% = perfect symmetry" icon={<Activity size={14} color="var(--muted)" />}>
        <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
          <ResponsiveContainer width="50%" height={220}>
            <RadarChart data={siData}>
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
              <Radar dataKey="value" stroke={sevColor} fill={sevColor} fillOpacity={0.25} />
            </RadarChart>
          </ResponsiveContainer>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {siData.map(d => {
              const pct = d.value
              const c = pct >= 30 ? 'var(--red)' : pct >= 15 ? 'var(--orange)' : 'var(--green)'
              return (
                <div key={d.metric}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
                    <span style={{ color: 'var(--muted)' }}>{d.metric}</span>
                    <span style={{ color: c, fontWeight: 700 }}>{pct.toFixed(1)}%</span>
                  </div>
                  <div style={{ height: 5, background: 'var(--surface2)', borderRadius: 999 }}>
                    <div style={{ height: '100%', width: `${Math.min(pct, 100)}%`, background: c, borderRadius: 999 }} />
                  </div>
                </div>
              )
            })}
            <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4 }}>
              SI &gt;15% = asymmetric · &gt;30% = severe
            </p>
          </div>
        </div>
      </Card>

      {/* ── Compensatory signs ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <Card title="Compensatory Signs">
          <StatRow label="Head Bob (nose std)" value={g.head_bob?.toFixed(1)} unit="px"
            highlight={g.head_bob > 8 ? 'var(--orange)' : 'var(--green)'} />
          <StatRow label="Hip Hike (hip std)" value={g.hip_hike?.toFixed(1)} unit="px"
            highlight={g.hip_hike > 8 ? 'var(--orange)' : 'var(--green)'} />
        </Card>
        <Card title="Stance Ratio">
          {Object.entries(g.stance_ratios || {}).map(([leg, val]) => (
            <StatRow key={leg} label={LEG_FULL[leg] || leg}
              value={(val * 100).toFixed(1)} unit="%"
              highlight={leg === g.affected_leg ? sevColor : undefined} />
          ))}
        </Card>
      </div>

      {/* ── Clinical findings ── */}
      {g.reasons?.length > 0 && (
        <Card title="Clinical Findings" accent={`${sevColor}55`}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {g.reasons.map((r, i) => (
              <div key={i} style={{
                display: 'flex', gap: 10, alignItems: 'flex-start',
                padding: '10px 12px',
                background: 'var(--surface2)', borderRadius: 'var(--radius-sm)',
                fontSize: 13,
              }}>
                <span style={{ color: sevColor, marginTop: 1 }}>⚠</span>
                <span style={{ color: 'var(--text)' }}>{r}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* ── Video stats ── */}
      <Card title="Processing Stats">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0 }}>
          <StatRow label="Total Frames"      value={summary.total_frames} />
          <StatRow label="Frames with Dog"   value={summary.frames_with_dog} />
          <StatRow label="Detection Rate"    value={summary.detection_rate} unit="%" />
          <StatRow label="Avg Dogs / Frame"  value={summary.avg_dogs_per_frame} />
          <StatRow label="Avg Keypoints / Frame" value={summary.avg_kpts_per_frame} />
          <StatRow label="Output Size"       value={summary.output_size_mb} unit="MB" />
        </div>
      </Card>

    </div>
  )
}

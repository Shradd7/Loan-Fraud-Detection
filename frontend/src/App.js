import React, { useEffect, useMemo, useState } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

const sampleCreditRecord = {
  ACCT_AGE: 1.6,
  LIMIT: 1005500,
  OUTS: 494161,
  LOAN_TENURE: 914,
  INSTALAMT: 38513,
  AGE: 57,
  KYC_SCR: 110,
  ALL_LON_LIMIT: 1805500,
  ALL_LON_OUTS: 527742,
  NO_ENQ: 1,
  CRIFF_33: 513,
  INCOME_BAND1: 'G',
  PRODUCT_TYPE: 'PERSONAL LOAN',
  AGREG_GROUP: '#Total Xpress Credit'
};

const hackathonResults = {
  accuracy: 0.9,
  f1: 0.6,
  precision: 0.61,
  recall: 0.61
};

function App() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [accountId, setAccountId] = useState('123456');
  const [uniqueId, setUniqueId] = useState('2045');
  const [locationResult, setLocationResult] = useState(null);
  const [locationError, setLocationError] = useState('');
  const [riskResult, setRiskResult] = useState(null);
  const [riskError, setRiskError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', model_trained: false, location_records: 0 }));

    fetch(`${API_BASE}/model-info`)
      .then((response) => response.json())
      .then((data) => setMetrics(parseMetrics(data.metrics_json)))
      .catch(() => setMetrics(null));
  }, []);

  const readiness = useMemo(() => {
    if (!health) return 'Checking';
    return health.model_trained ? 'Model ready' : 'Training pending';
  }, [health]);

  const verifyLocation = async () => {
    setLoading(true);
    setLocationError('');
    setLocationResult(null);
    try {
      const response = await fetch(`${API_BASE}/location-verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ account_id: accountId })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to verify location');
      setLocationResult(data);
    } catch (error) {
      setLocationError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const scoreRisk = async () => {
    setRiskError('');
    setRiskResult(null);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ records: [sampleCreditRecord] })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to score risk');
      setRiskResult(data.results[0]);
    } catch (error) {
      setRiskError(error.message);
    }
  };

  const scoreRiskById = async () => {
    setRiskError('');
    setRiskResult(null);
    try {
      const response = await fetch(`${API_BASE}/predict-by-id`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ unique_id: uniqueId })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to score risk');
      setRiskResult(data);
    } catch (error) {
      setRiskError(error.message);
    }
  };

  const probability = riskResult?.fraud_probability ?? riskResult?.default_probability ?? 0;

  return (
    <main>
      <section className="hero">
        <nav className="navbar">
          <div className="logo">FICOFORCE</div>
          <div className="nav-links">
            <a href="#task1">TASK 1</a>
            <a href="#task2">TASK 2</a>
            <a href="#metrics">METRICS</a>
          </div>
        </nav>

        <div className="hero-grid">
          <div className="hero-copy">
            <p className="eyebrow">Fraud Intelligence Console</p>
            <h1>DETECT, TRACK<br />SECURE</h1>
            <p>
              Structured fraud-risk scoring and evidence-based location review for credit teams.
            </p>
          </div>

          <div className="search-panel">
            <div className="panel-title">
              <span>Account Review</span>
              <strong>{readiness}</strong>
            </div>
            <input
              value={accountId}
              onChange={(event) => setAccountId(event.target.value)}
              placeholder="Enter Account ID"
            />
            <div className="button-row">
              <button onClick={verifyLocation} disabled={loading}>
                {loading ? 'Checking' : 'Verify Location'}
              </button>
              <button className="secondary-button" onClick={scoreRisk}>
                Score Sample Risk
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="status-band" id="metrics">
        <Status label="API" value={health?.status || 'checking'} tone={health?.status === 'ok' ? 'good' : 'warn'} />
        <Status label="ML Model" value={readiness} tone={health?.model_trained ? 'good' : 'warn'} />
        <Status label="Task 2 Records" value={health?.location_records ?? '-'} tone="info" />
        <Metric label="Hackathon F1" value={formatMetric(hackathonResults.f1)} />
        <Metric label="Precision" value={formatMetric(hackathonResults.precision)} />
        <Metric label="Recall" value={formatMetric(hackathonResults.recall)} />
      </section>

      <section className="workspace">
        <section className="panel" id="task1">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Task 1</p>
              <h2>ML Fraud Risk Detection</h2>
            </div>
            <span className="pill">LightGBM</span>
          </div>
          <p className="muted">
            Scores Task 1 records by `UNIQUE_ID` with the trained LightGBM model and tuned threshold.
          </p>
          <div className="input-row">
            <input value={uniqueId} onChange={(event) => setUniqueId(event.target.value)} placeholder="Task 1 UNIQUE_ID" />
            <button onClick={scoreRiskById}>Score ID</button>
          </div>
          <button className="wide-button secondary-button" onClick={scoreRisk}>Score Sample Application</button>
          {riskError && <p className="error">{riskError}</p>}
          {riskResult && (
            <div className="risk-meter">
              <div className="meter-track">
                <div className="meter-fill" style={{ width: `${Math.round(probability * 100)}%` }} />
              </div>
              <div className="score-line">
                <strong>{Math.round(probability * 100)}% fraud probability</strong>
                <span className={`confidence ${riskResult.prediction ? 'low' : 'high'}`}>{riskResult.label}</span>
              </div>
              <p>Decision threshold: {riskResult.threshold}</p>
            </div>
          )}
          {metrics && (
            <>
              <div className="comparison-table">
                <div className="comparison-row header">
                  <span>Metric</span>
                  <span>Hackathon</span>
                </div>
                <ComparisonRow label="Accuracy" value={hackathonResults.accuracy} />
                <ComparisonRow label="F1" value={hackathonResults.f1} />
                <ComparisonRow label="Precision" value={hackathonResults.precision} />
                <ComparisonRow label="Recall" value={hackathonResults.recall} />
              </div>
              <div className="metric-grid">
                <Metric label="Threshold" value={formatMetric(metrics.threshold)} />
                <Metric label="Test Rows" value={formatCount(metrics.test_rows)} />
              </div>
            </>
          )}
        </section>

        <section className="panel" id="task2">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Task 2</p>
              <h2>RAG Location Verification</h2>
            </div>
            <span className="pill">4-agent RAG</span>
          </div>
          <p className="muted">
            Retrieves branch/state knowledge and compares identity signals with activity locations.
          </p>
          <div className="input-row">
            <input value={accountId} onChange={(event) => setAccountId(event.target.value)} placeholder="Account ID" />
            <button onClick={verifyLocation} disabled={loading}>{loading ? 'Checking' : 'Verify'}</button>
          </div>
          {locationError && <p className="error">{locationError}</p>}
          {locationResult && (
            <div className="result-block">
              <div className="score-line">
                <strong>{locationResult.predicted_location}</strong>
                <span className={`confidence ${locationResult.confidence.toLowerCase()}`}>{locationResult.confidence}</span>
              </div>
              <p>
                {locationResult.predicted_state} - Score {locationResult.score} - Conflicts {locationResult.conflict_count}
              </p>
              <p className={locationResult.manual_review ? 'review warn-text' : 'review good-text'}>
                {locationResult.manual_review ? 'Manual review recommended' : 'Auto-review eligible'}
              </p>
              {locationResult.agent_outputs && (
                <div className="agent-stack">
                  {locationResult.agent_outputs.map((agent) => (
                    <div className="agent-card" key={agent.agent}>
                      <div>
                        <span>{agent.agent}</span>
                        <strong>{agent.summary}</strong>
                      </div>
                      <em>Score {formatMetric(agent.score)}</em>
                    </div>
                  ))}
                </div>
              )}
              <div className="evidence-list">
                {locationResult.evidence.slice(0, 6).map((item, index) => (
                  <div className="evidence" key={`${item.source}-${index}`}>
                    <span>{item.source.replaceAll('_', ' ')}</span>
                    <p>{item.reason}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

function Status({ label, value, tone }) {
  return (
    <div className={`status ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ComparisonRow({ label, value }) {
  return (
    <div className="comparison-row">
      <span>{label}</span>
      <strong>{formatMetric(value)}</strong>
    </div>
  );
}

function formatMetric(value) {
  if (value === undefined || value === null) return '-';
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue.toFixed(3) : '-';
}

function formatCount(value) {
  if (value === undefined || value === null) return '-';
  return Number(value).toLocaleString('en-US');
}

function parseMetrics(value) {
  if (!value) return null;
  if (typeof value === 'object') return value;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

export default App;

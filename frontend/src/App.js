// App.js
import React, { useState } from 'react';
import './App.css';

function App() {
  const [id, setId] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSearch = async () => {
    setError('');
    setResult(null);
    setSubmitted(true);

    try {
      const res = await fetch(`http://localhost:5000/get-result/${id.trim()}`);
      if (!res.ok) throw new Error('ID not found');

      const data = await res.json();
      setResult(data.PREDICTED_TARGET);
    } catch (err) {
      setError(err.message);
    }
  };

  if (!submitted) {
    return (
      <div className="landing-container">
        <nav className="navbar">
          <div className="logo">FICOFORCE</div>
          <div className="nav-links">
            <a href="#">HOME</a>
            <a href="#">LOGIN</a>
            <a href="#">PROFILE</a>
          </div>
        </nav>
        <div className="main-content">
          <div className="tagline">
            <h1>DETECT, TRACK<br />SECURE</h1>
            <input
              type="text"
              placeholder="Enter Account ID"
              value={id}
              onChange={(e) => setId(e.target.value)}
            />
            <button onClick={handleSearch}>SEARCH</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="result-page">
      <nav className="navbar">
        <div className="logo">FICOFORCE</div>
        <div className="nav-links">
          <a href="#">HOME</a>
          <a href="#">LOGIN</a>
          <a href="#">PROFILE</a>
        </div>
      </nav>
      <div className="result-content">
        {error && <p className="error">{error}</p>}
        {result === '0' && (
          <div className="result-box safe">✅ User is <strong>not</strong> a fraudster.</div>
        )}
        {result === '1' && (
          <div className="result-box fraud">⚠️ User is <strong>detected</strong> as a fraudster!</div>
        )}
      </div>
    </div>
  );
}

export default App;

const express = require('express');
const fs = require('fs');
const readline = require('readline');
const cors = require('cors');

const app = express();
const PORT = 5000;

app.use(cors());

let data = [];

// Read and parse CSV manually line by line
const rl = readline.createInterface({
  input: fs.createReadStream('./predicted_results_lgbm.csv'),
  crlfDelay: Infinity
});

let isFirstLine = true;

rl.on('line', (line) => {
  if (isFirstLine) {
    isFirstLine = false;
    return; // Skip header
  }

  const cleaned = line.replace(/"/g, ''); // Remove quotes
  const [UNIQUE_ID, TARGET, PREDICTED_TARGET] = cleaned.split(',');

  if (UNIQUE_ID && TARGET && PREDICTED_TARGET) {
    data.push({
      UNIQUE_ID: UNIQUE_ID.trim(),
      TARGET: TARGET.trim(),
      PREDICTED_TARGET: PREDICTED_TARGET.trim()
    });
  }
});

rl.on('close', () => {
  console.log('✅ CSV loaded. Records:', data.length);
});

// Root route
app.get('/', (req, res) => {
  res.send('✅ Backend running. Use /get-result/:id');
});

// Search by ID
app.get('/get-result/:id', (req, res) => {
  const id = req.params.id.trim();
  const result = data.find(row => row.UNIQUE_ID === id);

  if (result) {
    res.json(result);
  } else {
    res.status(404).json({ message: 'ID not found' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});

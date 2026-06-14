import { render, screen } from '@testing-library/react';
import App from './App';

test('renders FICOFORCE dashboard', () => {
  render(<App />);
  expect(screen.getByText(/FICOFORCE/i)).toBeInTheDocument();
  expect(screen.getByText(/ML Fraud Risk Detection/i)).toBeInTheDocument();
  expect(screen.getByText(/RAG Location Verification/i)).toBeInTheDocument();
});

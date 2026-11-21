import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/tokens.css';
import './styles/components.css';
import './styles/visuals.css';
import './styles/compare.css';
import App from './App.tsx';
import './styles/visual-compare.css';



createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

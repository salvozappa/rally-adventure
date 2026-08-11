import { Game } from './Game';

const overlay = document.createElement('div');
Object.assign(overlay.style, {
  position: 'fixed',
  inset: '0',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: '#0a0d10',
  color: '#c8b98a',
  font: 'bold 16px "Courier New", monospace',
  letterSpacing: '2px',
  zIndex: '10',
} as CSSStyleDeclaration);
overlay.textContent = 'LOADING…';
document.body.appendChild(overlay);

const game = new Game();

game
  .init()
  .then(() => {
    overlay.remove();
    game.start();
  })
  .catch((err) => {
    console.error(err);
    overlay.style.color = '#e05a3a';
    overlay.textContent = `FAILED TO START — ${err?.message ?? err}`;
  });

// Expose for console poking during development.
(globalThis as unknown as { game: Game }).game = game;

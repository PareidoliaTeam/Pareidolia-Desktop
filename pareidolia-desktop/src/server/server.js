/**
 * Headless Express server entry point
 * Run with: node src/server.js
 * 
 * This allows running the API without Electron overhead or display requirements.
 * All file system utilities are in storage.js (no Electron dependencies).
 */

import createServer from '../express.js';
import { ensurePareidoliaFolder } from '../storage.js';
import { setupPythonVenv } from '../python.js';

console.log('Starting Pareidolia Express server...');

try {
   // makes folder if it doesn't exist, and returns the path to it.
  ensurePareidoliaFolder();
  setupPythonVenv();

  const app = createServer();

  // Catch any unhandled errors
  process.on('unhandledRejection', (reason) => {
    console.error('Unhandled Rejection:', reason);
  });

  process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
  });
} catch (error) {
  console.error('Failed to start server:', error);
  process.exit(1);
}

/**
 * Unit tests for getDatasetsList and getModelsList in main.js
 *
 * All file system and Electron calls are mocked so no real files are touched.
 * These tests verify the logic of how folders are read and returned as JSON dictionaries.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// --- Mock Electron and its dependencies (must be before importing main.js) ---

vi.mock('electron', () => ({
  app: {
    quit: vi.fn(),
    whenReady: vi.fn(() => Promise.resolve()),
    on: vi.fn(),
  },
  BrowserWindow: vi.fn().mockImplementation(function () {
    this.loadURL = vi.fn();
    this.loadFile = vi.fn();
    this.webContents = { openDevTools: vi.fn() };
  }),
  ipcMain: { handle: vi.fn() },
}));

vi.mock('electron-squirrel-startup', () => ({ default: false }));

// Mock express.js to prevent the actual server from starting
vi.mock('../../src/express.js', () => ({ default: vi.fn() }));

// Mock python.js to prevent venv/script calls
vi.mock('../../src/python.js', () => ({
  getVenvPath: vi.fn(() => '/mock/venv'),
  setupPythonVenv: vi.fn(),
  executePythonScript: vi.fn(),
}));

// Mock python-shell (used via require() in main.js)
vi.mock('python-shell', () => ({
  PythonShell: { run: vi.fn() },
}));

// Mock the Node.js file system module
vi.mock('node:fs');

// Mock node:os so homedir always returns a predictable path
vi.mock('node:os', () => ({
  default: {
    homedir: vi.fn(() => '/Users/testuser'),
    networkInterfaces: vi.fn(() => ({
      en0: [{ family: 'IPv4', internal: false, address: '192.168.1.100' }],
    })),
  },
}));

// Now import functions under test (after all mocks are in place)
import { getDatasetsList, getModelsList } from '../../src/main.js';
import fs from 'node:fs';

// ============================================================
// getDatasetsList
// ============================================================

describe('getDatasetsList', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns an empty dictionary when the datasets folder does not exist', async () => {
    fs.existsSync.mockReturnValue(false);

    const result = await getDatasetsList();

    expect(result).toEqual({});
  });

  it('returns only directories, not files', async () => {
    fs.existsSync.mockReturnValue(true);
    // Simulate a folder that has two dataset dirs and one stray file
    fs.readdirSync.mockReturnValue(['cats', 'dogs', 'notes.txt']);
    fs.statSync.mockImplementation((p) => ({
      isDirectory: () => !p.endsWith('.txt'),
    }));

    const result = await getDatasetsList();

    expect(result).toHaveProperty('cats');
    expect(result).toHaveProperty('dogs');
    expect(result).not.toHaveProperty('notes.txt');
  });

  it('returns objects with correct path for each dataset', async () => {
    fs.existsSync.mockReturnValue(true);
    fs.readdirSync.mockReturnValue(['my-dataset']);
    fs.statSync.mockReturnValue({ isDirectory: () => true });

    const result = await getDatasetsList();

    expect(result['my-dataset']).toMatchObject({
      path: expect.stringContaining('my-dataset'),
    });
    // Path should be inside the PareidoliaApp/datasets directory
    expect(result['my-dataset'].path).toContain('datasets');
  });

  it('returns an empty dictionary when the datasets folder is empty', async () => {
    fs.existsSync.mockReturnValue(true);
    fs.readdirSync.mockReturnValue([]);

    const result = await getDatasetsList();

    expect(result).toEqual({});
  });

  it('throws when the file system returns an unexpected error', async () => {
    fs.existsSync.mockReturnValue(true);
    fs.readdirSync.mockImplementation(() => {
      throw new Error('Permission denied');
    });

    await expect(getDatasetsList()).rejects.toThrow('Permission denied');
  });
});

// ============================================================
// getModelsList
// ============================================================

describe('getModelsList', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns an empty dictionary when the models folder does not exist', async () => {
    fs.existsSync.mockReturnValue(false);

    const result = await getModelsList();

    expect(result).toEqual({});
  });

  it('returns only directories, not files', async () => {
    fs.existsSync.mockReturnValue(true);
    fs.readdirSync.mockReturnValue(['cat-detector', 'dog-detector', 'readme.md']);
    fs.statSync.mockImplementation((p) => ({
      isDirectory: () => !p.endsWith('.md'),
    }));

    const result = await getModelsList();

    expect(result).toHaveProperty('cat-detector');
    expect(result).toHaveProperty('dog-detector');
    expect(result).not.toHaveProperty('readme.md');
  });

  it('returns objects with correct path for each model', async () => {
    fs.existsSync.mockReturnValue(true);
    fs.readdirSync.mockReturnValue(['my-model']);
    fs.statSync.mockReturnValue({ isDirectory: () => true });

    const result = await getModelsList();

    expect(result['my-model']).toMatchObject({
      path: expect.stringContaining('my-model'),
    });
    // Path should be inside the PareidoliaApp/models directory
    expect(result['my-model'].path).toContain('models');
  });

  it('returns an empty dictionary when the models folder is empty', async () => {
    fs.existsSync.mockReturnValue(true);
    fs.readdirSync.mockReturnValue([]);

    const result = await getModelsList();

    expect(result).toEqual({});
  });

  it('throws when the file system returns an unexpected error', async () => {
    fs.existsSync.mockReturnValue(true);
    fs.readdirSync.mockImplementation(() => {
      throw new Error('Permission denied');
    });

    await expect(getModelsList()).rejects.toThrow('Permission denied');
  });
});

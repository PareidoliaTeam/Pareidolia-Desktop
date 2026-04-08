/*
  * Last modified by Alexangelo Orozco Gutierrez on 2-28-2026
  * Renamed functions to distinguish dataset and model creation. 
*/

import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'node:path';
import started from 'electron-squirrel-startup';
import fs from 'node:fs';
import os from 'node:os';
import { execSync } from 'node:child_process';
import createServer from './express.js';
import { getVenvPath, setupPythonVenv, executePythonScript } from './python.js';
import { getDatasetsList, getModelsList, createDatasetFolder, createModelFolder, getPareidoliaFolderPath, getLocalIP, getModelSettings, updateModelSettings, getProjectImages, modelDetailsForPython } from './storage.js';

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (started) {
  app.quit();
}

const createWindow = () => {
  // Create the browser window.
  if(process.env.HEADLESS) {
    return;
  }
  const mainWindow = new BrowserWindow({
    width: 1000,
    height: 800,
    autoHideMenuBar: true, // remove top bar
    webPreferences: {
      // MUST CHANGE LATER ONLY FOR TESTING
      webSecurity: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // and load the index.html of the app.
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`));
  }

  // Open the DevTools.
  // mainWindow.webContents.openDevTools();
};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  createServer();
  createWindow();

  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});



// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.



// ============================================
// FOLDER MANAGEMENT FUNCTIONS
// ============================================
// Note: All storage/file system utilities have been moved to storage.js
// See storage.js for: getDocumentsPath, getLocalIP, getPareidoliaFolderPath,
// ensurePareidoliaFolder, createDatasetFolder, createModelFolder, updateModelSettings,
// getModelSettings, modelDetailsForPython, getDatasetsList, getModelsList, getProjectImages



// ============================================
// ELECTRON-SPECIFIC FUNCTIONS
// ============================================

/**
 * Converts a video file into split images
 * @param {string} projectPath - the filepath to the project
 * @returns {string} path to video file
 */

const {dialog} = require("electron");
async function convertVideo(projectPath){
  // open a dialog window for the user to convert a video
  const result = await dialog.showOpenDialog({
    title: 'Select a video to convert',
    properties: ['openFile'],
    filters: [{ name: 'Videos', extensions: ['mp4', 'mov'] }],
  });

  if(result.canceled) {
    return null;
  } else {
    const videoPath = result.filePaths[0];
    const venvPath = getVenvPath();
    // run conversion
    console.log("Converting...")
    return await executePythonScript('py/extract_images.py', [
      videoPath,
      projectPath,
    ], venvPath);
  }
}
/**
 * Calls extract_images.py and exports images.
 * @param {string} videoPath - the filepath to the video
 * @param {string} projectPath - the filepath to the project
 */
const { PythonShell } = require('python-shell');
function runConversion(videoPath, projectPath){
  let options = {
    mode: 'text',
    pythonOptions: ['-u'],
    scriptPath: path.join(app.getAppPath(), 'py'),
    args: [videoPath, projectPath]
  };
  PythonShell.run('extract_images.py', options).then(messages =>{
    console.log('Conversion complete:', messages);
  });
}



// ============================================
// IPC HANDLERS
// ============================================

/**
 * Handle getting the datasets list via IPC from renderer process
 */
ipcMain.handle('get-datasets-list', async () => {
  return await getDatasetsList();
});

/**
 * Handle getting the models list via IPC from renderer process
 */
ipcMain.handle('get-models-list', async () => {
  return await getModelsList();
});

/**
 * Handle getting the local IP address via IPC from renderer process
 */
ipcMain.handle('get-local-ip', () => {
  return getLocalIP();
});

/**
 * Handle creating a dataset folder via IPC from renderer process
 */
ipcMain.handle('create-dataset-folder', async (event, projectName) => {
  return await createDatasetFolder(projectName);
});

/**
 * Handle getting the Pareidolia folder path via IPC from renderer process
 */
ipcMain.handle('get-pareidolia-path', async () => {
  return await ensurePareidoliaFolder();
});


/**
 * Handle creating a model folder via IPC from renderer process
 */
ipcMain.handle('create-model-folder', async (event, modelName) => {
  return await createModelFolder(modelName);
});


// ============================================
// PYTHON EXECUTION HANDLERS
// ============================================

/**
 * IPC handler to setup Python virtual environment
 */
ipcMain.handle('setup-python-venv', async () => {
  return await setupPythonVenv();
});

/**
 * Handle training model execution via IPC from renderer process.
 * @param {Object} event - IPC event
 * @param {Object} params - Training parameters
 * @param {Object} params.labelsJson  - Object mapping label names to arrays of folder paths
 *                                      e.g. { "Apple": ["/path/folder1"], "Orange": ["/path/a", "/path/b"] }
 * @param {string} params.modelFolderPath - Path to the model folder where outputs will be saved
 * @param {number} params.epochs      - Number of training epochs
 */
ipcMain.handle('execute-train', async (event, params) => {
  const { labelsJson, modelFolderPath, epochs } = params;

  // Validate labelsJson
  if (!labelsJson || typeof labelsJson !== 'object' || Object.keys(labelsJson).length === 0) {
    return {
      success: false,
      error: 'labelsJson must be a non-empty object mapping label names to arrays of folder paths.',
      timestamp: new Date().toISOString()
    };
  }

  // Validate that every folder path referenced in labelsJson actually exists
  for (const [label, folders] of Object.entries(labelsJson)) {
    for (const folder of folders) {
      if (!fs.existsSync(folder)) {
        return {
          success: false,
          error: `Folder not found for label "${label}": ${folder}`,
          timestamp: new Date().toISOString()
        };
      }
    }
  }

  // Ensure the model output directory exists
  if (!fs.existsSync(modelFolderPath)) {
    fs.mkdirSync(modelFolderPath, { recursive: true });
  }

  const modelPath = modelFolderPath;

  // Determine the correct Python script path (dev vs production)
  let pythonScriptPath;
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    pythonScriptPath = path.join(__dirname, '../../py/train_model.py');
  } else {
    pythonScriptPath = path.join(process.resourcesPath, 'py/train_model.py');
  }

  console.log('Python script path:', pythonScriptPath);
  console.log('Labels JSON:', labelsJson);
  console.log('Model path:', modelPath);
  console.log('Epochs:', epochs);

  // Pass labels_json, model_path, and epochs as arguments
  const venvPath = getVenvPath();
  return await executePythonScript(pythonScriptPath, [
    JSON.stringify(labelsJson),
    modelPath,
    epochs.toString()
  ], venvPath);
});
/**
 * Handle getting the images in a selected project
 */
ipcMain.handle('get-project-images', async (event, projectPath) => {
  return await getProjectImages(projectPath);
});
/**
 * Handle converting a video file to an image
 */
ipcMain.handle('convert-video', async (event, projectPath) => {
  return await convertVideo(projectPath);
});

/**
 * Handle getting model details for Python training via IPC from renderer process.
 * Returns the pre-built labelsJson, model output folder path, and full settings.
 */
ipcMain.handle('get-model-details-for-python', async (event, modelName) => {
  return await modelDetailsForPython(modelName);
});

/**
 * Handle getting model settings via IPC from renderer process
 */
ipcMain.handle('get-model-settings', async (event, modelName) => {
  return await getModelSettings(modelName);
});

/**
 * Handle updating model settings via IPC from renderer process
 */
ipcMain.handle('update-model-settings', async (event, params) => {
  const { modelName, newSettings } = params;
  return await updateModelSettings(modelName, newSettings);
});
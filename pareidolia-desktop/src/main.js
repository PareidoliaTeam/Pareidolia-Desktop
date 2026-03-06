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

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (started) {
  app.quit();
}

const createWindow = () => {
  // Create the browser window.
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



/**
 * Detects the user's operating system and returns the Documents folder path.
 * @returns {string} The full path to the user's Documents folder
 */
function getDocumentsPath() {
  const userHome = os.homedir();
  const platform = process.platform;

  if (platform === 'win32') {
    // Windows
    return path.join(userHome, 'Documents');
  } else if (platform === 'darwin') {
    // macOS
    return path.join(userHome, 'Documents');
  } else {
    // Linux and others
    return path.join(userHome, 'Documents');
  }
}

/**
 * Gets the local IP address of the machine.
 * Returns the first non-loopback IPv4 address found.
 * @returns {string|null} The local IP address or null if not found
 */
export function getLocalIP() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      // Skip internal and non-IPv4 interfaces
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return null;
}

/**
 * Gets the Pareidolia projects folder path based on the current OS.
 * @returns {string} The full path to the PareidoliaApp folder
 */
export function getPareidoliaFolderPath() {
  const documentsPath = getDocumentsPath();
  return path.join(documentsPath, 'PareidoliaApp');
}

/**
 * Creates the Pareidolia folder if it doesn't already exist.
 * Also creates datasets and models subdirectories.
 * @returns {Promise<string>} The path to the created or existing Pareidolia folder
 */
async function ensurePareidoliaFolder() {
  const pareidoliaPath = getPareidoliaFolderPath();

  try {
    if (!fs.existsSync(pareidoliaPath)) {
      fs.mkdirSync(pareidoliaPath, { recursive: true });
      console.log(`Created Pareidolia folder at: ${pareidoliaPath}`);
    } else {
      console.log(`Pareidolia folder already exists at: ${pareidoliaPath}`);
    }

    // Create datasets folder
    const datasetsPath = path.join(pareidoliaPath, 'datasets');
    if (!fs.existsSync(datasetsPath)) {
      fs.mkdirSync(datasetsPath, { recursive: true });
      console.log(`Created datasets folder at: ${datasetsPath}`);
    }

    // Create models folder
    const modelsPath = path.join(pareidoliaPath, 'models');
    if (!fs.existsSync(modelsPath)) {
      fs.mkdirSync(modelsPath, { recursive: true });
      console.log(`Created models folder at: ${modelsPath}`);
    }

    return pareidoliaPath;
  } catch (error) {
    console.error(`Error creating Pareidolia folder: ${error.message}`);
    throw error;
  }
}

/**
 * Creates a dataset folder inside the datasets folder within Pareidolia.
 * @param {string} projectName - The name of the dataset folder to create
 * @returns {Promise<string>} The full path to the created dataset folder
 */
export async function createDatasetFolder(projectName) {
  try {
    const pareidoliaPath = await ensurePareidoliaFolder();
    const datasetsPath = path.join(pareidoliaPath, 'datasets');
    const projectPath = path.join(datasetsPath, projectName);

    if (!fs.existsSync(projectPath)) {
      fs.mkdirSync(projectPath, { recursive: true });
      console.log(`Created project folder at: ${projectPath}`);
    } else {
      console.log(`Project folder already exists at: ${projectPath}`);
    }

    // Create positives and negatives folders
    const positivesPath = path.join(projectPath, 'positives');
    const negativesPath = path.join(projectPath, 'negatives');

    if (!fs.existsSync(positivesPath)) {
      fs.mkdirSync(positivesPath, { recursive: true });
      console.log(`Created positives folder at: ${positivesPath}`);
    }

    if (!fs.existsSync(negativesPath)) {
      fs.mkdirSync(negativesPath, { recursive: true });
      console.log(`Created negatives folder at: ${negativesPath}`);
    }
    return projectPath;
  } catch (error) {
    console.error(`Error creating project folder: ${error.message}`);
    throw error;
  }
}

/**
 * Creates a model folder inside the models folder within Pareidolia.
 * Also creates a model-settings.json file with initial configuration.
 * @param {string} modelName - The name of the model folder to create
 * @returns {Promise<string>} The full path to the created model folder
 */
export async function createModelFolder(modelName) {
  try {
    const pareidoliaPath = await ensurePareidoliaFolder();
    const modelsPath = path.join(pareidoliaPath, 'models');
    const modelPath = path.join(modelsPath, modelName);

    if (!fs.existsSync(modelPath)) {
      fs.mkdirSync(modelPath, { recursive: true });
      console.log(`Created model folder at: ${modelPath}`);
    } else {
      console.log(`Model folder already exists at: ${modelPath}`);
    }

    // Create models subfolder where trained models are saved
    const modelsSubfolderPath = path.join(modelPath, 'models');
    if (!fs.existsSync(modelsSubfolderPath)) {
      fs.mkdirSync(modelsSubfolderPath, { recursive: true });
      console.log(`Created models subfolder at: ${modelsSubfolderPath}`);
    }

    // Create model-settings.json file
    const settingsPath = path.join(modelPath, 'model-settings.json');
    const defaultSettings = {
      labels: {},
      epochs: 10
    };

    if (!fs.existsSync(settingsPath)) {
      fs.writeFileSync(settingsPath, JSON.stringify(defaultSettings, null, 2));
      console.log(`Created model settings file at: ${settingsPath}`);
    } else {
      console.log(`Model settings file already exists at: ${settingsPath}`);
    }

    return modelPath;
  } catch (error) {
    console.error(`Error creating model folder: ${error.message}`);
    throw error;
  }
}

/**
 * To be used for the training page to update the model-settings.json file with new settings after setting labels, datasets, and epochs.
 * Example changes could be new labels added so the labels dictionary needs to add its new keys and updating the values of keys with the paths of
 * the datasets used.
 * @param {string} modelName - name of the model to focus on, needed to find the model-settings.json file
 * @param {json} newSettings updated json of the settings to overwrite the file with
 */
export async function updateModelSettings(modelName, newSettings) {
  try {
    const pareidoliaPath = getPareidoliaFolderPath();
    const settingsPath = path.join(pareidoliaPath, 'models', modelName, 'model-settings.json');

    if (!fs.existsSync(settingsPath)) {
      throw new Error(`Model settings file not found for model: ${modelName}`);
    }

    fs.writeFileSync(settingsPath, JSON.stringify(newSettings, null, 2));
    console.log(`Updated model settings for ${modelName} at: ${settingsPath}`);
  } catch (error) {
    console.error(`Error updating model settings: ${error.message}`);
    throw error;
  }
}

/**
 * Gets the model settings from the model-settings.json file for a given model.
 * @param {string} modelName - name of the model to focus on, needed to find the model-settings.json file
 * @returns {json} settings - the parsed JSON content of the model-settings.json file
 */
export async function getModelSettings(modelName) {
  try {
    const pareidoliaPath = getPareidoliaFolderPath();
    const settingsPath = path.join(pareidoliaPath, 'models', modelName, 'model-settings.json');
    
    if (!fs.existsSync(settingsPath)) {
      throw new Error(`Model settings file not found for model: ${modelName}`);
    }
    const settings = fs.readFileSync(settingsPath, 'utf-8');
    return JSON.parse(settings);
  } catch (error) {
    console.error(`Error getting model settings: ${error.message}`);
    throw error;
  }
}

/**
 * REQUIRES PROPER IMPLEMENTATION
 * Will create a JSON string to pass to the Python training script with the model path and settings for the specified model. 
 * This is needed because the training script needs to know where to save the trained model and what settings to train with.
 * It is easiest to create a string to pass in rather then to give it a file path
 * @param {string} modelName - name of the model to focus on, needed to find the model folder and settings  
 * @returns {string} An object containing the model path and settings for the specified model, to be used for training in Python. Example: { modelPath: '/path/to/model/folder', settings: { labels: {}, epochs: 10 } }
 */
export async function modelDetailsForPython(modelName) {
  try {
    const pareidoliaPath = getPareidoliaFolderPath();
    const modelPath = path.join(pareidoliaPath, 'models', modelName);
    const settings = await getModelSettings(modelName);
    
    return {
      modelPath,
      settings
    };
  } catch (error) {
    console.error(`Error getting model details for Python: ${error.message}`);
    throw error;
  }
}

/**
 * Gets a list of all project folders in the datasets folder within Pareidolia.
 * @returns {Promise<Array>} Array of objects with name and path properties
 */
export async function getDatasetsList() {
  try {
    const pareidoliaPath = getPareidoliaFolderPath();
    const datasetsPath = path.join(pareidoliaPath, 'datasets');
    
    if (!fs.existsSync(datasetsPath)) {
      return {};
    }

    const files = fs.readdirSync(datasetsPath);
    const datasets = {};

    for (const file of files) {
      const filePath = path.join(datasetsPath, file);
      const stats = fs.statSync(filePath);

      // Only include directories
      if (stats.isDirectory()) {
        datasets[file] = {
          path: filePath
        };
      }
    }

    return datasets;
  } catch (error) {
    console.error(`Error getting projects list: ${error.message}`);
    throw error;
  }
}

/**
 * Gets a list of all model folders in the models folder within Pareidolia.
 * @returns {Promise<Array>} Array of objects with name and path properties
 */
export async function getModelsList() {
  try {
    const pareidoliaPath = getPareidoliaFolderPath();
    const modelsPath = path.join(pareidoliaPath, 'models');
    
    if (!fs.existsSync(modelsPath)) {
      return {};
    }

    const files = fs.readdirSync(modelsPath);
    const models = {};

    for (const file of files) {
      const filePath = path.join(modelsPath, file);
      const stats = fs.statSync(filePath);
      
      // Only include directories
      if (stats.isDirectory()) {
        models[file] = {
          path: filePath
        };
      }
    }

    return models;
  } catch (error) {
    console.error(`Error getting models list: ${error.message}`);
    throw error;
  }
}
/**
 * Gets all images in a selected project.
 * @param {string} projectPath - the filepath to the project folder
 * @returns {string<Array>} images - an array of urls to specified images
 */
export async function getProjectImages(projectPath) {
  try {
    // Read all files in path
    const files = fs.readdirSync(projectPath);

    // Filter for only images
    const imageExtensions = ['.jpg', '.jpeg', '.png'];
    const images = files.filter(file => imageExtensions.includes(path.extname(file).toLowerCase())).map(file=> {
      // Return an object
      return {
        name: file,
        url: `file://${path.join(projectPath, file)}`
      };
    });

    return images;
  } catch(error) {
    console.error("Failed to read directory:", error);
    return [];
  }
}
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

  const modelPath = path.join(modelFolderPath, 'model.keras');

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
})
/**
 * Storage and file system utilities
 * No Electron dependencies - safe to use in Node and Electron contexts
 */

import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';

/**
 * Detects the user's operating system and returns the Documents folder path.
 * @returns {string} The full path to the user's Documents folder
 */
function getDocumentsPath() {
  console.log("Detecting Documents path for platform:", process.platform);
  const userHome = os.homedir();
  const platform = process.platform;
  console.log("User home directory:", userHome);
  console.log("Platform:", platform);

  if (platform === 'win32') {
    // Windows
    console.log("Usng Windows");
    return path.join(userHome, 'Documents');
  } else if (platform === 'darwin') {
    // macOS
    console.log("Usng macOS");
    return path.join(userHome, 'Documents');
  } else {
    // Linux and others
    // more complex, checks if it is a Desktop version and if not creates its own folder.
    console.log("Usng Linux/Other");
    //if Documents exists, treat like other platforms.
    if (fs.existsSync(path.join(userHome, 'Documents'))) {
      return path.join(userHome, 'Documents');
    } else {
      // returns userHome, so PareidoliaApp just appears in the home directory.
      return userHome;
    }
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
export async function ensurePareidoliaFolder() {
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
 * 
 * Example output: 
 * { name: "Fruits",
 *  labels: 
 *  { "Apple": 
 *    { "Fuji Apples": "/path/folder1" , 
 *     "Gala Apples": "/path/folder2" }, 
 *    "Orange": 
 *      { "Navel Oranges": "/path/a", 
 *      "Blood Oranges": "/path/b" } 
 *  }, 
 *  epochs: 10 
 * }
 * Labels is a key with a value of a dictionary with the keys Labels.
 * Labels hav values of dictionaries with the Dataset name as keys and their paths as values. 
 * This way we can have multiple datasets for each label.
 * 
 * The current structure helps organize data while also makign future additions possible, such as storing settings.
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
 * Builds the training details to pass to the Python training script.
 * Constructs labelsJson ({ LabelName: [path1, path2, ...] }) from model settings
 * and resolves the model output folder path.
 * @param {string} modelName - name of the model to focus on
 * @returns {Object} { modelFolderPath, labelsJson, settings }
 *   modelFolderPath - path to the models subfolder where trained outputs are saved
 *   labelsJson      - flat label-to-paths map ready for the Python script
 *   settings        - full model-settings.json contents
 */
export async function modelDetailsForPython(modelName) {
  try {
    const pareidoliaPath = getPareidoliaFolderPath();
    const modelPath = path.join(pareidoliaPath, 'models', modelName);
    const settings = await getModelSettings(modelName);

    const labelsJson = {};
    for (const [labelName, datasets] of Object.entries(settings.labels || {})) {
      // Each dataset folder contains a positives/ subfolder with the training images
      labelsJson[labelName] = Object.values(datasets).map(p => path.join(p, 'positives'));
    }
    
    return {
      modelFolderPath: path.join(modelPath, 'models'),
      labelsJson,
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
    console.error(`Error getting datasets list: ${error.message}`);
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
      const modelSettingsPath = path.join(filePath, 'model-settings.json');
      const modelSettings = fs.existsSync(modelSettingsPath) ? JSON.parse(fs.readFileSync(modelSettingsPath, 'utf-8')) : null;

      // Only include directories
      if (stats.isDirectory()) {
        if (modelSettings) {
          models[file] = {
            path: filePath,
            labels: modelSettings.labels || {}
          };
        }
        else{
          models[file] = {
            path: filePath
          };
        }
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

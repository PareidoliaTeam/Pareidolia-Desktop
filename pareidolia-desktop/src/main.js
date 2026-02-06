import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'node:path';
import started from 'electron-squirrel-startup';
import fs from 'node:fs';
import os from 'node:os';

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
 * Gets the Pareidolia projects folder path based on the current OS.
 * @returns {string} The full path to the PareidoliaApp folder
 */
function getPareidoliaFolderPath() {
  const documentsPath = getDocumentsPath();
  return path.join(documentsPath, 'PareidoliaApp');
}

/**
 * Creates the Pareidolia folder if it doesn't already exist.
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
    return pareidoliaPath;
  } catch (error) {
    console.error(`Error creating Pareidolia folder: ${error.message}`);
    throw error;
  }
}

/**
 * Creates a project folder inside the Pareidolia folder.
 * @param {string} projectName - The name of the project folder to create
 * @returns {Promise<string>} The full path to the created project folder
 */
async function createProjectFolder(projectName) {
  try {
    const pareidoliaPath = await ensurePareidoliaFolder();
    const projectPath = path.join(pareidoliaPath, projectName);

    if (!fs.existsSync(projectPath)) {
      fs.mkdirSync(projectPath, { recursive: true });
      console.log(`Created project folder at: ${projectPath}`);
    } else {
      console.log(`Project folder already exists at: ${projectPath}`);
    }
    return projectPath;
  } catch (error) {
    console.error(`Error creating project folder: ${error.message}`);
    throw error;
  }
}

/**
 * Gets a list of all project folders in the Pareidolia folder.
 * @returns {Promise<Array>} Array of objects with name and path properties
 */
async function getProjectsList() {
  try {
    const pareidoliaPath = getPareidoliaFolderPath();
    
    if (!fs.existsSync(pareidoliaPath)) {
      return [];
    }

    const files = fs.readdirSync(pareidoliaPath);
    const projects = [];

    for (const file of files) {
      const filePath = path.join(pareidoliaPath, file);
      const stats = fs.statSync(filePath);
      
      // Only include directories
      if (stats.isDirectory()) {
        projects.push({
          name: file,
          path: filePath
        });
      }
    }

    return projects;
  } catch (error) {
    console.error(`Error getting projects list: ${error.message}`);
    throw error;
  }
}
/**
 * Gets all images in a selected project.
 * @param {string} projectPath - the filepath to the project folder
 * @returns
 */
async function getProjectImages(projectPath) {
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


// ============================================
// IPC HANDLERS
// ============================================

/**
 * Handle getting the projects list via IPC from renderer process
 */
ipcMain.handle('get-projects-list', async () => {
  return await getProjectsList();
});

/**
 * Handle creating a project folder via IPC from renderer process
 */
ipcMain.handle('create-project-folder', async (event, projectName) => {
  return await createProjectFolder(projectName);
});

/**
 * Handle getting the Pareidolia folder path via IPC from renderer process
 */
ipcMain.handle('get-pareidolia-path', async () => {
  return await ensurePareidoliaFolder();
});
/**
 * Handle getting the images in a selected project
 */
ipcMain.handle('get-project-images', async (event, projectPath) => {
  return await getProjectImages(projectPath);
});

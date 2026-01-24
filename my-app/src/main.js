import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import started from 'electron-squirrel-startup';

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (started) {
  app.quit();
}

const createWindow = () => {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
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
  mainWindow.webContents.openDevTools();
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

// IPC Handler for creating projects
ipcMain.handle('create-project', async (event, projectName) => {
  try {
    // Get user's documents folder
    const userDocuments = app.getPath('documents');
    
    // Create Projects folder path
    const projectsFolder = path.join(userDocuments, 'Pareidolia', 'Projects');
    
    // Create the Projects folder if it doesn't exist
    if (!fs.existsSync(projectsFolder)) {
      fs.mkdirSync(projectsFolder, { recursive: true });
    }
    
    // Create project-specific folder
    const projectFolder = path.join(projectsFolder, projectName);
    
    // Check if project already exists
    if (fs.existsSync(projectFolder)) {
      return { success: false, error: 'Project already exists' };
    }
    
    // Create project folder
    fs.mkdirSync(projectFolder, { recursive: true });
    
    // Create project JSON file
    const projectData = {
      name: projectName,
      directory: projectFolder,
      createdAt: new Date().toISOString(),
      images: [],
      settings: {}
    };
    
    const jsonFilePath = path.join(projectFolder, `${projectName}.json`);
    fs.writeFileSync(jsonFilePath, JSON.stringify(projectData, null, 2));
    
    return { 
      success: true, 
      directory: projectFolder,
      jsonPath: jsonFilePath
    };
  } catch (error) {
    console.error('Error creating project:', error);
    return { success: false, error: error.message };
  }
});

// IPC Handler for loading existing projects
ipcMain.handle('load-projects', async (event) => {
  try {
    // Get user's documents folder
    const userDocuments = app.getPath('documents');
    
    // Create Projects folder path
    const projectsFolder = path.join(userDocuments, 'Pareidolia', 'Projects');
    
    // Create the Projects folder if it doesn't exist
    if (!fs.existsSync(projectsFolder)) {
      fs.mkdirSync(projectsFolder, { recursive: true });
      return { success: true, projects: [] };
    }
    
    // Read all folders in the Projects directory
    const folders = fs.readdirSync(projectsFolder, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name);
    
    // Load project data from JSON files
    const projects = [];
    for (const folderName of folders) {
      const projectFolder = path.join(projectsFolder, folderName);
      const jsonFilePath = path.join(projectFolder, `${folderName}.json`);
      
      if (fs.existsSync(jsonFilePath)) {
        try {
          const jsonData = fs.readFileSync(jsonFilePath, 'utf8');
          const projectData = JSON.parse(jsonData);
          projects.push(projectData);
        } catch (error) {
          console.error(`Error reading project ${folderName}:`, error);
          // If JSON doesn't exist or is invalid, create basic project data
          projects.push({
            name: folderName,
            directory: projectFolder
          });
        }
      } else {
        // If no JSON file, create basic project data
        projects.push({
          name: folderName,
          directory: projectFolder
        });
      }
    }
    
    return { success: true, projects };
  } catch (error) {
    console.error('Error loading projects:', error);
    return { success: false, error: error.message, projects: [] };
  }
});

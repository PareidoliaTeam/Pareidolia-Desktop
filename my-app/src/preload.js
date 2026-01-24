// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  createProject: (projectName) => ipcRenderer.invoke('create-project', projectName),
  loadProjects: () => ipcRenderer.invoke('load-projects')
});

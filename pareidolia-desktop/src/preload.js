// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts
import { contextBridge, ipcRenderer,webUtils } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  invoke: (channel, args) => ipcRenderer.invoke(channel, args),
  // future IPC handlers will go here in order for the web page to call Electron functions
  executeTrain: (epochs) => ipcRenderer.invoke('execute-train', epochs),

  onTrainingStdout: (callback) => ipcRenderer.on('training-stdout', (event, value) => callback(value)),
  onTrainingStderr: (callback) => ipcRenderer.on('training-stderr', (event, value) => callback(value)),

  getPathForFile: (file) => webUtils.getPathForFile(file)
});

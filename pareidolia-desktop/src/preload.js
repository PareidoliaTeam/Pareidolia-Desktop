// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts
import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  invoke: (channel, args) => ipcRenderer.invoke(channel, args),
  // future IPC handlers will go here in order for the web page to call Electron functions
  executeTrain: (epochs) => ipcRenderer.invoke('execute-train', epochs),
});

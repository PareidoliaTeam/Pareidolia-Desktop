/*
 * Python execution utilities
 * Handles virtual environment setup and Python script execution
 */

import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { spawn } from 'node:child_process';
import { getPareidoliaFolderPath } from './main.js';

/**
 * Gets the path to the Python venv folder.
 * Used in setupPythonVenv and executePythonScript functions to ensure the correct Python environment is used.
 * @returns {string} The full path to the venv folder
 */
export function getVenvPath() {
  const pareidoliaPath = getPareidoliaFolderPath();
  return path.join(pareidoliaPath, 'venv');
}

/**
 * Gets the path to the Python executable in the venv.
 * Used in setupPythonVenv and executePythonScript functions to ensure the correct Python environment is used.
 * @returns {string} The full path to the Python executable
 */
export function getVenvPythonExecutable() {
  const venvPath = getVenvPath();
  const platform = process.platform;

  if (platform === 'win32') {
    // Windows
    return path.join(venvPath, 'Scripts', 'python.exe');
  } else {
    // macOS and Linux
    return path.join(venvPath, 'bin', 'python');
  }
}

/**
 * Checks if a Python virtual environment exists, 
 * creates it if it doesn't, and installs required packages.
 * Runs in the background and does not affect the UI.
 * @returns {Promise<Object>} Object with success flag, path, and message
 */
export function setupPythonVenv() {
  return new Promise((resolve) => {
    const venvPath = getVenvPath();
    
    // Check if venv already exists
    if (fs.existsSync(venvPath)) {
      console.log('Virtual environment already exists at:', venvPath);
      resolve({
        success: true,
        venvPath: venvPath,
        pythonExecutable: getVenvPythonExecutable(),
        message: 'Virtual environment already exists'
      });
      //returns early if venv already exists to avoid unnecessary setup time on every startup
      return;
    }

    //starts venv creation process if venv does not exist

    console.log('Creating virtual environment at:', venvPath);
    
    // Create venv using Python 3.11
    const platform = process.platform;
    const pythonCommand = platform === 'win32' ? 'py' : 'python3.11';
    const pythonArgs = platform === 'win32' ? ['-3.11', '-m', 'venv', venvPath] : ['-m', 'venv', venvPath];
    const createVenvProcess = spawn(pythonCommand, pythonArgs);
    
    let venvError = '';
    
    createVenvProcess.stderr.on('data', (data) => {
      venvError += data.toString();
    });
    
    createVenvProcess.on('close', (code) => {
      if (code === 0) {
        console.log('Virtual environment created successfully');
        
        // Install required packages
        const pythonExecutable = getVenvPythonExecutable();
        const packages = ['tensorflow', 'opencv-python', 'numpy'];
        
        console.log('Installing required packages:', packages);
        
        const pipProcess = spawn(pythonExecutable, ['-m', 'pip', 'install', ...packages]);
        
        // Capture stdout from pip process and log it
        let pipOutput = '';
        pipProcess.stdout.on('data', (data) => {
          pipOutput += data.toString();
          console.log('pip:', data.toString());
        });
        
        // Capture stderr from pip process and log any errors
        let pipError = '';
        pipProcess.stderr.on('data', (data) => {
          pipError += data.toString();
          console.log('pip error:', data.toString());
        });
        
        pipProcess.on('close', (pipCode) => {
          if (pipCode === 0) {
            console.log('Packages installed successfully');
            resolve({
              success: true,
              venvPath: venvPath,
              pythonExecutable: pythonExecutable,
              message: 'Virtual environment created and packages installed successfully'
            });
          } else {
            console.error('Error installing packages:', pipError);
            resolve({
              success: false,
              error: pipError || `pip exited with code ${pipCode}`,
              venvPath: venvPath,
              pythonExecutable: pythonExecutable
            });
          }
        });
        
        pipProcess.on('error', (err) => {
          console.error('Error running pip:', err);
          resolve({
            success: false,
            error: err.message,
            venvPath: venvPath,
            pythonExecutable: pythonExecutable
          });
        });
      } else {
        console.error('Error creating virtual environment:', venvError);
        resolve({
          success: false,
          error: venvError || `venv creation exited with code ${code}`
        });
      }
    });
    
    createVenvProcess.on('error', (err) => {
      console.error('Error running python -m venv:', err);
      resolve({
        success: false,
        error: err.message
      });
    });
  });
}

/**
 * Executes a Python script and returns the output.
 * @param {string} pythonPath - The path to the Python script
 * @param {Array} args - Command line arguments to pass to the script
 * @param {string} venvPath - Optional path to the venv folder
 * @returns {Promise<Object>} Object with success flag, output/error, and timestamp
 * 
 * This specific function will be used in any IPC handler involving a python script.
 * However, it is encouraged to make your own IPC handler and use this function within it instead of a
 * general purpose IPC handler for all Python scripts due to specific needs on the images and train pages.
 */
export function executePythonScript(pythonPath, args = [], venvPath = null) {
  return new Promise((resolve) => {
    // Determine which Python executable to use
    let pythonExecutable = 'python3';
    if (venvPath) {
      pythonExecutable = getVenvPythonExecutable();
    }
    
    console.log('[Python] Starting script:', pythonPath);
    console.log('[Python] Arguments:', args);
    console.log('[Python] Using executable:', pythonExecutable);
    console.log('[Python] Timestamp:', new Date().toISOString());
    
    const startTime = Date.now();
    const pythonProcess = spawn(pythonExecutable, [pythonPath, ...args]);
    
    let output = '';
    let error = '';
    let lastStatusTime = startTime;
    
    console.log('[Python] Process spawned with PID:', pythonProcess.pid);
    
    pythonProcess.stdout.on('data', (data) => {
      const dataStr = data.toString();
      output += dataStr;
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      lastStatusTime = Date.now();
      console.log(`[Python] [${elapsed}s] STDOUT:`, dataStr);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      const dataStr = data.toString();
      error += dataStr;
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      lastStatusTime = Date.now();
      console.log(`[Python] [${elapsed}s] STDERR:`, dataStr);
    });
    
    pythonProcess.on('close', (code) => {
      const totalTime = Math.round((Date.now() - startTime) / 1000);
      console.log(`[Python] Process closed with code: ${code} (Total time: ${totalTime}s)`);
      
      if (code === 0) {
        console.log('[Python] Execution successful');
        resolve({
          success: true,
          output: output.trim(),
          timestamp: new Date().toISOString(),
          executionTime: totalTime
        });
      } else {
        console.error(`[Python] Execution failed with code ${code}`);
        resolve({
          success: false,
          error: error || `Process exited with code ${code}`,
          timestamp: new Date().toISOString(),
          executionTime: totalTime
        });
      }
    });
    
    pythonProcess.on('error', (err) => {
      const totalTime = Math.round((Date.now() - startTime) / 1000);
      console.error('[Python] Process error:', err.message);
      resolve({
        success: false,
        error: err.message,
        timestamp: new Date().toISOString(),
        executionTime: totalTime
      });
    });
  });
}

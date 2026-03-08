/*
 * Created by Aleaxngelo Orozco Gutierrez on 2-10-2026
 * The JavaSript File handling all the of Train Page interactions
 * 
 * Currently gives the Train button functionality and calls IPC handler to execute the Python training script
 */

// Get project name and path from sessionStorage
const projectName = sessionStorage.getItem('projectName') || 'Project';
const projectPath = sessionStorage.getItem('projectPath') || 'No path';

// Display project name and path in navbar
// Will likely be changed later on as the train page will become seperate from individual projects.
if (document.getElementById('navbar-name')) {
    document.getElementById('navbar-name').textContent = projectName;
    document.getElementById('navbar-path').textContent = projectPath;
}

// Get elements
const epochSlider = document.getElementById('epoch-slider');
const epochDisplay = document.getElementById('epoch-display');
const trainBtn = document.getElementById('train-btn');
const labelsList = document.getElementById('labels-list');
const newLabelInput = document.getElementById('new-label-input');
const addLabelBtn = document.getElementById('add-label-btn');

// Dataset modal elements
const datasetModal = document.getElementById('dataset-modal');
const datasetModalTitle = document.getElementById('dataset-modal-title');
const datasetModalClose = document.getElementById('dataset-modal-close');
const assignedDatasetsList = document.getElementById('assigned-datasets-list');
const availableDatasetsList = document.getElementById('available-datasets-list');

// Tracks which label the dataset modal is currently open for
let currentLabelName = null;

// Holds the full model-settings.json structure: { labels: {}, epochs: 10 }
let modelSettings = null;

console.log('Project Name:', projectName);
console.log('Project Path:', projectPath);

// Update epoch display when slider changes (live)
epochSlider.addEventListener('input', () => {
    epochDisplay.textContent = epochSlider.value;
});

// Save epochs when user finishes dragging the slider
epochSlider.addEventListener('change', async () => {
    if (!modelSettings) return;
    modelSettings.epochs = parseInt(epochSlider.value);
    await saveSettings();
});

// ============================================================
// LABEL MANAGEMENT
// ============================================================

/**
 * Renders the labels list by creating label items with remove buttons
 */
function renderLabels() {
    labelsList.innerHTML = '';
    
    const labelKeys = Object.keys(modelSettings?.labels || {});
    if (labelKeys.length === 0) {
        labelsList.innerHTML = '<div style="text-align: center; color: #999; padding: 0.5rem;">No labels added</div>';
        return;
    }
    
    labelKeys.forEach((labelName) => {
        const labelItem = document.createElement('div');
        labelItem.classList.add('label-item');
        labelItem.title = 'Click to assign datasets';
        labelItem.addEventListener('click', () => openDatasetModal(labelName));
        
        const labelNameSpan = document.createElement('span');
        labelNameSpan.classList.add('label-item-name');
        labelNameSpan.textContent = labelName;
        
        const removeBtn = document.createElement('button');
        removeBtn.classList.add('remove-label-btn');
        removeBtn.textContent = '−';
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            removeLabel(labelName);
        });
        
        labelItem.appendChild(labelNameSpan);
        labelItem.appendChild(removeBtn);
        labelsList.appendChild(labelItem);
    });
}

/**
 * Adds a new label to the list
 * @param {string} labelName - The name of the label to add
 */
function addLabel(labelName) {
    const trimmedName = labelName.trim();
    
    if (!trimmedName) {
        alert('Label name cannot be empty');
        return;
    }
    
    if (trimmedName in modelSettings.labels) {
        alert('Label already exists');
        return;
    }
    
    modelSettings.labels[trimmedName] = {};
    renderLabels();
    newLabelInput.value = '';
    newLabelInput.focus();
    saveSettings();
}

/**
 * Removes a label from the list
 * @param {string} labelName - The name of the label to remove
 */
function removeLabel(labelName) {
    delete modelSettings.labels[labelName];
    renderLabels();
    saveSettings();
}

/**
 * Persists the full modelSettings object back to model-settings.json via IPC
 */
async function saveSettings() {
    if (!modelSettings) return;
    try {
        await window.electronAPI.invoke('update-model-settings', { modelName: projectName, newSettings: modelSettings });
        console.log('Model settings saved');
    } catch (error) {
        console.error('Error saving model settings:', error);
    }
}

// ============================================================
// DATASET MODAL
// ============================================================

/**
 * Opens the dataset assignment modal for the given label.
 * @param {string} labelName
 */
async function openDatasetModal(labelName) {
    currentLabelName = labelName;
    datasetModalTitle.textContent = `Datasets for "${labelName}"`;
    datasetModal.classList.add('show');
    await renderDatasetModal();
}

/**
 * Closes the dataset modal and clears state.
 */
function closeDatasetModal() {
    datasetModal.classList.remove('show');
    currentLabelName = null;
}

/**
 * Renders both assigned and available dataset lists inside the modal.
 */
async function renderDatasetModal() {
    if (!currentLabelName) return;

    const assigned = modelSettings.labels[currentLabelName] || {};

    // Fetch full datasets list
    let allDatasets = {};
    try {
        allDatasets = await window.electronAPI.invoke('get-datasets-list');
    } catch (error) {
        console.error('Error fetching datasets list:', error);
    }

    // --- Assigned datasets ---
    assignedDatasetsList.innerHTML = '';
    const assignedNames = Object.keys(assigned);
    if (assignedNames.length === 0) {
        assignedDatasetsList.innerHTML = '<div class="dataset-empty-msg">No datasets assigned</div>';
    } else {
        assignedNames.forEach((datasetName) => {
            const item = document.createElement('div');
            item.classList.add('dataset-item', 'assigned');

            const nameSpan = document.createElement('span');
            nameSpan.classList.add('dataset-item-name');
            nameSpan.textContent = datasetName;

            const removeBtn = document.createElement('button');
            removeBtn.classList.add('dataset-remove-btn');
            removeBtn.textContent = '\u2212';
            removeBtn.addEventListener('click', () => removeDatasetFromLabel(datasetName));

            item.appendChild(nameSpan);
            item.appendChild(removeBtn);
            assignedDatasetsList.appendChild(item);
        });
    }

    // --- Available datasets (exclude already assigned) ---
    availableDatasetsList.innerHTML = '';
    const available = Object.entries(allDatasets).filter(([name]) => !(name in assigned));
    if (available.length === 0) {
        availableDatasetsList.innerHTML = '<div class="dataset-empty-msg">No more datasets available</div>';
    } else {
        available.forEach(([datasetName, datasetInfo]) => {
            const item = document.createElement('div');
            item.classList.add('dataset-item', 'available');

            const nameSpan = document.createElement('span');
            nameSpan.classList.add('dataset-item-name');
            nameSpan.textContent = datasetName;

            const addBtn = document.createElement('button');
            addBtn.classList.add('dataset-add-btn');
            addBtn.textContent = '+';
            addBtn.addEventListener('click', () => addDatasetToLabel(datasetName, datasetInfo.path));

            item.appendChild(nameSpan);
            item.appendChild(addBtn);
            availableDatasetsList.appendChild(item);
        });
    }
}

/**
 * Adds a dataset to the current label and saves settings.
 * @param {string} datasetName
 * @param {string} datasetPath
 */
async function addDatasetToLabel(datasetName, datasetPath) {
    if (!currentLabelName) return;
    modelSettings.labels[currentLabelName][datasetName] = datasetPath;
    await saveSettings();
    await renderDatasetModal();
}

/**
 * Removes a dataset from the current label and saves settings.
 * @param {string} datasetName
 */
async function removeDatasetFromLabel(datasetName) {
    if (!currentLabelName) return;
    delete modelSettings.labels[currentLabelName][datasetName];
    await saveSettings();
    await renderDatasetModal();
}

// Close dataset modal via X button
datasetModalClose.addEventListener('click', closeDatasetModal);

// Close dataset modal by clicking backdrop
datasetModal.addEventListener('click', (e) => {
    if (e.target === datasetModal) closeDatasetModal();
});

// Add label button click handler
addLabelBtn.addEventListener('click', () => {
    addLabel(newLabelInput.value);
});

// Enter key in label input field
newLabelInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        addLabel(newLabelInput.value);
    }
});

// Load model settings and populate labels and epochs on page load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        modelSettings = await window.electronAPI.invoke('get-model-settings', projectName);
        if (!modelSettings.labels) modelSettings.labels = {};
        epochSlider.value = modelSettings.epochs ?? 10;
        epochDisplay.textContent = epochSlider.value;
        console.log('Loaded model settings:', modelSettings);
    } catch (error) {
        console.error('Error loading model settings:', error);
        modelSettings = { labels: {}, epochs: 10 };
    }
    renderLabels();
});

// Train button click handler
trainBtn.addEventListener('click', async () => {
    const epochs = epochSlider.value;
    console.log(`%c[UI] Training started with ${epochs} epochs!`, 'color: #007acc; font-weight: bold;');
    
    try {
        // Disable button during execution
        trainBtn.disabled = true;
        trainBtn.textContent = 'Training in progress...';
        
        const resultsDisplay = document.getElementById('results');
        resultsDisplay.textContent = 'Training in progress...';
        resultsDisplay.style.color = '#FFA500';
        
        console.log('%c[UI] Calling IPC handler: executeTrain', 'color: #007acc; font-weight: bold;');
        const callStartTime = Date.now();

        // Fetch pre-built labels JSON and model folder path from main process
        const { labelsJson, modelFolderPath } = await window.electronAPI.invoke('get-model-details-for-python', projectName);

        /* ------------------------------------------------
         Call the Python script via IPC with labelsJson, modelFolderPath, and epochs
         returns an object with success flag and output/error message
        --------------------------------------------------- */
        const result = await window.electronAPI.executeTrain({
            labelsJson,
            modelFolderPath,
            epochs: parseInt(epochs)
        });
        
        const callDuration = Math.round((Date.now() - callStartTime) / 1000);
        console.log(`%c[UI] IPC handler completed in ${callDuration}s`, 'color: #007acc; font-weight: bold;');
        console.log('[UI] Result object:', result);
        
        // Update UI with result
        if (result.success) {
            const execTime = result.executionTime ? ` (${result.executionTime}s)` : '';
            resultsDisplay.textContent = `Training completed successfully!${execTime}`;
            resultsDisplay.style.color = '#28a745';
            console.log('%c[UI] Training successful!', 'color: #28a745; font-weight: bold;');
            console.log('[UI] Output:', result.output);
        } else {
            const execTime = result.executionTime ? ` (${result.executionTime}s)` : '';
            resultsDisplay.textContent = `Training failed.${execTime} Check console for details.`;
            resultsDisplay.style.color = '#dc3545';
            console.error('%c[UI] Training failed!', 'color: #dc3545; font-weight: bold;');
            console.error('[UI] Error:', result.error);
        }
    } catch (error) {
        // error handling if for some reason Electron or Python do not run successfully
        console.error('%c[UI] IPC error:', 'color: #dc3545; font-weight: bold;', error);
        document.getElementById('results').textContent = `IPC Error: ${error.message}`;
        document.getElementById('results').style.color = '#dc3545';
    } finally {
        // runs regardless of success or failure to ensure button is re-enabled
        // Re-enable button
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train';
    }
});


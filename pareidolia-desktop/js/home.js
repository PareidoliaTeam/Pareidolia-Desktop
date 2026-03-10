// ============================================================
// Query Selectors
// ============================================================
// SessionStorage elements
const projectName = sessionStorage.getItem('projectName') || 'Project';
const pageValue = sessionStorage.getItem('pageValue') || 'Page';
const projectPath = sessionStorage.getItem('projectPath') || 'No path';

// Data set elements
const datasetNameDisplay = document.getElementById('current-dataset-title');
const datasetPathDisplay = document.getElementById('current-dataset-filepath');
const datasetsList = document.getElementById('datasetsList');

// Model elements
const modelNameDisplay = document.getElementById('current-model-title');
const modelPathDisplay = document.getElementById('current-model-path');
const modelsList = document.getElementById('modelsList');
const addModelBtn = document.querySelector('.create-btn');

// Model modal elements
const addProjectModal = document.getElementById('add-model-modal');
const projectNameInput = document.getElementById('project-name-input');
const modalCreateBtn = document.getElementById('modal-create-btn');
const modalCancelBtn = document.getElementById('modal-cancel-btn');
const modalClose = document.querySelector('.modal-close');

// Help modal elements
const helpModal = document.getElementById('help-modal');
const helpBtn = document.getElementById('help-btn');
const helpModalClose = document.getElementById('help-modal-close');

// Settings modal elements
const settingsModal = document.getElementById('settings-modal');
const settingsBtn = document.getElementById('settings-btn');
const settingsModalClose = document.getElementById('settings-modal-close');

//QR modal elements
const qrCodeContainer = document.getElementById('qr-code-container');
const qrModal = document.getElementById('qr-modal');
const uploadBtn = document.getElementById('upload-btn');
const qrModalClose = document.getElementById('qr-modal-close');

// Train Menu
const epochSlider = document.getElementById('epoch-slider');
const epochValueDisplay = document.getElementById('epoch-value');
const modelTrainBtn = document.getElementById('model-train-btn');
const modelTrainResults = document.getElementById('model-train-results');

// Gallery
const galleryContainer = document.querySelector('.gallery-grid');
const optionsModal = document.getElementById('image-options-modal');
const modalImg = document.getElementById('modal-preview-img');
const optionsModalClose = document.getElementById('close-modal');
const carousel = document.querySelector('.carousel');

// Label management
const labelsList = document.getElementById('labels-list');
const newLabelInput = document.getElementById('new-label-input');
const addLabelBtn = document.getElementById('add-label-btn');

// Dataset modal elements
const datasetModal = document.getElementById('dataset-modal');
const datasetModalTitle = document.getElementById('dataset-modal-title');
const datasetModalClose = document.getElementById('dataset-modal-close');
const assignedDatasetsList = document.getElementById('assigned-datasets-list');
const availableDatasetsList = document.getElementById('available-datasets-list');

// Current model settings state
let modelSettings = null;
let currentLabelName = null;


// ============================================================
// Functions
// ============================================================


/**
 * Shows the view requested and hides all other views
 * @param {string} viewId - The name of the view to display
 */
function showView(viewId) {

    const views = document.querySelectorAll('.app-view');
    views.forEach((view) => {
        view.style.display = 'none';
    });

    const targetView = document.getElementById(viewId);
    if(targetView) {
        targetView.style.display = 'flex';
    }
}
/**
 * Shows the model info view and displays the model name.
 * @param {string} modelName - The name of the model to display
 * @param {string} modelNamePath - The path of the model to display
 */
async function showModel(modelName,modelNamePath) {
    if (modelNameDisplay){
        modelNameDisplay.textContent = modelName;
    }
    if (modelPathDisplay){
        modelPathDisplay.textContent = modelNamePath;
    }
    sessionStorage.setItem('projectName', modelName);
    sessionStorage.setItem('projectPath', modelNamePath);
    showView('view-model-info');
    await loadModelSettingsForView(modelName);
}

/**
 * Shows the dataset info view and displays the dataset name.
 * @param {string} datasetName - The name of the dataset to display
 * @param {string} datasetNamePath - The path of the dataset to display
 */
function showDataset(datasetName,datasetPath) {
    if (datasetNameDisplay) {
        datasetNameDisplay.textContent = datasetName;
    }
    if (datasetPathDisplay){
        datasetPathDisplay.textContent = datasetPath;
    }
    sessionStorage.setItem('projectName', datasetName);
    sessionStorage.setItem('projectPath', datasetPath);
    showView('view-dataset-info');
    loadCarousel();
}

function showDatasetGallery() {
    showView('view-dataset-gallery');
    loadGallery();
}

/**
 * Opens the add project modal dialog.
 */
function openAddProjectModal() {
    addProjectModal.style.display= 'flex';
    projectNameInput.focus();
}

/**
 * Closes the add project modal dialog.
 */
function closeAddProjectModal() {
    addProjectModal.style.display= 'none';
}

/**
 * Opens the help modal dialog.
 */
function openHelpModal() {
    helpModal.style.display= 'flex';
    projectNameInput.focus();
}

/**
 * Closes the help modal dialog.
 */
function closeHelpModal() {
    helpModal.style.display= 'none';
}

/**
 * Opens the settings modal dialog.
 */
function openSettingsModal() {
    settingsModal.style.display= 'flex';
}

/**
 * Closes the settings modal dialog.
 */
function closeSettingsModal() {
    settingsModal.style.display= 'none';
}

/**
 * Opens the QR modal dialog.
 */
function openQRModal() {
    qrModal.style.display= 'flex';
}

/**
 * Closes the qe project modal dialog.
 */

function closeQRModal() {
    qrModal.style.display= 'none';
}

/**
 * Opens the image options model
 */
function openOptionsModal(imgData) {
    optionsModal.style.display = 'flex';
    modalImg.src = imgData.url;
    //modalTitle.textContent = imgData.name;
    // store path for delete option
    optionsModal.dataset.currentPath = imgData.url;
}

/**
 * Opens the imageoptions model
 */
function closeOptionsModal() {
    optionsModal.style.display = 'none';
}

// ============================================================
// LABEL & DATASET MANAGEMENT
// ============================================================

/**
 * Loads model settings from disk and populates the labels list and epoch slider.
 * Called whenever a model is opened.
 * @param {string} modelName
 */
async function loadModelSettingsForView(modelName) {
    try {
        modelSettings = await window.electronAPI.invoke('get-model-settings', modelName);
        if (!modelSettings.labels) modelSettings.labels = {};
        epochSlider.value = modelSettings.epochs ?? 10;
        epochValueDisplay.textContent = epochSlider.value;
    } catch (error) {
        console.error('Error loading model settings:', error);
        modelSettings = { labels: {}, epochs: 10 };
    }
    renderLabels();
}

/**
 * Persists the full modelSettings object back to model-settings.json via IPC.
 */
async function saveModelSettings() {
    if (!modelSettings) return;
    const modelName = sessionStorage.getItem('projectName');
    try {
        await window.electronAPI.invoke('update-model-settings', { modelName, newSettings: modelSettings });
        console.log('Model settings saved');
    } catch (error) {
        console.error('Error saving model settings:', error);
    }
}

/**
 * Renders the labels list in the settings panel.
 */
function renderLabels() {
    if (!labelsList) return;
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
        removeBtn.textContent = '\u2212';
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
 * Adds a new label to model settings and re-renders the list.
 * @param {string} labelName
 */
function addLabel(labelName) {
    const trimmedName = labelName.trim();
    if (!trimmedName) { 
        //replace with pop up modals, screws up texet cursor on windows
        // alert('Label name cannot be empty'); 
        return; }
    if (trimmedName in modelSettings.labels) { 
        // alert('Label already exists'); 
        return; }

    modelSettings.labels[trimmedName] = {};
    renderLabels();
    newLabelInput.value = '';
    newLabelInput.focus();
    saveModelSettings();
}

/**
 * Removes a label from model settings and re-renders the list.
 * @param {string} labelName
 */
function removeLabel(labelName) {
    delete modelSettings.labels[labelName];
    renderLabels();
    saveModelSettings();
}

/**
 * Opens the dataset assignment modal for the given label.
 * @param {string} labelName
 */
async function openDatasetModal(labelName) {
    currentLabelName = labelName;
    datasetModalTitle.textContent = `Datasets for "${labelName}"`;
    datasetModal.style.display = 'flex';
    await renderDatasetModal();
}

/**
 * Closes the dataset assignment modal.
 */
function closeDatasetModal() {
    datasetModal.style.display = 'none';
    currentLabelName = null;
}

/**
 * Renders both the assigned and available dataset lists inside the modal.
 */
async function renderDatasetModal() {
    if (!currentLabelName) return;

    const assigned = modelSettings.labels[currentLabelName] || {};

    let allDatasets = {};
    try {
        allDatasets = await window.electronAPI.invoke('get-datasets-list');
    } catch (error) {
        console.error('Error fetching datasets list:', error);
    }

    // Assigned datasets
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

    // Available datasets (exclude already assigned)
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
 * Assigns a dataset to the current label and saves settings.
 * @param {string} datasetName
 * @param {string} datasetPath
 */
async function addDatasetToLabel(datasetName, datasetPath) {
    if (!currentLabelName) return;
    modelSettings.labels[currentLabelName][datasetName] = datasetPath;
    await saveModelSettings();
    await renderDatasetModal();
}

/**
 * Removes a dataset from the current label and saves settings.
 * @param {string} datasetName
 */
async function removeDatasetFromLabel(datasetName) {
    if (!currentLabelName) return;
    delete modelSettings.labels[currentLabelName][datasetName];
    await saveModelSettings();
    await renderDatasetModal();
}

/**
 * Switches between viewing models and datasets
 * @param {string} viewId - The name of the view to display
 * @param {string} sidebarClass - The name of the sidebar to display
 */
async function switchMode(viewId, sidebarClass) {
    // hide all views and sidebars
    const views = document.querySelectorAll('.app-view');
    views.forEach((view) => {view.style.display = 'none';});
    const sidebars = document.querySelectorAll('aside');
    sidebars.forEach(s => {s.classList.remove('sidebar-active');});
    // show selected view and sidebar
    const view = document.getElementById(viewId);
    if (view) {view.style.display = 'flex';}
    const selector = sidebarClass.startsWith('.') ? sidebarClass : `.${sidebarClass}`;
    const targetSidebar = document.querySelector(selector);

    if (targetSidebar) {
        targetSidebar.classList.add('sidebar-active');
    }
    await loadDatasetsFromFolder();
    await loadModelsFromFolder();
}

/**
 * Creates a new model by prompting the user for a name and creating a folder.
 * Uses IPC to communicate with the main process to create the folder.
 */
async function handleAddProject() {
    const modelName = projectNameInput.value.trim();

    if (!modelName) {

        // need to make a new modal for this pop up
        //alert('Model name cannot be empty');
        return;
    }

    try {
        const modelPath = await window.electronAPI.invoke('create-model-folder', modelName);
        console.log('Model created at:', modelPath);

        // Reset input and close modal
        projectNameInput.value = '';
        closeAddProjectModal();


        // Reload the projects list
    } catch (error) {
        console.error('Error creating project:', error);
    }
    // Reload the projects list
    await loadModelsFromFolder();
}

/**
 * Generates and displays a QR code for the local server connection.
 * The QR code contains the local IP address and port 3001.
 */
async function generateQRCode() {
    try {
        // Get the local IP address from the main process
        const localIP = await window.electronAPI.invoke('get-local-ip');

        if (!localIP) {
            console.error('Could not determine local IP address');
            qrCodeContainer.textContent = 'Unable to generate QR code';
            return;
        }

        // Construct the server URL
        const serverURL = `http://${localIP}:3001`;
        console.log('Generating QR code for:', serverURL);

        // Clear the container
        qrCodeContainer.innerHTML = '';

        // Generate QR code using qrcodejs library
        new QRCode(qrCodeContainer, {
            text: serverURL,
            width: 180,
            height: 180,
            colorDark: '#000000',
            colorLight: '#ffffff',
            correctLevel: QRCode.CorrectLevel.H
        });

        console.log('QR code generated successfully');
    } catch (error) {
        console.error('Error in generateQRCode:', error);
        qrCodeContainer.textContent = 'Error: ' + error.message;
    }
}


/**
 * Loads all datasets folders from the Pareidolia folder and creates buttons for them.
 * Each button has the folder path as its value.
 */
async function loadDatasetsFromFolder() {
    try {
        // First ensure the Pareidolia folder exists
        const pareidoliaPath = await window.electronAPI.invoke('get-pareidolia-path');
        console.log('Pareidolia path:', pareidoliaPath);

        // Clear existing project buttons
        datasetsList.innerHTML = '';

        // Call a new IPC handler to get the list of datasets
        const datasets = await window.electronAPI.invoke('get-datasets-list');

        // Create buttons for each project
        Object.entries(datasets).forEach(([datasetName, datasetInfo]) => {
            const li = document.createElement('li');
            li.classList.add('dataset-item');
            li.setAttribute('data-path', datasetInfo.path);

            const div = document.createElement('div');
            div.classList.add('dataset-open-btn');
            div.textContent = datasetName;

            li.addEventListener('click', (e) => {
                e.stopPropagation();
                const datasetPath = li.getAttribute('data-path');
                const datasetDisplayName = div.textContent;

                showDataset(datasetDisplayName,datasetPath);
            });
            li.appendChild(div);
            datasetsList.appendChild(li);
        });

        console.log(`Loaded ${datasets.length} projects`);
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}

/**
 * Loads all models (currently datasets) folders from the Pareidolia folder and creates buttons for them.
 * Each button has the folder path as its value.
 */
async function loadModelsFromFolder() {
    try {
        // First ensure the Pareidolia folder exists
        const pareidoliaPath = await window.electronAPI.invoke('get-pareidolia-path');
        console.log('Pareidolia path:', pareidoliaPath);

        // Clear existing project buttons
        modelsList.innerHTML = '';

        // Call a new IPC handler to get the list of models
        const models = await window.electronAPI.invoke('get-models-list');

        // Create buttons for each project
        Object.entries(models).forEach(([modelName, modelInfo]) => {
            const li = document.createElement('li');
            li.classList.add('model-item');
            li.setAttribute('data-path', modelInfo.path);

            const div = document.createElement('div');
            div.classList.add('model-open-btn');
            div.textContent = modelName;
            li.addEventListener('click', (e) => {
                e.stopPropagation();
                const modelPath = li.getAttribute('data-path');
                const modelDisplayName = div.textContent;
                showModel(modelDisplayName,modelPath);
            });
            li.appendChild(div);
            modelsList.appendChild(li);
        });

        console.log(`Loaded ${models.length} projects`);
    } catch (error) {
        console.error('Error loading models:', error);
    }
}
/**
 * Loads carousel and attatches images into it
 */
async function loadCarousel() {
    const currentPath = sessionStorage.getItem('projectPath');
    if(projectPath) {
        // Request images
        console.log(currentPath);
        const images = await window.electronAPI.invoke('get-project-images', currentPath + "/positives");

        // Loop through images and create elements
        carousel.innerHTML = '';
        images.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.src = imgData.url;
            imgElement.alt = imgData.name;
            imgElement.className = 'carousel-item';

            carousel.appendChild(imgElement);
        });
    }
}

/**
 * Loads gallery and attatches images into it
 */
async function loadGallery(){
    const currentPath = sessionStorage.getItem('projectPath');
    //const currentName = sessionStorage.getItem('projectName');
    if(projectPath){
        galleryContainer.innerHTML = '';
        const images = await window.electronAPI.invoke('get-project-images', currentPath + "/positives");

        images.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.src = imgData.url;
            imgElement.alt = imgData.name;
            imgElement.className = 'gallery-item';

            //Add click listener
            imgElement.addEventListener('click', () => {
                openOptionsModal(imgData);
            });

            galleryContainer.appendChild(imgElement);
        });
    }
}

// ============================================================
// Event Listeners
// ============================================================

// Upload Button - opens QR modal
uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    openQRModal();
});

// QR Modal close button
qrModalClose.addEventListener('click', (e) => {
    e.stopPropagation();
    closeQRModal();
})

// Add Project button - open modal to create new project
addModelBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    openAddProjectModal();
});

// Open Help modal
helpBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    openHelpModal();
})

// Help Modal Close button (X) - close modal
helpModalClose.addEventListener('click', (e) => {
    e.stopPropagation();
    closeHelpModal();
});

// Open Settings modal
settingsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    openSettingsModal();
})

// Close Settings modal
settingsModalClose.addEventListener('click', (e) => {
    e.stopPropagation();
    closeSettingsModal();
})

// Modal Create button - submit project creation
modalCreateBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    await handleAddProject();
});

// Modal Cancel button - close modal without creating
modalCancelBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    closeAddProjectModal();
});

// Modal Close button (X) - close modal
modalClose.addEventListener('click', (e) => {
    e.stopPropagation();
    closeAddProjectModal();
});

// Modal background click - close modal
addProjectModal.addEventListener('click', (e) => {
    if (e.target === addProjectModal) {
        closeAddProjectModal();
    }
});

// Enter key in input field - submit form
projectNameInput.addEventListener('keypress', async (e) => {
    if (e.key === 'Enter') {
        await handleAddProject();
    }
});

// closes option modal
optionsModalClose.addEventListener('click', (e) => {
    e.stopPropagation();
    closeOptionsModal();
})

// EpochSlider to change
epochSlider.addEventListener('input', (event) => {
    const val = event.target.value;

    // show value
    epochValueDisplay.textContent = val;
});

// Save epochs when user finishes dragging
epochSlider.addEventListener('change', async (event) => {
    if (!modelSettings) return;
    modelSettings.epochs = parseInt(event.target.value);
    await saveModelSettings();
});

// carpousel animation
carousel.addEventListener('animationiteration', () => {
    // Reset carousel position for seamless looping
    carousel.style.animation = 'none';
    setTimeout(() => {
        carousel.style.animation = '';
    }, 10);
});

// Add label button
addLabelBtn.addEventListener('click', () => {
    addLabel(newLabelInput.value);
});

// Enter key in label input
newLabelInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addLabel(newLabelInput.value);
});

// Dataset modal close button
datasetModalClose.addEventListener('click', closeDatasetModal);

// Dataset modal backdrop click
datasetModal.addEventListener('click', (e) => {
    if (e.target === datasetModal) closeDatasetModal();
});

// Train Model button click handler
modelTrainBtn.addEventListener('click', async () => {
    const epochs = epochSlider.value;
    const currentModelName = sessionStorage.getItem('projectName');
    console.log(`%c[UI] Training started with ${epochs} epochs!`, 'color: #007acc; font-weight: bold;');

    try {
        modelTrainBtn.disabled = true;
        modelTrainBtn.textContent = 'Training in progress...';
        modelTrainResults.textContent = 'Training in progress...';
        modelTrainResults.style.color = '#FFA500';

        const callStartTime = Date.now();

        const { labelsJson, modelFolderPath } = await window.electronAPI.invoke('get-model-details-for-python', currentModelName);

        const result = await window.electronAPI.executeTrain({
            labelsJson,
            modelFolderPath,
            epochs: parseInt(epochs)
        });

        const callDuration = Math.round((Date.now() - callStartTime) / 1000);
        console.log(`%c[UI] IPC handler completed in ${callDuration}s`, 'color: #007acc; font-weight: bold;');

        if (result.success) {
            const execTime = result.executionTime ? ` (${result.executionTime}s)` : '';
            modelTrainResults.textContent = `Training completed successfully!${execTime}`;
            modelTrainResults.style.color = '#28a745';
            console.log('%c[UI] Training successful!', 'color: #28a745; font-weight: bold;');
        } else {
            const execTime = result.executionTime ? ` (${result.executionTime}s)` : '';
            modelTrainResults.textContent = `Training failed.${execTime} Check console for details.`;
            modelTrainResults.style.color = '#dc3545';
            console.error('%c[UI] Training failed!', 'color: #dc3545; font-weight: bold;');
            console.error('[UI] Error:', result.error);
        }
    } catch (error) {
        console.error('%c[UI] IPC error:', 'color: #dc3545; font-weight: bold;', error);
        modelTrainResults.textContent = `IPC Error: ${error.message}`;
        modelTrainResults.style.color = '#dc3545';
    } finally {
        modelTrainBtn.disabled = false;
        modelTrainBtn.textContent = 'Train Model';
    }
});

// ============================================================
// Initialization
// ============================================================

// Load projects from folder when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    switchMode('view-home','.left-sidebar-models');

    await loadModelsFromFolder();
    await generateQRCode();
});


window.electronAPI.invoke('setup-python-venv')
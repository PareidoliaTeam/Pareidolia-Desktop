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

// Gallery
const galleryContainer = document.querySelector('.gallery-grid');
const optionsModal = document.getElementById('image-options-modal');
const modalImg = document.getElementById('modal-preview-img');
const optionsModalClose = document.getElementById('close-modal');
const carousel = document.querySelector('.carousel');


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
function showModel(modelName,modelNamePath) {
    if (modelNameDisplay){
        modelNameDisplay.textContent = modelName;
    }
    if (modelPathDisplay){
        modelPathDisplay.textContent = modelNamePath;
    }
    sessionStorage.setItem('projectName', modelName);
    sessionStorage.setItem('projectPath', modelNamePath);
    showView('view-model-info');
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
// Async Functions
// ============================================================

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

            const div = document.createElement('div');
            div.classList.add('dataset-open-btn');
            div.setAttribute('data-path', datasetInfo.path);
            div.textContent = datasetName;

            div.addEventListener('click', (e) => {
                e.stopPropagation();
                const datasetPath = div.getAttribute('data-path');
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

            const div = document.createElement('div');
            div.classList.add('model-open-btn');
            div.setAttribute('data-path', modelInfo.path);
            div.textContent = modelName;
            div.addEventListener('click', (e) => {
                e.stopPropagation();
                const modelPath = div.getAttribute('data-path');
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

// carpousel animation
carousel.addEventListener('animationiteration', () => {
    // Reset carousel position for seamless looping
    carousel.style.animation = 'none';
    setTimeout(() => {
        carousel.style.animation = '';
    }, 10);
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

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
const datasetSearchInput = document.getElementById('dataset-search');
const deleteDatasetBtn = document.getElementById('delete-dataset-btn');
const galleryNameDisplay = document.getElementById('current-gallery-title');
const galleryPathDisplay = document.getElementById('current-gallery-filepath');

// Model elements
const modelNameDisplay = document.getElementById('current-model-title');
const modelPathDisplay = document.getElementById('current-model-path');
const modelsList = document.getElementById('modelsList');
const modelSearchInput = document.getElementById('model-search');
const sidebarModelsRadio = document.getElementById('sidebar-models-active');
const sidebarDatasetsRadio = document.getElementById('sidebar-models-datasets');
const addModelBtn = document.querySelector('.create-btn');
const datasetImportBtn = document.getElementById('dataset-import-btn');
const deleteModelBtn = document.getElementById('delete-model-btn');
const primarySidebar = document.querySelector('.left-sidebar-models');

// Model modal elements
const addProjectModal = document.getElementById('add-model-modal');
const projectNameInput = document.getElementById('project-name-input');
const projectTypeInputs = document.getElementsByName('project-type');
const modalCreateBtn = document.getElementById('modal-create-btn');
const modalCancelBtn = document.getElementById('modal-cancel-btn');
const closeCreateDatasetModal = document.getElementById('close-add-dataset-modal')
const modelModalClose = document.getElementById('model-modal-close');
const deleteModelModal = document.getElementById('delete-model-modal');
const deleteModelModalClose = document.getElementById('delete-model-modal-close');
const deleteModelName = document.getElementById('delete-model-name');
const deleteModelConfirmInput = document.getElementById('delete-model-confirm-input');
const deleteModelCancelBtn = document.getElementById('delete-model-cancel-btn');

// Help modal elements
//const helpModal = document.getElementById('help-modal');
//const helpBtn = document.getElementById('help-btn');
//const helpModalClose = document.getElementById('help-modal-close');

// Settings modal elements
//const settingsModal = document.getElementById('settings-modal');
//const settingsBtn = document.getElementById('settings-btn');
//const settingsModalClose = document.getElementById('settings-modal-close');

//QR modal elements
const sidebarQrPanel = document.querySelector('.sidebar-qr-panel');
const sidebarQrToggle = document.getElementById('sidebar-qr-toggle');
const sidebarQrArrow = document.getElementById('sidebar-qr-arrow');
const sidebarQrCodeContainer = document.getElementById('sidebar-qr-code-container');
const qrCodeContainer = document.getElementById('qr-code-container');
const qrModal = document.getElementById('qr-modal');
const uploadBtn = document.getElementById('upload-btn');
const qrModalClose = document.getElementById('qr-modal-close');

// Train Menu
const epochSlider = document.getElementById('epoch-slider');
const epochValueDisplay = document.getElementById('epoch-value');
const modelTrainBtn = document.getElementById('model-train-btn');
const modelTrainResults = document.getElementById('model-train-results');
const testLastTrainedFramework = document.getElementById('test-last-trained-framework');
const predictionLastTrainedFramework = document.getElementById('prediction-last-trained-framework');
const trainProjectTypeInputs = document.querySelectorAll('input[name="train-project-type"]');
const trainFrameworkInputs = document.querySelectorAll('[data-framework-toggle]');
const testFrameworkInputs = document.querySelectorAll('input[name="test-framework"]');
const predictionFrameworkInputs = document.querySelectorAll('input[name="prediction-framework"]');

// Dataset Modal
const datasetImportModal = document.getElementById('import-dataset-modal');

function getSelectedFramework() {
    const selectedRadio = document.querySelector('[data-framework-toggle]:checked');
    return selectedRadio && selectedRadio.value === 'false' ? 'pytorch' : 'tensorflow';
}

function syncFrameworkToggles(value) {
    trainFrameworkInputs.forEach((input) => {
        input.checked = input.value === value;
    });
}

function frameworkToInputValue(framework) {
    return framework === 'pytorch' ? 'false' : 'true';
}

function inputValueToFramework(value) {
    return value === 'false' ? 'pytorch' : 'tensorflow';
}

function setFrameworkInputs(inputs, framework) {
    const value = frameworkToInputValue(framework);
    inputs.forEach((input) => {
        input.checked = input.value === value;
    });
}

function syncRuntimeFrameworkToggles(framework) {
    const defaultFramework = framework || modelSettings?.lastTrainedFramework || modelSettings?.modelType || 'tensorflow';

    setFrameworkInputs(testFrameworkInputs, defaultFramework);
    setFrameworkInputs(predictionFrameworkInputs, defaultFramework);
}

function getSelectedRuntimeFramework(context) {
    const selector = context === 'prediction'
        ? 'input[name="prediction-framework"]:checked'
        : 'input[name="test-framework"]:checked';
    const selectedRadio = document.querySelector(selector);

    return inputValueToFramework(selectedRadio?.value);
}

function getFrameworkDisplayName(framework) {
    return framework === 'pytorch' ? 'PyTorch' : 'TensorFlow';
}

function updateLastTrainedFrameworkDisplays(settings = modelSettings) {
    const statusElements = [testLastTrainedFramework, predictionLastTrainedFramework].filter(Boolean);
    if (statusElements.length === 0) return;

    let statusText = 'Latest trained: Never';
    if (settings?.lastTrained && settings?.lastTrainedFramework) {
        statusText = `Latest trained: ${settings.lastTrained} with ${getFrameworkDisplayName(settings.lastTrainedFramework)}`;
    }

    statusElements.forEach((element) => {
        element.textContent = statusText;
    });
}

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
const datasetCleanupModal = document.getElementById('dataset-cleanup-modal');
const datasetCleanupModalClose = document.getElementById('dataset-cleanup-modal-close');
const datasetCleanupList = document.getElementById('dataset-cleanup-list');
const datasetCleanupOkBtn = document.getElementById('dataset-cleanup-ok-btn');
const trainingValidationModal = document.getElementById('training-validation-modal');
const trainingValidationModalClose = document.getElementById('training-validation-modal-close');
const trainingValidationList = document.getElementById('training-validation-list');
const trainingValidationOkBtn = document.getElementById('training-validation-ok-btn');
const deleteDatasetModal = document.getElementById('delete-dataset-modal');
const deleteDatasetModalClose = document.getElementById('delete-dataset-modal-close');
const deleteDatasetName = document.getElementById('delete-dataset-name');
const deleteDatasetConfirmInput = document.getElementById('delete-dataset-confirm-input');
const deleteDatasetCancelBtn = document.getElementById('delete-dataset-cancel-btn');

// Current model settings state
let modelSettings = null;
let currentLabelName = null;
let activeBlock = null;
let sidebarToggleAnimationFrame = null;

// builder ids
const builderModal = document.getElementById('builder-modal');
const builderModalBtn = document.getElementById('builder-modal-btn');
const builderModalCloseBtn = document.getElementById('builder-modal-close');
const layerItems = document.querySelectorAll('.layer-item');
const builderCanvas = document.getElementById('builder-canvas');
const layerDescriptionTitle = document.getElementById('layer-description-title');
const layerDescriptionText = document.getElementById('layer-description-text');
const viewCodeBtn = document.getElementById('view-code-btn');
const viewBlockBtn = document.getElementById('view-blocks-btn');

const layerDescriptions = {
    Conv2D: {
        title: 'Conv2D',
        text: 'Finds visual patterns by sliding filters across an image. Early Conv2D layers learn edges and textures, while later ones combine those signals into shapes or object parts.'
    },
    MaxPooling2D: {
        title: 'MaxPooling2D',
        text: 'Shrinks feature maps by keeping the strongest value in each small region. This reduces computation and helps the model focus on the most obvious features.'
    },
    AveragePooling2D: {
        title: 'AveragePooling2D',
        text: 'Shrinks feature maps by averaging each small region. It smooths activations and keeps broader context instead of only preserving the strongest response.'
    },
    GlobalAveragePooling2D: {
        title: 'Global Average Pooling',
        text: 'Turns each feature map into one average value. It is often used near the end of an image model to summarize learned features before classification.'
    },
    Flatten: {
        title: 'Flatten',
        text: 'Converts multi-dimensional image features into one long vector. This prepares convolution or pooling output for Dense layers.'
    },
    Dense: {
        title: 'Dense',
        text: 'Connects every input value to every output unit. Dense layers combine learned features to make decisions, especially near the end of a classifier.'
    },
    Dropout: {
        title: 'Dropout',
        text: 'Randomly ignores some activations during training. This helps prevent overfitting by making the model less dependent on any single feature.'
    },
    RandomFlip: {
        title: 'Random Flip',
        text: 'Randomly flips training images horizontally, vertically, or both. This teaches the model that flipped versions should still count as the same kind of example.'
    },
    RandomRotation: {
        title: 'Random Rotation',
        text: 'Randomly rotates training images by a small amount. It helps the model handle tilted examples without needing more labeled images.'
    },
    RandomZoom: {
        title: 'Random Zoom',
        text: 'Randomly zooms training images in or out. This improves robustness when the subject appears at different sizes or distances.'
    },
    RandomContrast: {
        title: 'Random Contrast',
        text: 'Randomly changes image contrast during training. This helps the model work better across lighting differences and camera conditions.'
    }
};

// charts
let accuracyChart, lossChart;
let chartResizeObserver = null;
let chartResizeFrame = null;
const chartStateByModel = new Map();
let activeChartModelName = null;
let activeTrainingModelName = null;

// prediction ids
const dropZone = document.getElementById('drop-zone');
const predictionPreview = document.getElementById('prediction-preview');
const resultsArea = document.getElementById('prediction-results');

// test ids
const runTestButton = document.getElementById('run-test-btn');

// ============================================================
// Functions
// ============================================================

function createEmptyChartState() {
    return {
        labels: [],
        accuracy: {
            train: [],
            val: []
        },
        loss: {
            train: [],
            val: []
        }
    };
}

function cloneChartState(chartState) {
    const normalized = chartState || createEmptyChartState();

    return {
        labels: [...(normalized.labels || [])],
        accuracy: {
            train: [...(normalized.accuracy?.train || [])],
            val: [...(normalized.accuracy?.val || [])]
        },
        loss: {
            train: [...(normalized.loss?.train || [])],
            val: [...(normalized.loss?.val || [])]
        }
    };
}

function isEmptyChartState(chartState) {
    return !chartState
        || (chartState.labels?.length || 0) === 0
        && (chartState.accuracy?.train?.length || 0) === 0
        && (chartState.accuracy?.val?.length || 0) === 0
        && (chartState.loss?.train?.length || 0) === 0
        && (chartState.loss?.val?.length || 0) === 0;
}

function normalizeChartState(chartState) {
    if (!chartState || typeof chartState !== 'object') {
        return createEmptyChartState();
    }

    return cloneChartState(chartState);
}

function getChartStateForModel(modelName) {
    if (!chartStateByModel.has(modelName)) {
        chartStateByModel.set(modelName, createEmptyChartState());
    }

    return chartStateByModel.get(modelName);
}

function setChartStateForModel(modelName, chartState) {
    chartStateByModel.set(modelName, cloneChartState(chartState));
}

function renderChartState(chartState) {
    if (!accuracyChart || !lossChart) return;

    const normalized = normalizeChartState(chartState);

    accuracyChart.data.labels = [...normalized.labels];
    accuracyChart.data.datasets[0].data = [...normalized.accuracy.train];
    accuracyChart.data.datasets[1].data = [...normalized.accuracy.val];

    lossChart.data.labels = [...normalized.labels];
    lossChart.data.datasets[0].data = [...normalized.loss.train];
    lossChart.data.datasets[1].data = [...normalized.loss.val];

    accuracyChart.update();
    lossChart.update();
}

function renderSummaryCard(chartState) {
    const normalized = normalizeChartState(chartState);
    const lastIdx = normalized.accuracy.train.length - 1;

    const finalAcc = lastIdx >= 0 ? normalized.accuracy.train[lastIdx] : null;
    const finalValAcc = lastIdx >= 0 ? normalized.accuracy.val[lastIdx] : null;
    const finalLoss = lastIdx >= 0 ? normalized.loss.train[lastIdx] : null;
    const finalValLoss = lastIdx >= 0 ? normalized.loss.val[lastIdx] : null;

    const accDisplay = document.getElementById('sum-acc');
    const valAccDisplay = document.getElementById('sum-val-acc');
    const lossDisplay = document.getElementById('sum-loss');
    const valLossDisplay = document.getElementById('sum-val-loss');

    if (accDisplay) accDisplay.textContent = finalAcc === null ? '-' : `${(finalAcc * 100).toFixed(2)}%`;
    if (valAccDisplay) valAccDisplay.textContent = finalValAcc === null ? '-' : `${(finalValAcc * 100).toFixed(2)}%`;
    if (lossDisplay) lossDisplay.textContent = finalLoss === null ? '-' : parseFloat(finalLoss).toFixed(4);
    if (valLossDisplay) valLossDisplay.textContent = finalValLoss === null ? '-' : parseFloat(finalValLoss).toFixed(4);
}

function syncChartStateToCurrentModel() {
    if (!activeChartModelName) return;

    const chartState = getChartStateForModel(activeChartModelName);
    if (modelSettings) {
        modelSettings.chartHistory = cloneChartState(chartState);
    }

    renderChartState(chartState);
    renderSummaryCard(chartState);
}

async function persistModelSettings(modelName, settings) {
    if (!modelName || !settings) return;

    try {
        await window.electronAPI.invoke('update-model-settings', { modelName, newSettings: settings });
        console.log('Model settings saved');
    } catch (error) {
        console.error('Error saving model settings:', error);
    }
}

async function getInvalidDatasetReason(datasetPath) {
    if (!datasetPath) {
        return 'Dataset path not found';
    }

    const exists = await window.electronAPI.invoke('path-exists', datasetPath);
    if (!exists) {
        return 'Dataset path not found';
    }

    const images = await window.electronAPI.invoke('get-project-images', datasetPath);
    const imageCount = Array.isArray(images) ? images.length : 0;

    return imageCount < 3 ? `Needs at least 3 images (${imageCount} found)` : '';
}

function openDatasetCleanupModal(removedAssignments) {
    if (!datasetCleanupModal || !datasetCleanupList || removedAssignments.length === 0) {
        return;
    }

    datasetCleanupList.innerHTML = '';
    removedAssignments.forEach(({ datasetName, labelName, reason }) => {
        const item = document.createElement('div');
        item.classList.add('dataset-cleanup-item');

        const message = document.createElement('span');
        message.innerHTML = `Dataset <strong></strong> has been removed from label <strong></strong>.`;
        const strongTags = message.querySelectorAll('strong');
        strongTags[0].textContent = datasetName;
        strongTags[1].textContent = labelName;

        const reasonSpan = document.createElement('span');
        reasonSpan.classList.add('dataset-cleanup-reason');
        reasonSpan.textContent = reason;

        item.appendChild(message);
        item.appendChild(reasonSpan);
        datasetCleanupList.appendChild(item);
    });

    datasetCleanupModal.style.display = 'flex';
}

function closeDatasetCleanupModal() {
    if (!datasetCleanupModal) return;
    datasetCleanupModal.style.display = 'none';
}

async function removeInvalidAssignedDatasets(modelName, settings = modelSettings, options = {}) {
    const { showNotice = true } = options;
    if (!modelName || !settings) {
        return [];
    }

    const labels = settings?.labels || {};
    const removedAssignments = [];

    for (const [labelName, datasetEntries] of Object.entries(labels)) {
        if (!datasetEntries || typeof datasetEntries !== 'object') {
            continue;
        }

        for (const [datasetName, datasetPath] of Object.entries(datasetEntries)) {
            const reason = await getInvalidDatasetReason(datasetPath);
            if (!reason) {
                continue;
            }

            delete datasetEntries[datasetName];
            removedAssignments.push({ datasetName, labelName, reason });
        }
    }

    if (removedAssignments.length === 0) {
        return removedAssignments;
    }

    await persistModelSettings(modelName, settings);

    if (settings === modelSettings) {
        renderLabels();

        if (currentLabelName && datasetModal?.style.display === 'flex') {
            await renderDatasetModal();
        }
    }

    if (showNotice) {
        openDatasetCleanupModal(removedAssignments);
    }

    return removedAssignments;
}

function openTrainingValidationModal(validationResult) {
    if (!trainingValidationModal || !trainingValidationList) {
        return;
    }

    const { messages = [], removedAssignments = [] } = validationResult || {};
    trainingValidationList.innerHTML = '';

    removedAssignments.forEach(({ datasetName, labelName, reason }) => {
        const item = document.createElement('div');
        item.classList.add('training-validation-item');

        const message = document.createElement('span');
        message.innerHTML = `Dataset <strong></strong> was removed from label <strong></strong>.`;
        const strongTags = message.querySelectorAll('strong');
        strongTags[0].textContent = datasetName;
        strongTags[1].textContent = labelName;

        const detail = document.createElement('span');
        detail.classList.add('training-validation-detail');
        detail.textContent = reason;

        item.appendChild(message);
        item.appendChild(detail);
        trainingValidationList.appendChild(item);
    });

    messages.forEach((text) => {
        const item = document.createElement('div');
        item.classList.add('training-validation-item');
        item.textContent = text;
        trainingValidationList.appendChild(item);
    });

    trainingValidationModal.style.display = 'flex';
}

function closeTrainingValidationModal() {
    if (!trainingValidationModal) return;
    trainingValidationModal.style.display = 'none';
}

async function validateTrainingReadiness(modelName) {
    const messages = [];
    if (!modelName || !modelSettings) {
        return {
            canTrain: false,
            messages: ['Open a model before training.'],
            removedAssignments: []
        };
    }

    const removedAssignments = await removeInvalidAssignedDatasets(modelName, modelSettings, { showNotice: false });
    if (removedAssignments.length > 0) {
        messages.push('One or more assigned datasets were invalid and have been removed. Review the assignments before training again.');
    }

    const labels = modelSettings?.labels || {};
    const labelEntries = Object.entries(labels);
    const labelsWithValidDatasets = labelEntries.filter(([, datasetEntries]) => (
        datasetEntries
        && typeof datasetEntries === 'object'
        && Object.keys(datasetEntries).length > 0
    ));

    if (labelEntries.length < 2) {
        messages.push('Create at least two labels before training.');
    }

    if (labelsWithValidDatasets.length < 2) {
        messages.push('Assign at least one valid dataset to at least two labels.');
    }

    labelEntries.forEach(([labelName, datasetEntries]) => {
        const datasetCount = datasetEntries && typeof datasetEntries === 'object'
            ? Object.keys(datasetEntries).length
            : 0;

        if (datasetCount === 0) {
            messages.push(`Label "${labelName}" needs at least one valid assigned dataset.`);
        }
    });

    return {
        canTrain: messages.length === 0,
        messages,
        removedAssignments
    };
}


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

async function checkExistingDatasets(modelName, modelNamePath) {
    console.log('Checking datasets for model:', modelName, modelNamePath);

    try {
        const settings = await window.electronAPI.invoke('get-model-settings', modelName);
        console.log('Current model settings:', settings);
        if (!settings.labels) settings.labels = {};
        const removedAssignments = await removeInvalidAssignedDatasets(modelName, settings);

        if (removedAssignments.length > 0) {
            modelSettings = settings;
            renderLabels();

            if (currentLabelName && datasetModal?.style.display === 'flex') {
                await renderDatasetModal();
            }
        }
    } catch (error) {
        console.error('Error checking existing datasets:', error);
    }
}

/**
 * Shows the dataset info view and displays the dataset name.
 * @param {string} datasetName - The name of the dataset to display
 * @param {string} datasetPath - The path of the dataset to display
 */
async function showDataset(datasetName,datasetPath) {
    if (datasetNameDisplay) {
        datasetNameDisplay.textContent = datasetName;
    }
    if (datasetPathDisplay){
        datasetPathDisplay.textContent = await window.electronAPI.invoke('get-dataset-path', datasetPath);
    }
    sessionStorage.setItem('projectName', datasetName);
    sessionStorage.setItem('projectPath', datasetPath);
    showView('view-dataset-info');
    loadCarousel();
}
/**
 * Shows the dataset gallery view and loads the appropriate gallery
 */
function showDatasetGallery() {
    showView('view-dataset-gallery');
    loadGallery();
}

function openImportModal() {
    datasetImportModal.style.display = 'flex';
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

// /**
//  * Opens the help modal dialog.
//  */
// function openHelpModal() {
//     helpModal.style.display= 'flex';
//     projectNameInput.focus();
// }
//
// /**
//  * Closes the help modal dialog.
//  */
// function closeHelpModal() {
//     helpModal.style.display= 'none';
// }
//
// /**
//  * Opens the settings modal dialog.
//  */
// function openSettingsModal() {
//     settingsModal.style.display= 'flex';
// }
//
// /**
//  * Closes the settings modal dialog.
//  */
// function closeSettingsModal() {
//     settingsModal.style.display= 'none';
// }

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

function setSidebarQrExpanded(expanded) {
    if (!sidebarQrPanel || !sidebarQrToggle || !sidebarQrArrow) {
        return;
    }

    sidebarQrPanel.classList.toggle('is-expanded', expanded);
    sidebarQrPanel.classList.toggle('is-collapsed', !expanded);
    sidebarQrPanel.setAttribute('aria-expanded', String(expanded));
    sidebarQrToggle.setAttribute('aria-expanded', String(expanded));
    sidebarQrToggle.setAttribute('aria-label', expanded ? 'Collapse QR code' : 'Expand QR code');
    sidebarQrArrow.setAttribute('aria-hidden', 'true');
}

function toggleSidebarQrPanel() {
    if (!sidebarQrPanel) {
        return;
    }

    setSidebarQrExpanded(sidebarQrPanel.classList.contains('is-collapsed'));
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
 * Opens the image options model
 */
function closeOptionsModal() {
    optionsModal.style.display = 'none';
}

/**
 * Opens the delete dataset confirmation modal for display only.
 */
function openDeleteDatasetModal() {
    const currentDatasetName = sessionStorage.getItem('projectName') || datasetNameDisplay?.textContent || 'this dataset';

    deleteDatasetName.textContent = currentDatasetName;
    deleteDatasetConfirmInput.value = '';
    deleteDatasetModal.style.display = 'flex';
    deleteDatasetConfirmInput.focus();
}

/**
 * Closes the delete dataset confirmation modal.
 */
function closeDeleteDatasetModal() {
    deleteDatasetModal.style.display = 'none';
}

/**
 * Opens the delete model confirmation modal for display only.
 */
function openDeleteModelModal() {
    const currentModelName = sessionStorage.getItem('projectName') || modelNameDisplay?.textContent || 'this model';

    deleteModelName.textContent = currentModelName;
    deleteModelConfirmInput.value = '';
    deleteModelModal.style.display = 'flex';
    deleteModelConfirmInput.focus();
}

/**
 * Closes the delete model confirmation modal.
 */
function closeDeleteModelModal() {
    deleteModelModal.style.display = 'none';
}

function updateBuilderButtonState() {
    const selectedType = document.querySelector('input[name="train-project-type"]:checked')?.value || modelSettings?.projectType || 'scratch';
    const canOpenBuilder = selectedType === 'scratch';

    builderModalBtn.disabled = !canOpenBuilder;
    builderModalBtn.title = canOpenBuilder ? 'Open Editor' : 'Only available when Start From Scratch is selected';
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
        // epochs and labels
        if (!modelSettings.labels) modelSettings.labels = {};
        epochSlider.value = modelSettings.epochs ?? 10;
        epochValueDisplay.textContent = epochSlider.value;
        if (!modelSettings.modelType) modelSettings.modelType = 'tensorflow';

        // clears builder canvas
        builderCanvas.innerHTML = '';
        // adds saved layers
        if(modelSettings.layers && modelSettings.layers.length > 0){
            modelSettings.layers.forEach(layer => {
                addLayerToCanvas(layer.type, layer.parameters);
            });
        } else {
            builderCanvas.innerHTML = '<div class="builder-block-window">Drag Here</div>';
        }

        // loads last trained timestamp
        const lastTrainedSpan = document.querySelector('.last-trained');
        if (lastTrainedSpan) {
            lastTrainedSpan.textContent = `Last Trained: ${modelSettings.lastTrained || 'Never'}`;
        }
        updateLastTrainedFrameworkDisplays();
        syncRuntimeFrameworkToggles();

        if (!modelSettings.projectType) modelSettings.projectType = 'scratch';

        const persistedChartState = normalizeChartState(modelSettings.chartHistory);
        const cachedChartState = chartStateByModel.get(modelName);

        if (isEmptyChartState(cachedChartState)) {
            setChartStateForModel(modelName, persistedChartState);
        }

        activeChartModelName = modelName;
        modelSettings.chartHistory = cloneChartState(getChartStateForModel(modelName));
        renderChartState(modelSettings.chartHistory);
        renderSummaryCard(modelSettings.chartHistory);

        const selectedType = modelSettings.projectType === 'pretrained' ? 'pretrained' : 'scratch';
        const selectedTypeInput = document.querySelector(`input[name="train-project-type"][value="${selectedType}"]`);
        if (selectedTypeInput) {
            selectedTypeInput.checked = true;
        }
        updateBuilderButtonState();

        const selectedFramework = modelSettings.modelType === 'pytorch' ? 'false' : 'true';
        syncFrameworkToggles(selectedFramework);
        await removeInvalidAssignedDatasets(modelName, modelSettings);
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
    const blocks = document.querySelectorAll('.model-block');

    modelSettings.layers = Array.from(blocks).map(block => {
        const params = { ...block.dataset };
        const type = params.type;
        delete params.type;

        return {
            type: type,
            parameters: params
        };
    });

    modelSettings.chartHistory = cloneChartState(getChartStateForModel(modelName));
    await persistModelSettings(modelName, modelSettings);
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
        //replace with pop up modals, screws up text cursor on Windows
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
    await removeInvalidAssignedDatasets(sessionStorage.getItem('projectName'), modelSettings);
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
 * Opens the builder modal.
 */
function openBuilderModal(){
    builderModal.style.display = 'flex';
}

/**
 * Closes the builder modal.
 */
function closeBuilderModal(){
    builderModal.style.display = 'none';
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
        for (const [datasetName, datasetInfo] of available) {
            console.log('Available dataset:', datasetName, datasetInfo);

            const exists = await window.electronAPI.invoke('path-exists', datasetInfo.path);
            console.log(`Dataset path exists for "${datasetName}":`, exists);
            const images = exists ? await window.electronAPI.invoke('get-project-images', datasetInfo.path) : [];
            const imageCount = images.length;
            const disabledReason = !exists
                ? 'Dataset path not found'
                : imageCount < 3
                    ? `Needs at least 3 images (${imageCount} found)`
                    : '';
            console.log(`Number of images in dataset "${datasetName}":`, imageCount);

            const item = document.createElement('div');
            item.classList.add('dataset-item', 'available');
            if (disabledReason) {
                item.classList.add('unavailable');
            }

            const textWrap = document.createElement('div');
            textWrap.classList.add('dataset-item-text');

            const nameSpan = document.createElement('span');
            nameSpan.classList.add('dataset-item-name');
            nameSpan.textContent = datasetName;
            textWrap.appendChild(nameSpan);

            if (disabledReason) {
                const statusSpan = document.createElement('span');
                statusSpan.classList.add('dataset-item-status');
                statusSpan.textContent = disabledReason;
                textWrap.appendChild(statusSpan);
            }

            const addBtn = document.createElement('button');
            addBtn.classList.add('dataset-add-btn');
            addBtn.textContent = '+';
            addBtn.disabled = Boolean(disabledReason);
            addBtn.title = disabledReason || 'Add dataset';
            if (!disabledReason) {
                addBtn.addEventListener('click', () => addDatasetToLabel(datasetName, datasetInfo.path));
            }

            item.appendChild(textWrap);
            item.appendChild(addBtn);
            availableDatasetsList.appendChild(item);
        }
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
    const sidebarMode = selector === '.left-sidebar-datasets' ? 'datasets' : 'models';

    setSidebarCollectionMode(sidebarMode);
    await loadDatasetsFromFolder();
    await loadModelsFromFolder();
}

function setSidebarCollectionMode(mode, options = {}) {
    const { clearSearches = true } = options;
    const showDatasets = mode === 'datasets';

    if (primarySidebar) {
        primarySidebar.classList.add('sidebar-active');
    }

    if (clearSearches) {
        clearSidebarSearches();
    }

    if (modelsList) {
        modelsList.style.display = showDatasets ? 'none' : '';
    }
    if (datasetsList) {
        datasetsList.style.display = showDatasets ? '' : 'none';
    }
    if (modelSearchInput) {
        modelSearchInput.style.display = showDatasets ? 'none' : '';
    }
    if (datasetSearchInput) {
        datasetSearchInput.style.display = showDatasets ? '' : 'none';
    }
    if (addModelBtn) {
        addModelBtn.style.display = showDatasets ? 'none' : '';
    }
    if (datasetImportBtn) {
        datasetImportBtn.style.display = showDatasets ? '' : 'none';
    }

    syncSidebarSegmentedControls(showDatasets);
}

function syncSidebarSegmentedControls(showDatasets) {
    const modelsActive = document.getElementById('sidebar-models-active');
    const modelsDatasets = document.getElementById('sidebar-models-datasets');

    if (!modelsActive || !modelsDatasets) return;

    const applyState = () => {
        modelsActive.checked = !showDatasets;
        modelsDatasets.checked = showDatasets;
    };

    if (sidebarToggleAnimationFrame !== null) {
        cancelAnimationFrame(sidebarToggleAnimationFrame);
    }

    sidebarToggleAnimationFrame = requestAnimationFrame(() => {
        applyState();
        sidebarToggleAnimationFrame = null;
    });
}

function initializeSidebarModeToggle() {
    if (sidebarModelsRadio) {
        sidebarModelsRadio.addEventListener('change', () => {
            if (sidebarModelsRadio.checked) {
                setSidebarCollectionMode('models', { clearSearches: true });
            }
        });
    }

    if (sidebarDatasetsRadio) {
        sidebarDatasetsRadio.addEventListener('change', () => {
            if (sidebarDatasetsRadio.checked) {
                setSidebarCollectionMode('datasets', { clearSearches: true });
            }
        });
    }
}

/**
 * Creates a new model by prompting the user for a name and creating a folder.
 * Uses IPC to communicate with the main process to create the folder.
 */
async function handleAddProject() {
    const modelName = projectNameInput.value.trim();
    const projectType = Array.from(projectTypeInputs).find(input => input.checked)?.value || 'scratch';
    
    if (!modelName) {

        // need to make a new modal for this pop up
        //alert('Model name cannot be empty');
        return;
    }

    try {
        const modelPath = await window.electronAPI.invoke('create-model-folder', 
            { modelName, projectType }
        );
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
            return;
        }

        // Construct the server URL
        const serverURL = `http://${localIP}:3001`;
        console.log('Generating QR code for:', serverURL);

        const renderQRCode = (container, size) => {
            if (!container) {
                return;
            }

            container.innerHTML = '';

            new QRCode(container, {
                text: serverURL,
                width: size,
                height: size,
                colorDark: '#000000',
                colorLight: '#ffffff',
                correctLevel: QRCode.CorrectLevel.H
            });
        };

        renderQRCode(sidebarQrCodeContainer, 160);
        renderQRCode(qrCodeContainer, 180);

        console.log('QR code generated successfully');
    } catch (error) {
        console.error('Error in generateQRCode:', error);
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
                //
                const datasetDisplayName = div.textContent;

                showDataset(datasetDisplayName,datasetPath);
            });
            li.appendChild(div);
            datasetsList.appendChild(li);
        });

        applySearchFilter(datasetSearchInput, datasetsList, '.dataset-item', '.dataset-open-btn');

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
            li.setAttribute('data-model-name', modelName);

            const div = document.createElement('div');
            div.classList.add('model-open-btn');
            div.textContent = modelName;
            li.addEventListener('click', async (e) => {
                e.stopPropagation();
                const modelPath = li.getAttribute('data-path');
                const modelDisplayName = li.getAttribute('data-model-name') || div.textContent;

                console.log('Model item clicked:', modelDisplayName, modelPath);
                await showModel(modelDisplayName,modelPath);
            });
            li.appendChild(div);
            modelsList.appendChild(li);
        });

        applySearchFilter(modelSearchInput, modelsList, '.model-item', '.model-open-btn');

        console.log(`Loaded ${models.length} projects`);
    } catch (error) {
        console.error('Error loading models:', error);
    }
}
/**
 * Loads carousel and attaches images into it
 */
async function loadCarousel() {
    const currentPath = sessionStorage.getItem('projectPath');
    if(currentPath) {
        // Request images
        console.log(currentPath);
        const images = await window.electronAPI.invoke('get-project-images', currentPath);

        // Loop through images and create elements
        carousel.innerHTML = '';

        images.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.loading = 'lazy';
            imgElement.src = imgData.url;
            imgElement.alt = imgData.name;
            imgElement.className = 'carousel-item';

            carousel.appendChild(imgElement);
        });
    }
}

/**
 * Loads gallery and attaches images into it
 */
async function loadGallery(){
    const storedPath = sessionStorage.getItem('projectPath');
    const currentName = sessionStorage.getItem('projectName');

    if(storedPath){
        const currentPath = await window.electronAPI.invoke('get-dataset-path', storedPath);
        galleryNameDisplay.textContent = currentName || 'Dataset';
        galleryPathDisplay.textContent = currentPath;
        galleryContainer.innerHTML = '';
        const images = await window.electronAPI.invoke('get-project-images', currentPath);

        images.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.loading = 'lazy';
            imgElement.src = imgData.url;
            imgElement.alt = imgData.name;
            imgElement.className = 'gallery-item';

            //Add click listener
            imgElement.addEventListener('click', async () => {
                const result = await window.electronAPI.invoke('open-file', imgData.url);
                if (!result) {
                    console.error('Could not open image');
                }
            });

            galleryContainer.appendChild(imgElement);
        });
    }
}


/**
 * Search bar functionality
 */
function applySearchFilter(inputEl, listEl, itemSelector, labelSelector) {
    if (!inputEl || !listEl) {
        return;
    }

    const query = inputEl.value.toLowerCase().trim();

    listEl.querySelectorAll(itemSelector).forEach(item => {
        const labelEl = item.querySelector(labelSelector);
        const label = labelEl ? labelEl.textContent.toLowerCase() : '';
        item.style.display = label.includes(query) ? '' : 'none';
    });
}

/**
 * Initializes  search bar
 */
function initSearch(inputEl, listEl, itemSelector, labelSelector) {
    if (!inputEl || !listEl) {
        return;
    }

    inputEl.addEventListener('input', () => {
        applySearchFilter(inputEl, listEl, itemSelector, labelSelector);
    });
}

/**
 * Clears search bar
 */
function clearSidebarSearches() {
    if (modelSearchInput) {
        modelSearchInput.value = '';
    }
    if (datasetSearchInput) {
        datasetSearchInput.value = '';
    }
    applySearchFilter(modelSearchInput, modelsList, '.model-item', '.model-open-btn');
    applySearchFilter(datasetSearchInput, datasetsList, '.dataset-item', '.dataset-open-btn');
}


// ============================================================
// BUILDER RELATED FUNCTIONS
// ============================================================

function showLayerDescription(type) {
    const description = layerDescriptions[type];
    if (!description || !layerDescriptionTitle || !layerDescriptionText) return;

    layerDescriptionTitle.textContent = description.title;
    layerDescriptionText.textContent = description.text;

    layerItems.forEach(item => {
        item.classList.toggle('is-selected', item.dataset.type === type);
    });
}

/**
 * Adds a layer via a drag and drop system
 * @param type
 * @param savedParams
 */
function addLayerToCanvas(type, savedParams = null) {
    // block creation
    const block = document.createElement('div');
    block.className = 'model-block';
    block.draggable = true;
    block.dataset.type = type;
    block.innerHTML = `<strong>${type}</strong>`;

    // checks for saved parameters if not default
    if (savedParams) {
        Object.assign(block.dataset, savedParams);
    } else {
        if (type === 'Dense') block.dataset.units = 32;
    }

    block.innerHTML = `<strong>${type}</strong>`;

    // show settings
    block.onclick = () => {
        if(activeBlock) activeBlock.classList.remove('is-active');
        activeBlock = block;
        block.classList.add('is-active');
        showLayerDescription(type);
        showSettings(type, block);
    };

    // drag logic
    block.ondragstart = (e) => {
        e.dataTransfer.setData('isNew', 'false');
        block.style.opacity = "0.5";
    };

    // deleting logic
    block.ondragend = async(e) => {
        block.style.opacity = "1";
        const rect = builderCanvas.getBoundingClientRect();
        const isInside = (
            e.clientX >= rect.left && e.clientX <= rect.right &&
            e.clientY >= rect.top && e.clientY <= rect.bottom
        );

        if (!isInside) {
            block.remove();
            document.getElementById('settings-content').innerHTML = '<p class="hint">Layer removed.</p>';
            await saveModelSettings();
        }
    };
    // attaches block
    builderCanvas.appendChild(block);
}

/**
 * Dynamically shows the various parameters of a layer if applicable
 * @param type
 * @param block
 */
function showSettings(type, block) {
    const container = document.getElementById('settings-content');
    let html = `<h4>${type} Configuration</h4>`;

    // the most complex ones
    if (type === 'Dense' || type === 'Conv2D') {
        const label = (type === 'Dense') ? 'Units' : 'Filters';
        html += `<label>${label}:</label>
                 <input type="number" id="set-units" value="${block.dataset.units || (type === 'Dense' ? 128 : 32)}">`;

        html += `<label>Activation:</label>
                 <select id="set-activation">
                    <option value="relu" ${block.dataset.activation === 'relu' ? 'selected' : ''}>ReLU</option>
                    <option value="sigmoid" ${block.dataset.activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                    <option value="tanh" ${block.dataset.activation === 'tanh' ? 'selected' : ''}>Tanh</option>
                 </select>`;
    }

    // Layers that use pooling
    else if (type.includes('Pooling2D') && !type.includes('Global')) {
        html += `<label>Pool Size:</label>
                 <input type="number" id="set-pool" value="${block.dataset.pool_size || 2}">`;
    }
    // Layers that use a float parameter
    else if (['Dropout', 'RandomRotation', 'RandomZoom', 'RandomContrast'].includes(type)) {
        const label = (type === 'Dropout') ? 'Rate (0-1)' : 'Factor (0-1)';
        const key = (type === 'Dropout') ? 'rate' : 'factor';
        html += `<label>${label}:</label>
                 <input type="number" step="0.1" id="set-factor" value="${block.dataset[key] || 0.2}">`;
    }

    // Layers that just need a dropdown
    else if (type === 'RandomFlip') {
        html += `<label>Mode:</label>
                 <select id="set-flip">
                    <option value="horizontal" ${block.dataset.mode === 'horizontal' ? 'selected' : ''}>Horizontal</option>
                    <option value="vertical" ${block.dataset.mode === 'vertical' ? 'selected' : ''}>Vertical</option>
                    <option value="horizontal_and_vertical" ${block.dataset.mode === 'horizontal_and_vertical' ? 'selected' : ''}>Both</option>
                 </select>`;
    } else {
        html += `<p>No modifiable parameters.</p>`;
    }

    html += `<button id="apply-btn" class="train-btn" style="margin-top:10px">Apply</button>`;
    container.innerHTML = html;

    document.getElementById('apply-btn').onclick = async () => {
        const units = document.getElementById('set-units');
        const act = document.getElementById('set-activation');
        const factor = document.getElementById('set-factor');
        const pool = document.getElementById('set-pool');
        const flip = document.getElementById('set-flip');

        if (units) block.dataset.units = units.value;
        if (act) block.dataset.activation = act.value;
        if (pool) block.dataset.pool_size = pool.value;
        if (flip) block.dataset.mode = flip.value;
        if (factor) {
            if (type === 'Dropout') block.dataset.rate = factor.value;
            else block.dataset.factor = factor.value;
        }
        await saveModelSettings();
        alert("Settings Saved!");
    };
}

/**
 * Turns the ez block builder mode into the JSON advanced mode
 */
function syncBlocksToCode() {
    const blocks = document.querySelectorAll('.model-block');
    const layers = Array.from(blocks).map(block => ({
        type: block.dataset.type,
        parameters: { ...block.dataset }
    }));
    layers.forEach(l => delete l.parameters.type);

    editor.setValue(JSON.stringify(layers, null, 2));
}

/**
 * Turns the advanced mode layers into ez mode blocks
 */
function syncCodeToBlocks() {
    try {
        const layers = JSON.parse(editor.getValue());
        const canvas = document.getElementById('builder-canvas');
        canvas.innerHTML = '';

        layers.forEach(layer => {
            addLayerToCanvas(layer.type, layer.parameters);
        });
    } catch (e) {
        //alert("INVALID.");
    }
}

function layoutCodeEditor() {
    if (!editor) return;
    requestAnimationFrame(() => {
        editor.layout();
        requestAnimationFrame(() => editor.layout());
    });
}

/**
 * Switches the tab that the model page is currently on
 * @param event
 * @param tabId
 */
function openTab(event, tabId) {
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => {
        tab.classList.remove('active');
    });

    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => btn.classList.remove('active'));

    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');

    if (tabId === 'tab-train') {
        requestChartResize();
    }
}

function requestChartResize() {
    if (!accuracyChart || !lossChart) return;

    if (chartResizeFrame) {
        cancelAnimationFrame(chartResizeFrame);
    }

    chartResizeFrame = requestAnimationFrame(() => {
        chartResizeFrame = null;
        accuracyChart.resize();
        lossChart.resize();
    });
}

function setupChartResizeSync() {
    if (chartResizeObserver) {
        chartResizeObserver.disconnect();
        chartResizeObserver = null;
    }

    const chartContainers = document.querySelectorAll('.charts-container, .chart-wrapper');
    if (typeof ResizeObserver === 'undefined' || chartContainers.length === 0) {
        return;
    }

    chartResizeObserver = new ResizeObserver(() => {
        requestChartResize();
    });

    chartContainers.forEach((container) => {
        chartResizeObserver.observe(container);
    });

    window.addEventListener('resize', requestChartResize);
    document.addEventListener('fullscreenchange', requestChartResize);
}

function closeAddDatasetModal(){
    datasetImportModal.style.display= 'none';
}

// ============================================================
// Event Listeners
// ============================================================

// Upload Button - opens QR modal
if (uploadBtn) {
    uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        openQRModal();
    });
}

// Sidebar QR toggle button
if (sidebarQrToggle) {
    sidebarQrToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleSidebarQrPanel();
    });
}

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

// Import Button - open modal to import new data
if (datasetImportBtn) {
    datasetImportBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        openImportModal();
    });
}

// // Open Help modal
// helpBtn.addEventListener('click', (e) => {
//     e.stopPropagation();
//     openHelpModal();
// })
//
// // Help Modal Close button (X) - close modal
// helpModalClose.addEventListener('click', (e) => {
//     e.stopPropagation();
//     closeHelpModal();
// });
//
// // Open Settings modal
// settingsBtn.addEventListener('click', (e) => {
//     e.stopPropagation();
//     openSettingsModal();
// })
//
// // Close Settings modal
// settingsModalClose.addEventListener('click', (e) => {
//     e.stopPropagation();
//     closeSettingsModal();
// })

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
modelModalClose.addEventListener('click', (e) => {
    e.stopPropagation();
    closeAddProjectModal();
});

closeCreateDatasetModal.addEventListener('click', (e) =>{
    e.stopPropagation();
    closeAddDatasetModal();
});

datasetImportModal.addEventListener('click', (e) =>{
    e.stopPropagation();
    if(e.target ===  datasetImportModal){
        closeAddDatasetModal();
    }
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

// Open Block Builder Modal
builderModalBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (builderModalBtn.disabled) return;
    openBuilderModal();
})
// Close Block Builder Modal
builderModalCloseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    closeBuilderModal();
})

// adds dragging and layer logic to layer items
layerItems.forEach(item => {
    item.setAttribute('role', 'button');
    item.tabIndex = 0;

    item.addEventListener('click', () => {
        showLayerDescription(item.dataset.type);
    });

    item.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        e.preventDefault();
        showLayerDescription(item.dataset.type);
    });

    item.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('layerType', e.target.getAttribute('data-type'));
        e.dataTransfer.setData('isNew', 'true');
    });
});

// Implements drag
builderCanvas.addEventListener('dragover', (e) => e.preventDefault());

// Implements drop
builderCanvas.addEventListener('drop', async(e) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('layerType');
    const isNew = e.dataTransfer.getData('isNew') === 'true';

    if (isNew) {
        addLayerToCanvas(type);
        await saveModelSettings();
    }
});

// Switch to "code" mode for sequential model editor
viewCodeBtn.addEventListener('click', () => {
    syncBlocksToCode();
    document.getElementById('visual-workspace').style.display = 'none';
    document.getElementById('code-workspace').style.display = 'flex';
    viewBlockBtn.classList.remove('active');
    viewCodeBtn.classList.add('active');
    layoutCodeEditor();
});

// Switch to "block" mode for sequential model editor
viewBlockBtn.addEventListener('click', () => {
    syncCodeToBlocks();
    document.getElementById('code-workspace').style.display = 'none';
    document.getElementById('visual-workspace').style.display = 'flex';
    viewCodeBtn.classList.remove('active');
    viewBlockBtn.classList.add('active');
});

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

// Project type radio buttons change in training menu
trainProjectTypeInputs.forEach((input) => {
  input.addEventListener('change', async () => {
    if (!input.checked) return;
    updateBuilderButtonState();
    if (!modelSettings) return;
    modelSettings.projectType = input.value;
    await saveModelSettings();
  });
});

trainFrameworkInputs.forEach((input) => {
  input.addEventListener('change', async () => {
    if (!input.checked) return;
    syncFrameworkToggles(input.value);
    if (!modelSettings) return;
    modelSettings.modelType = input.value === 'false' ? 'pytorch' : 'tensorflow';
    await saveModelSettings();
  });
});


// carousel animation
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

// Dataset cleanup modal close buttons
if (datasetCleanupModalClose) {
    datasetCleanupModalClose.addEventListener('click', closeDatasetCleanupModal);
}
if (datasetCleanupOkBtn) {
    datasetCleanupOkBtn.addEventListener('click', closeDatasetCleanupModal);
}

// Dataset cleanup modal backdrop click
if (datasetCleanupModal) {
    datasetCleanupModal.addEventListener('click', (e) => {
        if (e.target === datasetCleanupModal) closeDatasetCleanupModal();
    });
}

// Training validation modal close buttons
if (trainingValidationModalClose) {
    trainingValidationModalClose.addEventListener('click', closeTrainingValidationModal);
}
if (trainingValidationOkBtn) {
    trainingValidationOkBtn.addEventListener('click', closeTrainingValidationModal);
}

// Training validation modal backdrop click
if (trainingValidationModal) {
    trainingValidationModal.addEventListener('click', (e) => {
        if (e.target === trainingValidationModal) closeTrainingValidationModal();
    });
}

// Delete dataset modal open button
deleteDatasetBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    openDeleteDatasetModal();
});

// Delete dataset modal close buttons
deleteDatasetModalClose.addEventListener('click', closeDeleteDatasetModal);
deleteDatasetCancelBtn.addEventListener('click', closeDeleteDatasetModal);

// Delete dataset modal backdrop click
deleteDatasetModal.addEventListener('click', (e) => {
    if (e.target === deleteDatasetModal) closeDeleteDatasetModal();
});

// Delete model modal open button
deleteModelBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    openDeleteModelModal();
});

// Delete model modal close buttons
deleteModelModalClose.addEventListener('click', closeDeleteModelModal);
deleteModelCancelBtn.addEventListener('click', closeDeleteModelModal);

// Delete model modal backdrop click
deleteModelModal.addEventListener('click', (e) => {
    if (e.target === deleteModelModal) closeDeleteModelModal();
});

// Train Model button click handler
modelTrainBtn.addEventListener('click', async () => {
    const epochs = epochSlider.value;
    const currentModelName = sessionStorage.getItem('projectName');

    modelTrainBtn.disabled = true;
    modelTrainBtn.textContent = 'Checking datasets...';

    try {
        const validationResult = await validateTrainingReadiness(currentModelName);
        if (!validationResult.canTrain) {
            openTrainingValidationModal(validationResult);
            return;
        }
    } catch (error) {
        console.error('Error validating training datasets:', error);
        openTrainingValidationModal({
            messages: ['Could not validate dataset assignments. Check the console for details.'],
            removedAssignments: []
        });
        return;
    } finally {
        modelTrainBtn.disabled = false;
        modelTrainBtn.textContent = 'Train Model';
    }

    activeTrainingModelName = currentModelName;
    activeChartModelName = currentModelName;

    const activeRunState = createEmptyChartState();
    setChartStateForModel(currentModelName, activeRunState);
    if (modelSettings) {
        modelSettings.chartHistory = cloneChartState(activeRunState);
    }
    renderChartState(activeRunState);

    const toggle = getSelectedFramework();
    console.log(`[UI] : ${toggle}`);
    console.log(`[UI] Train button clicked for model "${currentModelName}" with ${epochs} epochs and toggle "${toggle}"`);
    console.log(`%c[UI] Training started with ${epochs} epochs!`, 'color: #007acc; font-weight: bold;');

    try {
        modelTrainBtn.disabled = true;
        modelTrainBtn.textContent = 'Training in progress...';
        // modelTrainResults.textContent = 'Training in progress...';
        // modelTrainResults.style.color = '#FFA500';

        const callStartTime = Date.now();

        const { labelsJson, modelFolderPath } = await window.electronAPI.invoke('get-model-details-for-python', currentModelName);

        const projectType = modelSettings?.projectType || 'scratch';
        if (modelSettings) {
            modelSettings.modelType = toggle;
            await saveModelSettings();
        }

        const result = await window.electronAPI.executeTrain({
            labelsJson,
            modelFolderPath,
            epochs: parseInt(epochs),
            toggle: toggle,
            layers: modelSettings.layers,
            projectType: projectType
        });

        const callDuration = Math.round((Date.now() - callStartTime) / 1000);
        console.log(`%c[UI] IPC handler completed in ${callDuration}s`, 'color: #007acc; font-weight: bold;');

        if (result.success) {
            const execTime = result.executionTime ? ` (${result.executionTime}s)` : '';
            // modelTrainResults.textContent = `Training completed successfully!${execTime}`;
            // modelTrainResults.style.color = '#28a745';
            console.log('%c[UI] Training successful!', 'color: #28a745; font-weight: bold;');
            document.getElementById('epoch-progress-fill').style.width = `100%`;
            document.getElementById('progress-label').textContent = `Overall Progress: 100%`;

            const summaryCard = document.getElementById('final-summary-card');
            summaryCard.style.display = 'block';

            const completedChartState = getChartStateForModel(currentModelName);
            renderSummaryCard(completedChartState);

            // timestamp stuff
            const now = new Date();
            const timestamp = now.toLocaleString();

            const lastTrainedSpan = document.querySelector('.last-trained');
            if (lastTrainedSpan) {
                lastTrainedSpan.textContent = `Last Trained: ${timestamp}`;
            }
            // save to JSON
            if(modelSettings){
                modelSettings.chartHistory = cloneChartState(completedChartState);
                modelSettings.lastTrained = timestamp;
                modelSettings.lastTrainedFramework = toggle;
                updateLastTrainedFrameworkDisplays();
                syncRuntimeFrameworkToggles(toggle);
                await persistModelSettings(currentModelName, modelSettings);
            }

        } else {
            const execTime = result.executionTime ? ` (${result.executionTime}s)` : '';
            // modelTrainResults.textContent = `Training failed.${execTime} Check console for details.`;
            // modelTrainResults.style.color = '#dc3545';
            console.error('%c[UI] Training failed!', 'color: #dc3545; font-weight: bold;');
            console.error('[UI] Error:', result.error);
        }
    } catch (error) {
        console.error('%c[UI] IPC error:', 'color: #dc3545; font-weight: bold;', error);
        // modelTrainResults.textContent = `IPC Error: ${error.message}`;
        // modelTrainResults.style.color = '#dc3545';
    } finally {
        activeTrainingModelName = null;
        modelTrainBtn.disabled = false;
        modelTrainBtn.textContent = 'Train Model';
    }
});

// Test button in testing tab of a model
runTestButton.addEventListener('click',async ()=> {
    const modelName = sessionStorage.getItem('projectName');
    const testMessage = document.getElementById('test-results-message');
    const testAccuracyVal = document.getElementById('test-accuracy-val');
    const testLossVal = document.getElementById('test-loss-val');
    const testCountVal = document.getElementById('test-count-val');

    if (testMessage) {
        testMessage.textContent = 'Running evaluation...';
    }

    const result = await window.electronAPI.invoke('test-model', {
        modelName: modelName,
        modelType: getSelectedRuntimeFramework('test')
    });
    if (result.success) {
        console.log(result);
        if (testAccuracyVal) testAccuracyVal.textContent = (result.accuracy * 100).toFixed(2) + "%";
        if (testLossVal) testLossVal.textContent = result.loss.toFixed(4);
        if (testCountVal) testCountVal.textContent = result.total_images;
        if (testMessage) {
            testMessage.textContent = result.model_classes && result.dataset_classes
                ? `Evaluated ${result.dataset_classes} labels against a ${result.model_classes}-class model.`
                : '';
        }
    } else {
        const errorMessage = result.error || 'Evaluation failed.';
        if (testAccuracyVal) testAccuracyVal.textContent = 'Error';
        if (testLossVal) testLossVal.textContent = '—';
        if (testCountVal) testCountVal.textContent = '—';
        if (testMessage) testMessage.textContent = errorMessage;
        console.error(errorMessage);
    }
})

// ============================================================
// Initialization
// ============================================================

// Load projects from folder when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    initSearch(modelSearchInput, modelsList, '.model-item', '.model-open-btn');
    initSearch(datasetSearchInput, datasetsList, '.dataset-item', '.dataset-open-btn');
    initializeSidebarModeToggle();
    switchMode('view-home','.left-sidebar-models');
    setSidebarQrExpanded(true);

    await loadModelsFromFolder();
    await generateQRCode();

    // Initialize and setup charts
    initCharts();
    setupChartResizeSync();
    requestChartResize();
    const epochRegex = /epoch\s*=\s*(\d+)\s*train_loss\s*=\s*([\d.]+)\s*train_acc\s*=\s*([\d.]+)\s*val_loss\s*=\s*([\d.]+)\s*val_acc\s*=\s*([\d.]+)/i;
    let stdoutBuffer = '';
    let stderrBuffer = '';

    const processTrainingChunk = (chunk, source) => {
        const normalizedChunk = chunk.replace(/\r/g, '\n');
        const nextBuffer = (source === 'stderr' ? stderrBuffer : stdoutBuffer) + normalizedChunk;
        const lines = nextBuffer.split('\n');
        const remainder = lines.pop() ?? '';

        if (source === 'stderr') {
            stderrBuffer = remainder;
        } else {
            stdoutBuffer = remainder;
        }

        lines.forEach((line) => {
            const trimmedLine = line.trim();
            if (!trimmedLine) return;

            if (source === 'stderr') {
                console.error(`[Python stderr] ${trimmedLine}`);
            } else {
                console.log(`[Python stdout] ${trimmedLine}`);
            }

            const match = trimmedLine.match(epochRegex);
            if (match && activeTrainingModelName) {
                const [_, epoch, tLoss, tAcc, vLoss, vAcc] = match;
                const eLabel = `E${epoch}`;

                const chartState = getChartStateForModel(activeTrainingModelName);
                chartState.labels.push(eLabel);
                chartState.accuracy.train.push(parseFloat(tAcc));
                chartState.accuracy.val.push(parseFloat(vAcc));
                chartState.loss.train.push(parseFloat(tLoss));
                chartState.loss.val.push(parseFloat(vLoss));

                if (activeChartModelName === activeTrainingModelName) {
                    renderChartState(chartState);
                }

                const currentEpoch = parseInt(epoch);
                const totalEpochs = parseInt(epochSlider.value);
                const percent = Math.round((currentEpoch / totalEpochs) * 100);

                document.getElementById('epoch-progress-fill').style.width = `${percent}%`;
                document.getElementById('progress-label').textContent = `Overall Progress: ${percent}% (Epoch ${currentEpoch}/${totalEpochs})`;
            }
        });
    };

    if (window.electronAPI?.onTrainingStdout) {
        window.electronAPI.onTrainingStdout((data) => processTrainingChunk(data, 'stdout'));
    }

    if (window.electronAPI?.onTrainingStderr) {
        window.electronAPI.onTrainingStderr((data) => processTrainingChunk(data, 'stderr'));
    }

});


window.electronAPI.invoke('setup-python-venv')

// Load monaco editor
let editor;
require.config({
    paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs'}
});

//
require(['vs/editor/editor.main'], function() {
    editor = monaco.editor.create(document.getElementById('monaco-editor-container'), {
        value: "[]",
        language: 'json',
        theme: 'vs-light',
        automaticLayout: true
    });
});


// Implements Chart JS need to reorg
function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        animation: true,
    };
    accuracyChart = new Chart(document.getElementById('accuracyChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Train Acc', data: [], borderColor: '#28a745', tension: 0.1 },
                { label: 'Val Acc', data: [], borderColor: '#17a2b8', tension: 0.1 }
            ]
        },
        options: commonOptions
    });

    lossChart = new Chart(document.getElementById('lossChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Train Loss', data: [], borderColor: '#dc3545', tension: 0.1 },
                { label: 'Val Loss', data: [], borderColor: '#ffc107', tension: 0.1 }
            ]
        },
        options: commonOptions
    });

    if (activeChartModelName) {
        renderChartState(getChartStateForModel(activeChartModelName));
    }
}

// Prediction Tab Logic (definetly need to reorganize into different files after this PR) - Actually leaving it here because the breakup into different JS files will be done as part of the polish phase


['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, e => { e.preventDefault(); e.stopPropagation(); });
});

dropZone.addEventListener('dragover', () => dropZone.classList.add('drag-over'));
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (e) => {
    dropZone.classList.remove('drag-over');

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];

        const filePath = window.electronAPI.getPathForFile(file);

        console.log("Real File Path found:", filePath);

        if (file.type.startsWith('image/')) {
            processPrediction(filePath);
        }
    }
});

const predictFileInput = document.getElementById('predict-file-input');

// delete
dropZone.addEventListener('click', () => {
    if (predictionPreview.style.display === 'none') {
        predictFileInput.click();
    }
});


predictFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
        const file = e.target.files[0];
        const filePath = window.electronAPI.getPathForFile(file);
        processPrediction(filePath);
    }
});

function resetPredictionZone() {
    // clears area
    predictionPreview.src = "";
    predictionPreview.style.display = 'none';
    document.querySelector('.drop-zone-content').style.display = 'block';


    resultsArea.style.display = 'none';
    predictFileInput.value = "";
}
predictionPreview.addEventListener('click', (e) => {
    e.stopPropagation();
    resetPredictionZone();
});

async function processPrediction(imagePath) {
    predictionPreview.src = `file://${imagePath}`;
    predictionPreview.style.display = 'block';
    document.querySelector('.drop-zone-content').style.display = 'none';

    resultsArea.style.display = 'block';
    document.getElementById('predicted-label').textContent = "Analyzing...";
    document.getElementById('confidence-fill').style.width = '0%';

    const currentModelName = sessionStorage.getItem('projectName');
    const result = await window.electronAPI.invoke('predict-image', {
        modelName: currentModelName,
        imagePath: imagePath,
        modelType: getSelectedRuntimeFramework('prediction')
    });


    if (result.success) {
        const labelKeys = Object.keys(modelSettings?.labels || {});
        const classIdx = result.label.includes('Class') ? result.label.split(' ')[1] : null;
        const displayLabel = (classIdx !== null && labelKeys[classIdx]) ? labelKeys[classIdx] : result.label;
        document.getElementById('predicted-label').textContent = displayLabel;

        const confidencePct = (result.confidence * 100).toFixed(2);
        document.getElementById('confidence-fill').style.width = `${confidencePct}%`;
        document.getElementById('confidence-text').textContent = `Model Confidence: ${confidencePct}%`;
    } else {
        document.getElementById('predicted-label').textContent = "Error during prediction";
        console.error(result.error);
    }
}

const importVideoBtn = document.getElementById('import-video-btn');
const importFolderBtn = document.getElementById('import-folder-btn');
const datasetNameInput = document.getElementById('dataset-name-input');


importVideoBtn.addEventListener('click', async () => {
    const datasetName = datasetNameInput?.value.trim();

    if (!datasetName) {
        return;
    }

    try {
        const datasetPath = await window.electronAPI.invoke('create-dataset-folder', datasetName);
        const conversionResult = await window.electronAPI.invoke('convert-video', datasetPath);

        if (conversionResult !== null) {
            datasetNameInput.value = '';
            datasetImportModal.style.display = 'none';
            await loadDatasetsFromFolder();
        }
    } catch (error) {
        console.error('Error importing video dataset:', error);
    }
});


importFolderBtn.addEventListener('click', async () => {
    const datasetName = datasetNameInput?.value.trim();
    if (!datasetName) {
        return;
    }
    try {
        const selectedFolder = await window.electronAPI.invoke('select-folder');
        if (!selectedFolder) {
            return;
        }

        const datasetPath = await window.electronAPI.invoke('create-dataset-reference', {
            datasetName,
            sourcePath: selectedFolder
        });

        if (datasetPath) {
            datasetNameInput.value = '';
            datasetImportModal.style.display = 'none';
            await loadDatasetsFromFolder();
        }
    } catch (error) {
        console.error('Error importing folder dataset:', error);
    }
});



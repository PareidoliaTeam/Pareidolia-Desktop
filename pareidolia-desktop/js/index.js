// ============================================================
// Query Selectors
// ============================================================
const addProjectBtn = document.querySelector('.add-project-btn');
const projectOpenBtns = document.querySelectorAll('.project-open-btn');
const rightSidebar = document.querySelector('.right-sidebar');
const contentArea = document.querySelector('.content');
const homeView = document.getElementById('home');
const projectView = document.getElementById('project-view');
const projectNameDisplay = document.getElementById('project-name');
const gridBtns = document.querySelectorAll('.project-grid-btn');
const projectsList = document.querySelector('.projects-list');
const qrCodeContainer = document.getElementById('qr-code-container');

// Modal elements
const addProjectModal = document.getElementById('add-project-modal');
const projectNameInput = document.getElementById('project-name-input');
const modalCreateBtn = document.getElementById('modal-create-btn');
const modalCancelBtn = document.getElementById('modal-cancel-btn');
const modalClose = document.querySelector('.modal-close');

// ============================================================
// Functions
// ============================================================

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
 * Shows the home view and hides the project view.
 * Works by adding/removing the 'active' class on the respective elements.
 * Used by listeners for Sidebar clicks (outside buttons) below.
 */
function showHome() {
  homeView.classList.add('active');
  projectView.classList.remove('active');
}
/**
 * Shows the project view and displays the project name.
 * Inverse of showHome() - hides home view and adds 'active' class to project view.
 * @param {string} projectName - The name of the project to display
 */
function showProject(projectName) {
  homeView.classList.remove('active');
  projectView.classList.add('active');
  projectNameDisplay.textContent = projectName;
}

/**
 * Creates a new project by prompting the user for a name and creating a folder.
 * Uses IPC to communicate with the main process to create the folder.
 */
async function handleAddProject() {
  const projectName = projectNameInput.value.trim();
  
  if (!projectName) {
    alert('Project name cannot be empty');
    return;
  }

  try {
    const projectPath = await window.electronAPI.invoke('create-project-folder', projectName);
    console.log('Project created at:', projectPath);
    
    // Reset input and close modal
    projectNameInput.value = '';
    closeAddProjectModal();

    
    // Reload the projects list
    await loadProjectsFromFolder();
  } catch (error) {
    console.error('Error creating project:', error);
  }
}

/**
 * Opens the add project modal dialog.
 */
function openAddProjectModal() {
  addProjectModal.classList.add('show');
  projectNameInput.focus();
}

/**
 * Closes the add project modal dialog.
 */
function closeAddProjectModal() {
  addProjectModal.classList.remove('show');
  projectNameInput.value = '';
}

/**
 * Loads all project folders from the Pareidolia folder and creates buttons for them.
 * Each button has the folder path as its value.
 */
async function loadProjectsFromFolder() {
  try {
    // First ensure the Pareidolia folder exists
    const pareidoliaPath = await window.electronAPI.invoke('get-pareidolia-path');
    console.log('Pareidolia path:', pareidoliaPath);

    // Clear existing project buttons
    projectsList.innerHTML = '';

    // Call a new IPC handler to get the list of datasets
    const projects = await window.electronAPI.invoke('get-datasets-list');
    
    // Create buttons for each project
    projects.forEach(projectInfo => {
      const li = document.createElement('li');
      li.classList.add('project-card');
      
      const button = document.createElement('button');
      button.classList.add('project-open-btn');
      button.value = projectInfo.path;
      button.textContent = projectInfo.name;
      button.addEventListener('click', (e) => {
        e.stopPropagation();
        const projectPath = button.getAttribute('value');
        const projectDisplayName = button.textContent;
        showProject(projectDisplayName);
        sessionStorage.setItem('projectPath', projectPath);
      });

      li.appendChild(button);
      projectsList.appendChild(li);
    });

    console.log(`Loaded ${projects.length} projects`);
  } catch (error) {
    console.error('Error loading projects:', error);
  }
}


// ============================================================
// Event Listeners
// ============================================================


// ========== Sidebar Buttons Navigation ==========

// Add Project button - open modal to create new project
addProjectBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  openAddProjectModal();
});

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

// Sidebar click (outside buttons) - return to home
rightSidebar.addEventListener('click', (e) => {
  if (!e.target.classList.contains('project-open-btn') && !e.target.classList.contains('add-project-btn')) {
    showHome();
  }
});

// ========== Grid Buttons Navigation ==========

// Main Page Grid buttons click - navigate to respective pages
gridBtns.forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const page = btn.getAttribute('data-page');
    const projectName = projectNameDisplay.textContent;
    const buttonValue = btn.getAttribute('value');
    const projectPath = sessionStorage.getItem('projectPath') || '';
    
    if (page) {
      // Store project name, button value, and project path in sessionStorage for the next page
      sessionStorage.setItem('projectName', projectName);
      sessionStorage.setItem('pageValue', buttonValue);
      sessionStorage.setItem('projectPath', projectPath);
      // Redirect to the page
      window.location.href = `${page}.html`;
    }
  });
});

// ============================================================
// Initialization
// ============================================================

// Load projects from folder when the page loads
document.addEventListener('DOMContentLoaded', async () => {
  // Set background images for grid buttons from data-image attribute
  gridBtns.forEach(btn => {
    const imageUrl = btn.getAttribute('data-image');
    if (imageUrl && imageUrl.trim() !== '') {
      btn.style.backgroundImage = `url('${imageUrl}')`;
    }
  });

  await loadProjectsFromFolder();
  await generateQRCode();
});


window.electronAPI.invoke('setup-python-venv')
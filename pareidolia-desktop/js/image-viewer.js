// ensures page is loaded
document.addEventListener("DOMContentLoaded", async() => {
    // Get project name, page value, and project path from sessionStorage
    const projectName = sessionStorage.getItem('projectName') || 'Project';
    const pageValue = sessionStorage.getItem('pageValue') || 'Page';
    const projectPath = sessionStorage.getItem('projectPath') || 'No path';

    // Display in navbar
    document.getElementById('navbar-name').textContent = projectName;
    document.getElementById('navbar-path').textContent = projectPath;

    // Log the project path for verification
    console.log('Project Name:', projectName);
    console.log('Page Value:', pageValue);
    console.log('Project Path:', projectPath);

    const galleryContainer = document.querySelector('.gallery-grid');

    if(projectPath) {
        // Request images
        const images = await window.electronAPI.invoke('get-project-images', projectPath + "/positives");

        // Loop through images and create elements
        images.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.src = imgData.url;
            imgElement.alt = imgData.name;
            imgElement.className = 'gallery-item';

            // Add click listener
            imgElement.addEventListener('click', () => {
                openOptionsModal(imgData);
            });

            galleryContainer.appendChild(imgElement);
        });
    }

    // Modal Logic
    const optionsModal = document.getElementById('options-modal');
    const modalImg = document.getElementById('modal-preview-img');
    const modalTitle = document.getElementById('modal-image-name');

    function openOptionsModal(imgData) {
        optionsModal.classList.add('show');
        modalImg.src = imgData.url;
        modalTitle.textContent = imgData.name;
        // store path for delete option
        optionsModal.dataset.currentPath = imgData.url;
    }

    window.addEventListener('click', (event) => {
        if (event.target == optionsModal) {
            optionsModal.classList.remove('show');
        }
    });

    // Close
    document.getElementById('close-options').addEventListener('click', () => {
        if (event.target == optionsModal) optionsModal.classList.remove('show');
    })
});
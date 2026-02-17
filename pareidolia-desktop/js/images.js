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

// Modal Logic
const modal = document.getElementById('qr-modal');
const importBtn = document.getElementById('import-btn');
const closeBtn = document.getElementById('close-modal');
const qrContainer = document.getElementById('qr-container');
const modalProjectName = document.getElementById('modal-project-name');

function openQRModal() {
    modal.classList.add('show');
    modalProjectName.textContent = projectName;

    // Generate a sample QR code using qrserver.com
    // In a real app, this would be a link to a mobile upload page
    const qrContent = `PareidoliaProject:${projectName}:${projectPath}`;
    const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=${encodeURIComponent(qrContent)}`;

    qrContainer.innerHTML = `<img src="${qrUrl}" alt="QR Code">`;
}

importBtn.addEventListener('click', openQRModal);

closeBtn.addEventListener('click', () => {
    modal.classList.remove('show');
});

window.addEventListener('click', (event) => {
    if (event.target == modal) {
        modal.classList.remove('show');
    }
});

// Handle carousel looping
const carousel = document.querySelector('.carousel');
const carouselWrapper = document.querySelector('.carousel-wrapper');

carousel.addEventListener('animationiteration', () => {
    // Reset carousel position for seamless looping
    carousel.style.animation = 'none';
    setTimeout(() => {
        carousel.style.animation = '';
    }, 10);
});

// Open image gallery page
const viewBtn = document.getElementById('view-btn');

viewBtn.addEventListener('click', (e)=> {
    e.stopPropagation();
    const page = viewBtn.getAttribute('data-page');


    if(page) {
        sessionStorage.setItem('projectName', projectName);
        sessionStorage.setItem('projectPath', projectPath);
        // redirect to page
        window.location.href = `${page}.html`;
    }
});

// Adding images to carousel
document.addEventListener("DOMContentLoaded", async() => {
    if(projectPath) {
        // Request images
        const images = await window.electronAPI.invoke('get-project-images', projectPath + "/positives");

        // Loop through images and create elements
        images.forEach(imgData => {
            const imgElement = document.createElement('img');
            imgElement.src = imgData.url;
            imgElement.alt = imgData.name;
            imgElement.className = 'carousel-item';

            carousel.appendChild(imgElement);
        });
    }
});

// Debug manual import
const manualImportBtn = document.getElementById('manual-import');
manualImportBtn.addEventListener('click', async () => {
    // 1. Wait for the user to pick a file
    console.log(projectPath);
    const filePath = await window.electronAPI.invoke('convert-video',projectPath + "/positives");
});

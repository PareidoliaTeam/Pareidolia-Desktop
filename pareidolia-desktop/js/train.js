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

console.log('Project Name:', projectName);
console.log('Project Path:', projectPath);

// Update epoch display when slider changes
epochSlider.addEventListener('input', () => {
    epochDisplay.textContent = epochSlider.value;
});

// Train button click handler
// Will do actual training in the future, however currently it errors out
// issue will be fixed once virtual envoirments and package management is implemented
trainBtn.addEventListener('click', async () => {
    const epochs = epochSlider.value;
    console.log(`Training started with ${epochs} epochs!`);
    
    /* 
    Code to call Python script via IPC handler
    */
    try {
        // Disable button during execution
        trainBtn.disabled = true;
        //trainBtn.textContent = 'Training...';
        
        /* ------------------------------------------------
         Call the Python script via IPC with project path and epochs
         retuns an object with success flag and output/error message
        --------------------------------------------------- */
        const result = await window.electronAPI.executeTrain({
            projectPath: projectPath,
            epochs: parseInt(epochs)
        });
        
        // Update UI with result
        const randomNumberDisplay = document.getElementById('random-number');
        
        if (result.success) {
            //randomNumberDisplay.textContent = result.output;
            randomNumberDisplay.textContent = 'Training completed successfully!';
            randomNumberDisplay.style.color = '#28a745';
            console.log('Training successful:', result.output);
        } else {
            //randomNumberDisplay.textContent = `Error: ${result.error}`;
            randomNumberDisplay.textContent = 'Training failed. Check console for details.';
            randomNumberDisplay.style.color = '#dc3545';
            console.error('Training failed:', result.error);
        }
    } catch (error) {
        // errr handling if for some reason Electron or Python do not run successfully
        console.error('IPC error:', error);
        document.getElementById('random-number').textContent = `IPC Error: ${error.message}`;
        document.getElementById('random-number').style.color = '#dc3545';
    } finally {
        // runs regardless of success or failure to ensure button is re-enabled
        // Re-enable button
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train';
    }
});


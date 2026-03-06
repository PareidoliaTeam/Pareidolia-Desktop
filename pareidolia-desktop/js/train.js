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
trainBtn.addEventListener('click', async () => {
    const epochs = epochSlider.value;
    console.log(`%c[UI] Training started with ${epochs} epochs!`, 'color: #007acc; font-weight: bold;');
    console.log(`%c[UI] Project Path: ${projectPath}`, 'color: #007acc;');
    
    try {
        // Disable button during execution
        trainBtn.disabled = true;
        trainBtn.textContent = 'Training in progress...';
        
        const randomNumberDisplay = document.getElementById('random-number');
        randomNumberDisplay.textContent = 'Training in progress...';
        randomNumberDisplay.style.color = '#FFA500';
        
        console.log('%c[UI] Calling IPC handler: executeTrain', 'color: #007acc; font-weight: bold;');
        const callStartTime = Date.now();
        
        /* ------------------------------------------------
         Call the Python script via IPC with project path and epochs
         returns an object with success flag and output/error message
        --------------------------------------------------- */
        const result = await window.electronAPI.executeTrain({
            projectPath: projectPath,
            epochs: parseInt(epochs)
        });
        
        const callDuration = Math.round((Date.now() - callStartTime) / 1000);
        console.log(`%c[UI] IPC handler completed in ${callDuration}s`, 'color: #007acc; font-weight: bold;');
        console.log('[UI] Result object:', result);
        
        // Update UI with result
        if (result.success) {
            const execTime = result.executionTime ? ` (${result.executionTime}s)` : '';
            randomNumberDisplay.textContent = `Training completed successfully!${execTime}`;
            randomNumberDisplay.style.color = '#28a745';
            console.log('%c[UI] Training successful!', 'color: #28a745; font-weight: bold;');
            console.log('[UI] Output:', result.output);
        } else {
            const execTime = result.executionTime ? ` (${result.executionTime}s)` : '';
            randomNumberDisplay.textContent = `Training failed.${execTime} Check console for details.`;
            randomNumberDisplay.style.color = '#dc3545';
            console.error('%c[UI] Training failed!', 'color: #dc3545; font-weight: bold;');
            console.error('[UI] Error:', result.error);
        }
    } catch (error) {
        // error handling if for some reason Electron or Python do not run successfully
        console.error('%c[UI] IPC error:', 'color: #dc3545; font-weight: bold;', error);
        document.getElementById('random-number').textContent = `IPC Error: ${error.message}`;
        document.getElementById('random-number').style.color = '#dc3545';
    } finally {
        // runs regardless of success or failure to ensure button is re-enabled
        // Re-enable button
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train';
    }
});


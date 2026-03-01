/*
  * Last modified by Alexangelo Orozco Gutierrez on 2-28-2026
  * Added download model route with parameter, uses test file for now
*/

import express from 'express';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { getDatasetsList, getModelsList, createDatasetFolder, getPareidoliaFolderPath, getLocalIP } from './main.js';
import { getVenvPath, executePythonScript } from './python.js';

const createServer = () => {
    const app = express();
    const port = 3001;

    // Middleware to parse JSON with larger payload size for video data
    app.use(express.json({ limit: '500mb' }));

    // Routes

    /**
     * GET /get-datasets
     * Returns a list of all project folders in the datasets directory
     */
    app.get('/get-datasets', async (req, res) => {
        try {
            const projects = await getDatasetsList();
            res.json(projects);
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    /**
     * GET /get-models
     * Returns a list of all model folders in the models directory
     */
    app.get('/get-models', async (req, res) => {
        try {
            const models = await getModelsList();
            res.json(models);
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    app.post('/add-dataset', async (req, res) => {
        const { datasetName } = req.body;
        if (!datasetName) {
            return res.status(400).json({ error: 'datasetName is required' });
        }
        try {
            await createDatasetFolder(datasetName);
            res.status(201).json({ success: true, datasetName });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    app.post('/add-model', async (req, res) => {
        const { modelName } = req.body;
        if (!modelName) {
            return res.status(400).json({ error: 'modelName is required' });
        }
        try {
            await createModelFolder(modelName);
            res.status(201).json({ success: true, modelName });
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    });

    //make image view pagination route for mobile


    /**
     * GET /download-model-mobile
     * Downloads a model by name
     * Query param: name (model name)
     * Example: GET /download-model-mobile?modelName=Flowers
     */
    app.get('/download-model-mobile', (req, res) => {
        const modelName = req.query.modelName;

        // Validate that modelName was provided
        if (!modelName) {
            return res.status(400).json({ error: 'Error, modelName parameter is required' });
        }

        try {
            // Construct path to the model folder
            const pareidoliaPath = getPareidoliaFolderPath();
            const modelPath = path.join(pareidoliaPath, 'models', modelName);

            // Check if model folder exists
            if (!fs.existsSync(modelPath)) {
                return res.status(404).json({ error: 'Error, failure to find model folder' });
            }

            // Construct path to the models subfolder
            const modelsSubfolderPath = path.join(modelPath, 'models');

            // Check for mock.png file TEST ONLY
            // const mockImagePath = path.join(modelsSubfolderPath, 'mock.png');
            // if (!fs.existsSync(mockImagePath)) {
            //     return res.status(404).json({ error: 'Error, no model created' });
            // }

            // Send the mock.png file
            // res.sendFile(mockImagePath);

            // TODO: When model training is complete, replace mock.png with actual model
            // Uncomment the code below to send the model.tflite file instead:
            const modelFilePath = path.join(modelsSubfolderPath, 'model.tflite');
            if (!fs.existsSync(modelFilePath)) {
                return res.status(404).json({ error: 'Error, no model created' });
            }
            res.download(modelFilePath, 'model.tflite');
        } catch (error) {
            console.error('Error downloading model:', error);
            res.status(500).json({ error: 'Error downloading model', message: error.message });
        }
    });

    /**
     * POST /upload-video
     * Uploads a video file to a specific dataset
     * Body: { fileName, fileData (base64), datasetName }
     */
    app.post('/upload-video', async (req, res) => {
        // Requirements of the request body, needed to reconstruct the video and where to save it
        const { fileName, fileData, datasetName } = req.body;
        console.log('VIDEO UPLOAD REQUEST RECEIVED');
        console.log('Timestamp:', new Date().toISOString());
        console.log('File name:', fileName || 'MISSING');
        console.log('Dataset name:', datasetName || 'MISSING');
        console.log('Data received:', fileData ? 'YES' : 'NO');
        console.log('Data size:', fileData ? `${(fileData.length / 1024 / 1024).toFixed(2)} MB (base64)` : 'N/A');
        
        // Validation, will provide the client side error messages to troubleshoot
        // Mobile side will ensure repeat files are not sent
        if (!fileName) {
            console.log('ERROR: No fileName provided');
            console.log('========================================\n');
            return res.status(400).json({ error: 'fileName is required' });
        }
        
        if (!datasetName) {
            console.log('ERROR: No datasetName provided');
            console.log('========================================\n');
            return res.status(400).json({ error: 'datasetName is required' });
        }
        
        if (!fileData) {
            console.log('ERROR: No fileData provided');
            console.log('========================================\n');
            return res.status(400).json({ error: 'fileData is required' });
        }
        
        try {
            // Construct dataset path
            const pareidoliaPath = getPareidoliaFolderPath();
            const datasetPath = path.join(pareidoliaPath, 'datasets', datasetName);
            
            // Check if dataset exists, if not create it
            if (!fs.existsSync(datasetPath)) {
                console.log('Dataset does not exist, creating:', datasetName);
                await createDatasetFolder(datasetName);
                console.log('Dataset created successfully');
            }
            
            // Decode base64 and write to file
            const buffer = Buffer.from(fileData, 'base64');
            const positivesPath = path.join(datasetPath, 'positives');
            const videoPath = path.join(positivesPath, fileName);
            fs.writeFileSync(videoPath, buffer);
            console.log('Video file saved:', videoPath);
            console.log('File size:', (buffer.length / 1024 / 1024).toFixed(2), 'MB');
            
            // Convert video to images
            const venvPath = getVenvPath();
            console.log('Starting video conversion...');
            const conversionResult = await executePythonScript('py/extract_images.py', [
                videoPath,
                positivesPath
            ], venvPath);
            
            if (conversionResult.success) {
                console.log('Video conversion successful:', conversionResult.output);
            } else {
                console.error('Video conversion failed:', conversionResult.error);
            }
            
            // Delete the video file after conversion
            try {
                fs.unlinkSync(videoPath);
                console.log('Video file deleted:', videoPath);
            } catch (deleteError) {
                console.error('Error deleting video file:', deleteError.message);
            }

            console.log('VIDEO UPLOAD SUCCESSFUL');
            console.log('========================================\n');
            
            res.status(201).json({ 
                success: true, 
                fileName: fileName,
                datasetName: datasetName,
                fileSize: buffer.length
            });
        } catch (error) {
            console.error('ERROR processing upload:', error);
            console.log('========================================\n');
            res.status(500).json({ 
                success: false, 
                error: 'Failed to process video upload',
                message: error.message 
            });
        }
    });

    app.post('ping', (req, res) => {
        console.log('Ping received at', new Date().toISOString());
        res.json({ message: 'pong' });
    });
    

    app.listen(port, '0.0.0.0', () => {
        const localIP = getLocalIP();
        console.log(`Express server is running on http://${localIP}:${port}`);
    });

    return app;
};

export default createServer;
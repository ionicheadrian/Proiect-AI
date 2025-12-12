// preload.js - Electron Preload Script
const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('api', {
  generateQuestion: async (category) => {
    try {
      const response = await fetch('http://localhost:5000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ category })
      });
      return await response.json();
    } catch (error) {
      throw new Error('Failed to generate question: ' + error.message);
    }
  }
});

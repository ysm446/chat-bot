const { contextBridge } = require('electron');

const API_PORT = '8765';
const API_HOST = '127.0.0.1';
const BASE_URL = `http://${API_HOST}:${API_PORT}`;

contextBridge.exposeInMainWorld('researchBotApi', {
  baseUrl: BASE_URL,
});

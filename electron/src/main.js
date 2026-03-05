const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

const API_PORT = process.env.RESEARCH_BOT_PORT || '8765';
const API_HOST = process.env.RESEARCH_BOT_HOST || '127.0.0.1';
const API_URL = `http://${API_HOST}:${API_PORT}`;

let mainWindow = null;
let backendProcess = null;

function startBackend() {
  const rootDir = path.resolve(__dirname, '..', '..');
  const pythonCmd = process.env.RESEARCH_BOT_PYTHON || 'python';
  const args = ['main.py', '--host', API_HOST, '--port', API_PORT];

  backendProcess = spawn(pythonCmd, args, {
    cwd: rootDir,
    stdio: 'pipe',
    windowsHide: true,
    env: {
      ...process.env,
      PYTHONUTF8: '1',
      PYTHONIOENCODING: 'utf-8',
    },
  });

  backendProcess.stdout.on('data', (data) => {
    console.log(`[backend] ${data.toString().trim()}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`[backend] ${data.toString().trim()}`);
  });

  backendProcess.on('exit', (code) => {
    console.log(`[backend] exited with code ${code}`);
    backendProcess = null;
  });
}

async function waitForBackendReady(timeoutMs = 60000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const res = await fetch(`${API_URL}/health`);
      if (res.ok) {
        return;
      }
    } catch (err) {
      // retry
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error('Backend did not become ready in time');
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 880,
    minWidth: 1024,
    minHeight: 720,
    title: 'Research-Bot',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.setMenuBarVisibility(false);
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

app.whenReady().then(async () => {
  startBackend();
  await waitForBackendReady();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
});

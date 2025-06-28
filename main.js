const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let pyProc = null;
let win = null;

function createWindow() {
  win = new BrowserWindow({
    width: 420,
    height: 600,
    resizable: false,
    fullscreenable: false,
    frame: false,
    transparent: true,
    hasShadow: true,
    webPreferences: {
      contextIsolation: true,
    },
  });

  win.loadURL('http://localhost:5000');
  win.setMenu(null);
}

function waitForServer(callback) {
  const http = require('http');
  const options = {
    hostname: 'localhost',
    port: 5000,
    path: '/test',
    method: 'GET',
  };

  const check = () => {
    const req = http.request(options, (res) => {
      if (res.statusCode === 200) callback();
      else setTimeout(check, 300);
    });

    req.on('error', () => setTimeout(check, 300));
    req.end();
  };

  check();
}

function startPythonServer() {
  // Get path to the backend executable inside the packaged app
  const exePath = path.join(process.resourcesPath, 'VoiceBoxBackend.exe');

  pyProc = spawn(exePath);

  pyProc.stdout.on('data', (data) => console.log(`[Backend] ${data}`));
  pyProc.stderr.on('data', (data) => console.error(`[Backend ERROR] ${data}`));

  pyProc.on('exit', (code) => {
    console.log(`Backend exited with code ${code}`);
  });
}

app.whenReady().then(() => {
  startPythonServer();
  waitForServer(createWindow);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (pyProc) pyProc.kill();
  if (process.platform !== 'darwin') app.quit();
});

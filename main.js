const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

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
  const script = path.join(__dirname, 'start_backend.js');
  pyProc = spawn('node', [script]);

  pyProc.stdout.on('data', (data) => console.log(`Backend: ${data}`));
  pyProc.stderr.on('data', (data) => console.error(`Backend error: ${data}`));
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

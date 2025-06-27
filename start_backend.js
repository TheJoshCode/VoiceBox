const { spawn } = require('child_process');
const path = require('path');

const exePath = path.join(__dirname, 'VoiceBoxBackend.exe');  // Adjust if in subfolder
const pyProc = spawn(exePath);

pyProc.stdout.on('data', (data) => console.log(`[Flask] ${data}`));
pyProc.stderr.on('data', (data) => console.error(`[Flask ERROR] ${data}`));

process.on('exit', () => pyProc.kill());

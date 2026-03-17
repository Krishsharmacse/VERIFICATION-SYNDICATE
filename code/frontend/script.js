/* ======================================================
    MATRIX RAIN
====================================================== */
const canvas = document.getElementById('matrix-canvas');
const ctx    = canvas.getContext('2d');

function resizeCanvas() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

const chars   = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
const fontSize = 16;
let drops;
function initDrops() {
    const cols = Math.floor(canvas.width / fontSize);
    drops = Array(cols).fill(1);
}
initDrops();
window.addEventListener('resize', initDrops);

function drawMatrix() {
    ctx.fillStyle = 'rgba(6, 12, 20, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#00f0b4';
    ctx.font = fontSize + 'px "Space Mono", monospace';
    for (let i = 0; i < drops.length; i++) {
        const text = chars[Math.floor(Math.random() * chars.length)];
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
        drops[i] += 0.75;
    }
}
window.matrixInterval = setInterval(drawMatrix, 48);

/* ======================================================
    GLOBAL STATE & TOAST
====================================================== */
let isAnalyzing = false;

function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

/* ======================================================
    TAB SWITCHING
====================================================== */
function switchTab(tab) {
    if (isAnalyzing) {
        showToast("Analysis in progress. Please wait before switching tabs.");
        return;
    }

    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    if (tab === 'text') {
        document.querySelector('.tab:nth-child(1)').classList.add('active');
        document.getElementById('text-tab').classList.add('active');
    } else if (tab === 'audio') {
        document.querySelector('.tab:nth-child(2)').classList.add('active');
        document.getElementById('audio-tab').classList.add('active');
    } else {
        document.querySelector('.tab:nth-child(3)').classList.add('active');
        document.getElementById('video-tab').classList.add('active');
    }
    document.getElementById('result').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
}

/* ======================================================
    FILE LABEL HELPER
====================================================== */
function updateFileLabel(inputId, labelId) {
    const input = document.getElementById(inputId);
    const label = document.getElementById(labelId);
    if (input.files && input.files[0]) {
        const file = input.files[0];
        const name = file.name;
        label.textContent = '✓ ' + (name.length > 30 ? name.substring(0, 28) + '…' : name);
        label.classList.add('has-file');

        if (inputId === 'audio-file') {
            const previewContainer = document.getElementById('audio-preview-container');
            const previewPlayer = document.getElementById('audio-preview');
            previewPlayer.src = URL.createObjectURL(file);
            previewContainer.style.display = 'block';
        } else if (inputId === 'video-file') {
            const previewContainer = document.getElementById('video-preview-container');
            const previewPlayer = document.getElementById('video-preview');
            previewPlayer.src = URL.createObjectURL(file);
            previewContainer.style.display = 'block';
        }
    }
}

/* ======================================================
    LOADER
====================================================== */
function showLoader(show) {
    document.getElementById('loader').style.display = show ? 'block' : 'none';
}

/* ======================================================
    DISPLAY RESULT  (from first code logic)
====================================================== */
function displayResult(data, type) {
    const resultDiv   = document.getElementById('result');
    const titleEl     = document.getElementById('result-title');
    const verdictBox  = document.getElementById('verdict-box');
    const confLabel   = document.getElementById('conf-label');
    const confPct     = document.getElementById('conf-pct');
    const confFill    = document.getElementById('conf-fill');
    const diagBox     = document.getElementById('diagnostics-box');
    const diagItems   = document.getElementById('diag-items');
    const reasonBox   = document.getElementById('reasoning-box');
    const reasonText  = document.getElementById('reasoning-text');

    // reset
    verdictBox.className = 'verdict-box';
    confFill.className   = 'confidence-fill';
    reasonBox.style.display = 'none';
    diagItems.innerHTML = '';

    if (type === 'text') {
        const isReal = data.label.includes('REAL');
        verdictBox.classList.add(isReal ? 'real' : 'fake');
        const dotHtml = '<span class="pulse-dot" style="display:inline-block; margin-right: 15px; margin-bottom: 5px; background: currentColor; box-shadow: 0 0 12px currentColor; transform: scale(1.5);"></span>';
        titleEl.innerHTML = dotHtml + (isReal ? 'REAL NEWS' : 'FAKE NEWS');

        const prob = isReal ? data.probability_real : data.probability_fake;
        confLabel.textContent = isReal ? 'Real probability' : 'Fake probability';
        confPct.textContent   = (prob * 100).toFixed(1) + '%';
        if (!isReal) confFill.classList.add('fake-bar');

        requestAnimationFrame(() => requestAnimationFrame(() => {
            confFill.style.width = (prob * 100) + '%';
        }));

        // Agent scores
        diagItems.innerHTML = `
            <div class="diag-item"><span class="diag-key">Real probability</span><span class="diag-val">${(data.probability_real*100).toFixed(1)}%</span></div>
            <div class="diag-item"><span class="diag-key">Fake probability</span><span class="diag-val">${(data.probability_fake*100).toFixed(1)}%</span></div>
            <div class="diag-item"><span class="diag-key">Evidence found</span><span class="diag-val">${data.evidence_found ? 'Yes' : 'No'}</span></div>
            <div class="diag-item"><span class="diag-key">FactCheck score</span><span class="diag-val">${(data.details.factcheck*100).toFixed(1)}%</span></div>
            <div class="diag-item"><span class="diag-key">NewsAPI score</span><span class="diag-val">${(data.details.newsapi*100).toFixed(1)}%</span></div>
            <div class="diag-item"><span class="diag-key">GNews score</span><span class="diag-val">${(data.details.gnews*100).toFixed(1)}%</span></div>
            <div class="diag-item"><span class="diag-key">LLM score</span><span class="diag-val">${(data.details.llm*100).toFixed(1)}%</span></div>
        `;

        if (data.reasoning) {
            reasonBox.style.display = 'block';
            reasonText.textContent  = data.reasoning;
        }

    } else if (type === 'audio') {
        const isFake = data.label.includes('FAKE');
        verdictBox.classList.add(isFake ? 'fake' : 'real');
        const dotHtml = '<span class="pulse-dot" style="display:inline-block; margin-right: 15px; margin-bottom: 5px; background: currentColor; box-shadow: 0 0 12px currentColor; transform: scale(1.5);"></span>';
        titleEl.innerHTML = dotHtml + (isFake ? 'FAKE / SYNTHETIC AUDIO' : 'REAL HUMAN VOICE');

        confLabel.textContent = 'Confidence';
        confPct.textContent   = (data.confidence * 100).toFixed(1) + '%';
        if (isFake) confFill.classList.add('fake-bar');

        requestAnimationFrame(() => requestAnimationFrame(() => {
            confFill.style.width = (data.confidence * 100) + '%';
        }));

        diagItems.innerHTML = `
            <div class="diag-item"><span class="diag-key">Score (0=real, 1=fake)</span><span class="diag-val">${data.score.toFixed(4)}</span></div>
            <div class="diag-item"><span class="diag-key">Confidence</span><span class="diag-val">${(data.confidence*100).toFixed(1)}%</span></div>
        `;

    } else { // video
        const isFake   = data.label.includes('FAKE');
        const isUnable = data.label.includes('UNABLE');
        verdictBox.classList.add(isFake ? 'fake' : 'real');
        const dotHtml = '<span class="pulse-dot" style="display:inline-block; margin-right: 15px; margin-bottom: 5px; background: currentColor; box-shadow: 0 0 12px currentColor; transform: scale(1.5);"></span>';
        titleEl.innerHTML = dotHtml + (isFake ? 'FAKE / SYNTHETIC VIDEO' : isUnable ? 'ANALYSIS ERROR' : 'REAL HUMAN VIDEO');

        if (isUnable) {
            confLabel.textContent = 'Status';
            confPct.textContent   = '—';
            diagItems.innerHTML   = `<div class="diag-item"><span class="diag-key">Message</span><span class="diag-val">${data.details.message}</span></div>`;
        } else {
            confLabel.textContent = 'Deepfake probability';
            confPct.textContent   = (data.score * 100).toFixed(1) + '%';
            if (isFake) confFill.classList.add('fake-bar');

            requestAnimationFrame(() => requestAnimationFrame(() => {
                confFill.style.width = (data.score * 100) + '%';
            }));

            diagItems.innerHTML = `
                <div class="diag-item"><span class="diag-key">Verdict</span><span class="diag-val">${data.label}</span></div>
                <div class="diag-item"><span class="diag-key">Total frames scanned</span><span class="diag-val">${data.details.frames_analyzed}</span></div>
                <div class="diag-item"><span class="diag-key">Frames flagged fake</span><span class="diag-val">${data.details.fake_frames} (${data.details.fake_percentage.toFixed(1)}%)</span></div>
            `;
        }
    }

    resultDiv.style.display = 'block';
}

/* ======================================================
    CLEAR MEDIA FILE HELPER
====================================================== */
function clearFile(type) {
    if (type === 'audio') {
        const input = document.getElementById('audio-file');
        const label = document.getElementById('audio-file-label');
        const container = document.getElementById('audio-preview-container');
        const player = document.getElementById('audio-preview');
        
        input.value = '';
        label.textContent = '🎤  Choose Audio File (.wav, .mp3, .flac, .m4a, .ogg)';
        label.classList.remove('has-file');
        player.pause();
        player.src = '';
        container.style.display = 'none';
        
        const aPlayBtn = document.getElementById('audio-play-btn');
        const aProgress = document.getElementById('audio-progress');
        const aTime = document.getElementById('audio-time');
        aTime.textContent = '0:00 / 0:00';
        aProgress.style.width = '0%';
        aPlayBtn.innerHTML = '▶';
        aPlayBtn.classList.remove('playing');
        
    } else if (type === 'video') {
        const input = document.getElementById('video-file');
        const label = document.getElementById('video-file-label');
        const container = document.getElementById('video-preview-container');
        const player = document.getElementById('video-preview');
        
        input.value = '';
        label.textContent = '🎥  Choose Video File (.mp4, .avi, .mov, .mkv, .webm)';
        label.classList.remove('has-file');
        player.pause();
        player.src = '';
        container.style.display = 'none';
    }
}

/* ======================================================
    SHOW ERROR
====================================================== */
function showError(message) {
    const eb = document.getElementById('errorBox');
    eb.textContent = '⚠️ ' + message;
    eb.style.display = 'block';
}

/* ======================================================
    ANALYZE FUNCTIONS  (from first code — unchanged logic)
====================================================== */
async function analyzeText() {
    const text = document.getElementById('claim-text').value.trim();
    if (!text) { alert('Please enter some text.'); return; }
    isAnalyzing = true;
    showLoader(true);
    document.getElementById('result').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
    try {
        const response = await fetch('/predict/text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'Server error'); }
        const data = await response.json();
        displayResult(data, 'text');
    } catch (err) {
        showError(err.message);
    } finally {
        showLoader(false);
        isAnalyzing = false;
    }
}

async function analyzeAudio() {
    const fileInput = document.getElementById('audio-file');
    const file = fileInput.files[0];
    if (!file) { alert('Please select an audio file.'); return; }
    const formData = new FormData();
    formData.append('file', file);
    isAnalyzing = true;
    showLoader(true);
    document.getElementById('result').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
    try {
        const response = await fetch('/predict/audio', { method: 'POST', body: formData });
        if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'Server error'); }
        const data = await response.json();
        displayResult(data, 'audio');
    } catch (err) {
        showError(err.message);
    } finally {
        showLoader(false);
        isAnalyzing = false;
    }
}

async function analyzeVideo() {
    const fileInput = document.getElementById('video-file');
    const file = fileInput.files[0];
    if (!file) { alert('Please select a video file.'); return; }
    const formData = new FormData();
    formData.append('file', file);
    isAnalyzing = true;
    showLoader(true);
    document.getElementById('result').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
    try {
        const response = await fetch('/predict/video', { method: 'POST', body: formData });
        if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'Server error'); }
        const data = await response.json();
        displayResult(data, 'video');
    } catch (err) {
        showError(err.message);
    } finally {
        showLoader(false);
        isAnalyzing = false;
    }
}

/* ======================================================
    CUSTOM AUDIO PLAYER LOGIC
====================================================== */
const aPlayer = document.getElementById('audio-preview');
const aPlayBtn = document.getElementById('audio-play-btn');
const aProgress = document.getElementById('audio-progress');
const aTrack = document.getElementById('audio-track');
const aTime = document.getElementById('audio-time');

function formatTime(s) {
    if (isNaN(s)) return "0:00";
    const min = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return min + ":" + (sec < 10 ? "0" + sec : sec);
}

aPlayBtn.addEventListener('click', () => {
    if (aPlayer.paused) {
        aPlayer.play();
        aPlayBtn.innerHTML = '⏸';
        aPlayBtn.classList.add('playing');
    } else {
        aPlayer.pause();
        aPlayBtn.innerHTML = '▶';
        aPlayBtn.classList.remove('playing');
    }
});

aPlayer.addEventListener('timeupdate', () => {
    if (aPlayer.duration) {
        const pct = (aPlayer.currentTime / aPlayer.duration) * 100;
        aProgress.style.width = pct + '%';
        aTime.textContent = formatTime(aPlayer.currentTime) + ' / ' + formatTime(aPlayer.duration);
    }
});

aPlayer.addEventListener('loadedmetadata', () => {
    aTime.textContent = '0:00 / ' + formatTime(aPlayer.duration);
    aProgress.style.width = '0%';
    aPlayBtn.innerHTML = '▶';
    aPlayBtn.classList.remove('playing');
});

aPlayer.addEventListener('ended', () => {
    aPlayBtn.innerHTML = '▶';
    aPlayBtn.classList.remove('playing');
    aProgress.style.width = '0%';
    aPlayer.currentTime = 0;
});

aTrack.addEventListener('click', (e) => {
    if (!aPlayer.duration) return;
    const rect = aTrack.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, x / rect.width));
    aPlayer.currentTime = pct * aPlayer.duration;
});

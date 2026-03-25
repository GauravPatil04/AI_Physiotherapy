let webcamStream = null;
let frameInterval = null;
let currentExercise = '';
let captureVideo = null;
const frameCanvas = document.createElement('canvas');
const frameCtx = frameCanvas.getContext('2d');

function toggleDropdown() {
    const dropdown = document.getElementById("profile-dropdown");
    dropdown.classList.toggle("hidden");
}

function ensureCaptureVideo() {
    if (!captureVideo) {
        captureVideo = document.createElement('video');
        captureVideo.autoplay = true;
        captureVideo.playsInline = true;
        captureVideo.muted = true;
    }
    return captureVideo;
}

function updateExerciseLabel(name) {
    if (!name) return;
    currentExercise = name;
    document.getElementById('selected-exercise').innerText = 'Selected Exercise: ' + name.replace(/_/g, ' ');
}

async function startTracking(){
    const capture = ensureCaptureVideo();
    const feedImage = document.getElementById('video-feed');
    try {
        if (!webcamStream) {
            webcamStream = await navigator.mediaDevices.getUserMedia({
    video: {
        width: 1280,
        height: 720,
        facingMode: "user"
    },
    audio: false
});
                capture.srcObject = webcamStream;
                await capture.play();
        }
        document.getElementById('camera-placeholder').classList.add('hidden');
            feedImage.classList.remove('hidden');
        await fetch('/start');
    } catch (err) {
        document.getElementById('feedback').innerText = 'Camera access denied. Please allow camera permission.';
        return;
    }

    if (!frameInterval) {
        frameInterval = setInterval(sendFrame, 700);
    }
}

async function stopTracking(){
    await fetch('/stop');

    const feedImage = document.getElementById('video-feed');
    document.getElementById('camera-placeholder').classList.remove('hidden');
    feedImage.classList.add('hidden');
    feedImage.removeAttribute('src');

    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    if (captureVideo) {
        captureVideo.srcObject = null;
    }
}

async function sendFrame(){
    if (!captureVideo || !captureVideo.videoWidth || !captureVideo.videoHeight) return;

    frameCanvas.width = captureVideo.videoWidth;
    frameCanvas.height = captureVideo.videoHeight;
    frameCtx.drawImage(captureVideo, 0, 0, frameCanvas.width, frameCanvas.height);
    const image = frameCanvas.toDataURL('image/jpeg', 0.7);

    const res = await fetch('/process_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image })
    });
    const data = await res.json();
    const stats = data.stats || {};
    if (data.image) {
        document.getElementById('video-feed').src = 'data:image/jpeg;base64,' + data.image;
    }

    document.getElementById('total').innerText = stats.total_reps || 0;
    document.getElementById('correct').innerText = stats.correct_reps || 0;
    document.getElementById('feedback').innerText = stats.feedback || "";
}

function updateStats(){
    fetch('/stats')
    .then(res => res.json())
    .then(data => {
        updateExerciseLabel(data.exercise_name);
        document.getElementById('total').innerText = data.total_reps || 0;
        document.getElementById('correct').innerText = data.correct_reps || 0;
        document.getElementById('feedback').innerText = data.feedback || "";
    });
}

function resetTracking() {
    if (!currentExercise) return;
    fetch('/exercise/' + currentExercise).then(() => updateStats());
}

function cleanupCameraOnExit() {
    const feedImage = document.getElementById('video-feed');
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (captureVideo) {
        captureVideo.srcObject = null;
    }
    if (feedImage) {
        feedImage.removeAttribute('src');
    }
    fetch('/stop', { keepalive: true });
}

window.onclick = function(event) {
    if (!event.target.closest('.profile-container')) {
        document.getElementById("profile-dropdown").classList.add("hidden");
    }
}

setInterval(updateStats, 1000);

// Auto update stats when page loads
updateStats();
window.addEventListener('beforeunload', cleanupCameraOnExit);
window.addEventListener('pagehide', cleanupCameraOnExit);

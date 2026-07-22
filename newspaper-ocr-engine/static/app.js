const gallery = document.getElementById('gallery');
const canvas = document.getElementById('docCanvas');
const ctx = canvas.getContext('2d');
const runBtn = document.getElementById('runBtn');
const resultText = document.getElementById('resultText');
const statusMsg = document.getElementById('statusMsg');
const modelSelect = document.getElementById('modelSelect');

let currentImage = null;
let currentFilename = '';
let isDrawing = false;
let startX = 0;
let startY = 0;
let currentBox = null; // {x, y, w, h}

// Image list (1-10)
const imageFiles = [
    '01.jpg', '02.jpg', '03.jpg', '04.jpg', '05.jpg',
    '06.jpg', '07.jpg', '08.jpg', '09.jpg', '10.jpg'
];

// Load gallery
imageFiles.forEach(filename => {
    const img = document.createElement('img');
    img.src = `images/${filename}`;
    img.className = 'thumbnail';
    img.addEventListener('click', () => loadToCanvas(filename, img));
    gallery.appendChild(img);
});

// Load first image by default
if (gallery.firstChild) {
    loadToCanvas(imageFiles[0], gallery.firstChild);
}

function loadToCanvas(filename, thumbnailElement) {
    // Update active state in gallery
    document.querySelectorAll('.thumbnail').forEach(el => el.classList.remove('active'));
    thumbnailElement.classList.add('active');

    currentFilename = filename;
    currentBox = null;
    runBtn.disabled = true;
    resultText.value = '';
    statusMsg.innerText = '';

    const img = new Image();
    img.src = `images/${filename}`;
    img.onload = () => {
        currentImage = img;
        
        // Setup canvas resolution to match image
        canvas.width = img.width;
        canvas.height = img.height;
        
        // CSS handles scaling it to fit the container
        redraw();
    };
}

function redraw() {
    if (!currentImage) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0);

    if (currentBox) {
        ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'; // Primary blue, transparent
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 4;
        
        ctx.fillRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
        ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
    }
}

// Mouse events for drawing
function getMousePos(evt) {
    const rect = canvas.getBoundingClientRect();
    // Scale mouse coordinates to actual canvas resolution
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
        x: (evt.clientX - rect.left) * scaleX,
        y: (evt.clientY - rect.top) * scaleY
    };
}

canvas.addEventListener('mousedown', (e) => {
    if (!currentImage) return;
    isDrawing = true;
    const pos = getMousePos(e);
    startX = pos.x;
    startY = pos.y;
    currentBox = { x: startX, y: startY, w: 0, h: 0 };
    runBtn.disabled = true;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const pos = getMousePos(e);
    
    currentBox.w = pos.x - startX;
    currentBox.h = pos.y - startY;
    
    redraw();
});

canvas.addEventListener('mouseup', (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    
    // Normalize box (handle dragging left/up)
    if (currentBox.w < 0) {
        currentBox.x += currentBox.w;
        currentBox.w = Math.abs(currentBox.w);
    }
    if (currentBox.h < 0) {
        currentBox.y += currentBox.h;
        currentBox.h = Math.abs(currentBox.h);
    }
    
    // Round coords
    currentBox = {
        x: Math.round(currentBox.x),
        y: Math.round(currentBox.y),
        w: Math.round(currentBox.w),
        h: Math.round(currentBox.h)
    };
    
    if (currentBox.w > 10 && currentBox.h > 10) {
        runBtn.disabled = false;
    } else {
        currentBox = null;
        redraw();
    }
});

// API Call
runBtn.addEventListener('click', async () => {
    if (!currentBox || !currentFilename) return;
    
    runBtn.disabled = true;
    const originalText = runBtn.innerText;
    runBtn.innerText = 'Processing...';
    resultText.value = 'Running inference...';
    statusMsg.innerText = '';
    
    const requestBody = {
        image_filename: currentFilename,
        x: currentBox.x,
        y: currentBox.y,
        w: currentBox.w,
        h: currentBox.h,
        engine: modelSelect.value
    };
    
    try {
        const response = await fetch('/api/evaluate-crop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            resultText.value = data.extracted_text;
            statusMsg.innerText = `Completed in ${data.time_ms}ms`;
        } else {
            resultText.value = 'Error processing region.';
            statusMsg.innerText = data.detail || 'Unknown error';
        }
    } catch (err) {
        resultText.value = 'Network error.';
        statusMsg.innerText = err.message;
    } finally {
        runBtn.disabled = false;
        runBtn.innerText = originalText;
    }
});

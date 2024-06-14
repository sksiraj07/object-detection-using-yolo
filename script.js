function selectOption(option) {
    document.getElementById('page1').classList.remove('active');
    document.getElementById('page2').classList.add('active');
    document.getElementById('selected-option').innerText = option;
    document.getElementById('fileInput').accept = option === 'image' ? 'image/*' : 'video/*';
}

function goBack(pageNumber) {
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    document.getElementById(`page${pageNumber}`).classList.add('active');
}

function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        alert('Please select a file to upload.');
        return;
    }

    const file = fileInput.files[0];
    const outputDiv = document.getElementById('output');

    if (file.type.startsWith('image/')) {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        outputDiv.innerHTML = '';
        outputDiv.appendChild(img);
    } else if (file.type.startsWith('video/')) {
        const video = document.createElement('video');
        video.controls = true;
        video.src = URL.createObjectURL(file);
        outputDiv.innerHTML = '';
        outputDiv.appendChild(video);
    }

    document.getElementById('page2').classList.remove('active');
    document.getElementById('page3').classList.add('active');
}

function saveFile() {
    const outputDiv = document.getElementById('output');
    const outputElement = outputDiv.children[0];

    if (!outputElement) {
        alert('No output to save.');
        return;
    }

    const link = document.createElement('a');
    if (outputElement.tagName === 'IMG') {
        link.href = outputElement.src;
        link.download = 'output.png';
    } else if (outputElement.tagName === 'VIDEO') {
        link.href = outputElement.src;
        link.download = 'output.mp4';
    }

    link.click();
}

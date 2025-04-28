document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const textInput = document.getElementById('textInput');
    const translateTextBtn = document.getElementById('translateTextBtn');
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const startListeningBtn = document.getElementById('startListeningBtn');
    const stopListeningBtn = document.getElementById('stopListeningBtn');
    const englishResult = document.getElementById('englishResult');
    const aslResult = document.getElementById('aslResult');
    const recentTranslations = document.getElementById('recentTranslations');
    const loadModelBtn = document.getElementById('loadModelBtn');
    const modelPath = document.getElementById('modelPath');
    const labelMappingsPath = document.getElementById('labelMappingsPath');
    const statusToast = document.getElementById('statusToast');
    const toastHeader = document.getElementById('toastHeader');
    const toastTitle = document.getElementById('toastTitle');
    const toastMessage = document.getElementById('toastMessage');
    
    // Bootstrap Toast
    const toast = new bootstrap.Toast(statusToast);
    
    // Variables
    let mediaRecorder;
    let audioChunks = [];
    let continuousPolling = false;
    let pollingInterval;
    
    // Functions
    function showToast(title, message, isSuccess = true) {
        toastTitle.textContent = title;
        toastMessage.textContent = message;
        
        if (isSuccess) {
            statusToast.classList.remove('error-toast');
            statusToast.classList.add('success-toast');
        } else {
            statusToast.classList.remove('success-toast');
            statusToast.classList.add('error-toast');
        }
        
        toast.show();
    }
    
    function translateText() {
        const text = textInput.value.trim();
        
        if (!text) {
            showToast('Error', 'Please enter text to translate', false);
            return;
        }
        
        // Show loading state
        translateTextBtn.disabled = true;
        translateTextBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Translating...';
        
        // Make API request
        fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'text': text
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Display results
                englishResult.textContent = data.english;
                aslResult.textContent = data.asl_gloss;
                
                // Add to recent translations
                addToRecentTranslations(data.english, data.asl_gloss);
                
                showToast('Success', 'Translation completed');
            } else {
                showToast('Error', data.error || 'Failed to translate text', false);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error', 'Failed to translate text', false);
        })
        .finally(() => {
            // Reset button state
            translateTextBtn.disabled = false;
            translateTextBtn.innerHTML = 'Translate Text';
        });
    }
    
    function addToRecentTranslations(english, asl) {
        // Create new list item
        const li = document.createElement('li');
        li.className = 'list-group-item translation-item';
        
        const timestamp = new Date().toLocaleTimeString();
        
        li.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <strong>English:</strong>
                <span class="translation-time">${timestamp}</span>
            </div>
            <p class="mb-1">${english}</p>
            <div><strong>ASL Gloss:</strong></div>
            <p class="mb-0">${asl}</p>
        `;
        
        // Remove "No recent translations" if it exists
        if (recentTranslations.querySelector('.text-muted')) {
            recentTranslations.innerHTML = '';
        }
        
        // Add to the top of the list
        recentTranslations.insertBefore(li, recentTranslations.firstChild);
        
        // Limit to 10 items
        while (recentTranslations.children.length > 10) {
            recentTranslations.removeChild(recentTranslations.lastChild);
        }
    }
    
    function startRecording() {
        // Request microphone access
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.addEventListener('dataavailable', event => {
                    audioChunks.push(event.data);
                });
                
                mediaRecorder.addEventListener('stop', () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('audio', audioBlob);
                    
                    // Show loading state
                    recordBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                    
                    // Send to server
                    fetch('/translate', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            // Display results
                            englishResult.textContent = data.english;
                            aslResult.textContent = data.asl_gloss;
                            
                            // Add to recent translations
                            addToRecentTranslations(data.english, data.asl_gloss);
                            
                            showToast('Success', 'Speech recognition completed');
                        } else {
                            showToast('Error', data.error || 'Failed to recognize speech', false);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        showToast('Error', 'Failed to process audio', false);
                    })
                    .finally(() => {
                        // Reset button state
                        recordBtn.disabled = false;
                        recordBtn.classList.remove('recording');
                        recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Record';
                        stopBtn.disabled = true;
                    });
                });
                
                // Start recording
                mediaRecorder.start();
                
                // Update UI
                recordBtn.classList.add('recording');
                recordBtn.disabled = true;
                stopBtn.disabled = false;
                
                showToast('Recording', 'Recording started... Speak now');
            })
            .catch(error => {
                console.error('Error accessing microphone:', error);
                showToast('Error', 'Could not access microphone', false);
            });
    }
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            showToast('Processing', 'Processing audio...');
        }
    }
    
    function startContinuousListening() {
        // Update UI
        startListeningBtn.disabled = true;
        startListeningBtn.classList.add('active');
        stopListeningBtn.disabled = false;
        
        // Start server-side continuous listening
        fetch('/listen/start', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('Listening', 'Continuous listening started');
                
                // Start polling for results
                continuousPolling = true;
                pollResults();
            } else {
                showToast('Error', data.error || 'Failed to start listening', false);
                resetContinuousListening();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error', 'Failed to start continuous listening', false);
            resetContinuousListening();
        });
    }
    
    function stopContinuousListening() {
        // Update UI
        resetContinuousListening();
        
        // Stop server-side continuous listening
        fetch('/listen/stop', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('Stopped', 'Continuous listening stopped');
            } else {
                showToast('Error', data.error || 'Failed to stop listening', false);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error', 'Failed to stop continuous listening', false);
        });
    }
    
    function resetContinuousListening() {
        startListeningBtn.disabled = false;
        startListeningBtn.classList.remove('active');
        stopListeningBtn.disabled = true;
        continuousPolling = false;
        clearTimeout(pollingInterval);
    }
    
    function pollResults() {
        if (!continuousPolling) return;
        
        fetch('/listen/results')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.results && data.results.length > 0) {
                    // Get the most recent result
                    const latestResult = data.results[0];
                    
                    // Display results
                    englishResult.textContent = latestResult.english;
                    aslResult.textContent = latestResult.asl_gloss;
                    
                    // Add to recent translations
                    addToRecentTranslations(latestResult.english, latestResult.asl_gloss);
                }
                
                // Continue polling
                if (continuousPolling) {
                    pollingInterval = setTimeout(pollResults, 2000);
                }
            })
            .catch(error => {
                console.error('Error polling results:', error);
                if (continuousPolling) {
                    pollingInterval = setTimeout(pollResults, 5000); // Retry after longer delay
                }
            });
    }
    
    function loadModel() {
        const modelPathValue = modelPath.value.trim();
        const labelMappingsPathValue = labelMappingsPath.value.trim();
        
        if (!modelPathValue || !labelMappingsPathValue) {
            showToast('Error', 'Please enter both model path and label mappings path', false);
            return;
        }
        
        // Show loading state
        loadModelBtn.disabled = true;
        loadModelBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
        
        // Make API request
        fetch('/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'model_path': modelPathValue,
                'label_mappings_path': labelMappingsPathValue
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('Success', 'Model loaded successfully');
                
                // Close modal
                const modelModal = bootstrap.Modal.getInstance(document.getElementById('modelModal'));
                modelModal.hide();
            } else {
                showToast('Error', data.error || 'Failed to load model', false);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error', 'Failed to load model', false);
        })
        .finally(() => {
            // Reset button state
            loadModelBtn.disabled = false;
            loadModelBtn.innerHTML = 'Load Model';
        });
    }
    
    // Event Listeners
    translateTextBtn.addEventListener('click', translateText);
    
    recordBtn.addEventListener('click', startRecording);
    
    stopBtn.addEventListener('click', stopRecording);
    
    startListeningBtn.addEventListener('click', startContinuousListening);
    
    stopListeningBtn.addEventListener('click', stopContinuousListening);
    
    loadModelBtn.addEventListener('click', loadModel);
    
    // Allow pressing Enter in the text input
    textInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            translateText();
        }
    });
}); 
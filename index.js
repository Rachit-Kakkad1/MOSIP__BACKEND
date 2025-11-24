// Central front-end interactions (safe across pages)
// API-ready helpers designed to work with a Python backend (FastAPI/Flask)
// NOTE: Ensure your Python backend enables CORS for the frontend origin.
const API_BASE = localStorage.getItem('ocrpro-api-base') || 'http://localhost:8000/api';

async function uploadFileToApi(file) {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: fd
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
}

async function fetchPreviewFromApi(docId) {
    const res = await fetch(`${API_BASE}/preview/${encodeURIComponent(docId)}`);
    if (!res.ok) throw new Error(`Preview fetch failed: ${res.status}`);
    return res.json();
}

async function postVerificationToApi(docId, payload) {
    const res = await fetch(`${API_BASE}/verify/${encodeURIComponent(docId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`Verify failed: ${res.status}`);
    return res.json();
}

document.addEventListener('DOMContentLoaded', () => {
    // Theme toggle (persisted)
    const themeToggle = document.querySelector('.theme-toggle');
    const prefersDark = localStorage.getItem('ocrpro-theme') !== 'light';
    let isDark = prefersDark;
    if (themeToggle) {
        themeToggle.textContent = isDark ? '☀️' : '🌙';
        themeToggle.addEventListener('click', () => {
            isDark = !isDark;
            themeToggle.textContent = isDark ? '☀️' : '🌙';
            localStorage.setItem('ocrpro-theme', isDark ? 'dark' : 'light');
            // Add theme switching logic here if you want to change CSS variables
        });
    }

    // Highlight active nav link
    const currentFile = (location.pathname.split('/').pop() || 'index.html');
    const navButtons = document.querySelectorAll('.nav-button');
    navButtons.forEach(btn => {
        try {
            if (btn.tagName === 'A') {
                const hrefName = (new URL(btn.href)).pathname.split('/').pop();
                if (hrefName === currentFile) btn.classList.add('active');
            }
        } catch (e) {
            // ignore malformed hrefs
        }
    });

    // Upload card: on Upload page, trigger file input and POST to backend
    const uploadCard = document.querySelector('.upload-card');
    if (uploadCard) {
        uploadCard.addEventListener('click', async () => {
            let fi = document.querySelector('#hidden-file-input');
            if (!fi) {
                fi = document.createElement('input');
                fi.type = 'file';
                fi.id = 'hidden-file-input';
                fi.style.display = 'none';
                document.body.appendChild(fi);
                fi.addEventListener('change', async (ev) => {
                    const f = ev.target.files && ev.target.files[0];
                    if (!f) return;
                    try {
                        // Show a simple loading hint
                        uploadCard.style.opacity = '0.7';
                        const json = await uploadFileToApi(f);
                        // Expecting backend response like { doc_id: 'abc123' }
                        if (json && json.doc_id) {
                            location.href = `preview.html?doc=${encodeURIComponent(json.doc_id)}`;
                        } else {
                            alert('Upload complete (no doc id returned).');
                        }
                    } catch (err) {
                        console.error(err);
                        alert('Upload failed. See console for details.');
                    } finally {
                        uploadCard.style.opacity = '';
                    }
                });
            }
            fi.click();
        });
    }

    // Sample button: navigate to Upload page with sample query if not on upload
    const sampleButton = document.querySelector('.sample-button');
    if (sampleButton) {
        sampleButton.addEventListener('click', (e) => {
            e.stopPropagation();
            if (currentFile === 'upload.html') {
                // If already on upload page, simulate loading sample
                alert('Sample document loaded into the uploader.');
            } else {
                location.href = 'upload.html?sample=1';
            }
        });
    }

    // If we're on upload page and ?sample=1 is present, simulate loading sample
    if (currentFile === 'upload.html' && location.search.indexOf('sample=1') !== -1) {
        setTimeout(() => {
            // Placeholder: in a real flow you'd request a sample from the API
            alert('Sample document preloaded (demo).');
        }, 250);
    }

    // Feature cards hover effect (if present)
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.boxShadow = '0 8px 32px rgba(163, 113, 247, 0.2)';
        });
        card.addEventListener('mouseleave', () => {
            card.style.boxShadow = 'none';
        });
    });
});
// Set your backend FastAPI API URL from Render or other cloud provider
const API_URL = "https://ai-medical-image-captioning-system.onrender.com/caption/";

// DOM element references
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const resultsDiv = document.getElementById("results");
const submitBtn = document.getElementById("submitBtn");
const spinner = document.getElementById("loadingSpinner");
const previewImage = document.getElementById("previewImage");

// Handle form submit event
uploadForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (!fileInput.files.length) {
        alert("Please upload an image.");
        return;
    }

    // Show spinner and disable button
    spinner.style.display = "inline-block";
    submitBtn.disabled = true;
    resultsDiv.innerHTML = "";

    // Show image preview
    const file = fileInput.files;
    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = "block";

    const formData = new FormData();
    formData.append("file", file); // 'file' matches FastAPI endpoint parameter

    try {
        // POST image to FastAPI backend
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (data.captions) {
            resultsDiv.innerHTML = `
                <h4>Generated Captions:</h4>
                <ul class="list-group">
                    ${data.captions.map(c => `<li class="list-group-item">${c}</li>`).join("")}
                </ul>
            `;
        } else {
            resultsDiv.innerHTML = "<p>No captions returned.</p>";
        }
    } catch (err) {
        resultsDiv.innerHTML = `<p class="text-danger">Error: ${err.message}</p>`;
    } finally {
        spinner.style.display = "none";
        submitBtn.disabled = false;
    }
});


document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file");
    const form = document.querySelector("form");
    const submitButton = form
        ? form.querySelector('button[type="submit"], button')
        : null;

    if (!fileInput || !form) {
        return;
    }

    // Display the selected filename.
    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];

        if (file && !file.name.toLowerCase().endsWith(".csv")) {
            alert("Please select a CSV file.");
            fileInput.value = "";
            return;
        }

        if (file) {
            console.log("Selected file:", file.name);
        }
    });

    // Prevent submission when no file is selected.
    form.addEventListener("submit", function (event) {
        if (!fileInput.files.length) {
            event.preventDefault();
            alert("Please select a patient CSV file first.");
            return;
        }

        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = "Processing prediction...";
        }
    });
});
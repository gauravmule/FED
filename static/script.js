document.addEventListener("DOMContentLoaded", function () {
    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const videoStream = document.getElementById("video-stream");
    const summaryContent = document.getElementById("summary-content");
    let updateInterval;

    async function initializeCamera() {
        try {
            videoStream.src = "/video_feed?t=" + Date.now();
            await new Promise((resolve) => {
                videoStream.onload = resolve;
                videoStream.onerror = () => {
                    throw new Error("Webcam not accessible. Please check your camera settings.");
                };
                setTimeout(resolve, 1000);
            });
            return true;
        } catch (error) {
            console.error("Camera initialization failed:", error);
            alert(error.message);
            return false;
        }
    }

    startBtn.addEventListener("click", async function () {
        try {
            startBtn.disabled = true;
            const response = await fetch("/start_session", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
            });
            if (!response.ok) throw new Error("Network response was not ok");
            const data = await response.json();

            if (data.success) {
                if (await initializeCamera()) {
                    stopBtn.disabled = false;
                    updateInterval = setInterval(async () => {
                        try {
                            const summaryResponse = await fetch("/get_emotion_summary");
                            if (!summaryResponse.ok) throw new Error("Failed to fetch summary");
                            const summary = await summaryResponse.json();
                            updateSummaryDisplay(summary);
                        } catch (error) {
                            console.error("Summary update failed:", error);
                        }
                    }, 1500);
                } else {
                    throw new Error("Camera initialization failed");
                }
            } else {
                throw new Error(data.message || "Failed to start session");
            }
        } catch (error) {
            console.error("Start session error:", error);
            alert(`Error: ${error.message}`);
            startBtn.disabled = false;
        }
    });

    stopBtn.addEventListener("click", async function () {
        try {
            clearInterval(updateInterval);
            videoStream.src = "";
            startBtn.disabled = false;
            stopBtn.disabled = true;
            summaryContent.innerHTML = "<p>No active session</p>";
            const response = await fetch("/stop_session", { method: "POST" });
            if (!response.ok) throw new Error("Failed to stop session");
        } catch (error) {
            console.error("Stop session error:", error);
            alert(`Error: ${error.message}`);
        }
    });

    function updateSummaryDisplay(summary) {
        let html = `<p>Detected Faces: ${summary.total_faces}</p><ul>`;
        if (summary.total_faces > 0) {
            for (const [emotion, count] of Object.entries(summary.emotions)) {
                html += `<li>${emotion}: ${count}</li>`;
            }
        } else {
            html += "<li>No faces detected</li>";
        }
        summaryContent.innerHTML = html + "</ul>";
    }
});
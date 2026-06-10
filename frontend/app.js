// API base url
const API_URL = "";

// Global App State
let activeBatchId = null;
let activeSlideIdx = 0;
let activeLayerId = null;
let batchData = null;

// Dragging state variables for canvas items
let isDragging = false;
let draggedElement = null;
let dragStartX = 0;
let dragStartY = 0;
let initialElementLeft = 0;
let initialElementTop = 0;

// Initialize components on load
document.addEventListener("DOMContentLoaded", () => {
    initUpload();
    initTabs();
    initButtons();
    checkExistingJobs();
});

// Helper: Add Console Log Line
function logConsole(message, type = "info") {
    const consoleBox = document.getElementById("console-logs");
    if (!consoleBox) return;
    
    const timestamp = new Date().toLocaleTimeString();
    const line = document.createElement("div");
    line.className = `log-line ${type}`;
    line.innerHTML = `<span style="color: #64748B;">[${timestamp}]</span> ${message}`;
    
    consoleBox.appendChild(line);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

// ----------------- Upload Handling -----------------
function initUpload() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");

    // Click to select files
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadFiles(e.target.files);
        }
    });

    // Drag-and-drop visual behaviors
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            uploadFiles(e.dataTransfer.files);
        }
    });
}

// Upload Files to API
async function uploadFiles(filesList) {
    logConsole(`Uploading ${filesList.length} diagram files...`, "info");
    
    const formData = new FormData();
    for (let i = 0; i < filesList.length; i++) {
        formData.append("files", filesList[i]);
    }
    
    try {
        const response = await fetch(`${API_URL}/api/upload`, {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Upload failed");
        }
        
        const job = await response.json();
        activeBatchId = job.batch_id;
        activeSlideIdx = 0;
        
        logConsole(`Upload successful. Created batch job: ${activeBatchId.slice(0, 8)}`, "success");
        logConsole("Queueing batch job in background pipeline...", "system");
        
        // Start polling status
        pollJobStatus(activeBatchId);
        
    } catch (error) {
        logConsole(`Upload failed: ${error.message}`, "error");
    }
}

// ----------------- Polling & State Tracking -----------------
let pollInterval = null;

async function fetchLogs(batchId, slideIdx) {
    try {
        const response = await fetch(`${API_URL}/api/batches/${batchId}/slides/${slideIdx}/logs`);
        if (response.ok) {
            const logsText = await response.text();
            const consoleBox = document.getElementById("console-logs");
            if (consoleBox && logsText.trim()) {
                const lines = logsText.trim().split("\n");
                consoleBox.innerHTML = lines.map(line => {
                    const timestampMatch = line.match(/^\[(.*?)\] (.*)/);
                    if (timestampMatch) {
                        return `<div class="log-line info"><span style="color: #64748B;">[${timestampMatch[1]}]</span> ${timestampMatch[2]}</div>`;
                    }
                    return `<div class="log-line info">${line}</div>`;
                }).join("");
                consoleBox.scrollTop = consoleBox.scrollHeight;
            }
        }
    } catch (e) {
        console.error("Error fetching logs:", e);
    }
}

function pollJobStatus(batchId) {
    if (pollInterval) clearInterval(pollInterval);
    
    // Refresh batch listing panel
    updateBatchesList(batchId, "processing", 20);
    
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/api/status/${batchId}`);
            if (!response.ok) return;
            
            const job = await response.json();
            batchData = job;
            
            // Poll pipeline logs for the active slide index
            await fetchLogs(batchId, activeSlideIdx);
            
            // Check status
            if (job.status === "completed") {
                clearInterval(pollInterval);
                logConsole(`Decomposition pipeline finished successfully!`, "success");
                logConsole("PowerPoint compiled and ready to download.", "success");
                
                // Update UI elements
                updateBatchesList(batchId, "completed", 100, job);
                selectSlide(0);
                
                // Enable buttons
                document.getElementById("export-ppt-btn").disabled = false;
                document.getElementById("save-slide-btn").disabled = false;
                
            } else if (job.status === "processing") {
                updateBatchesList(batchId, "processing", 60);
                
            } else if (job.status === "error") {
                clearInterval(pollInterval);
                logConsole(`Pipeline encountered error: ${job.error_message}`, "error");
                updateBatchesList(batchId, "error", 100);
            }
            
        } catch (error) {
            console.error("Error polling job status:", error);
        }
    }, 2000);
}

// Check existing jobs from localStorage on reload
function checkExistingJobs() {
    // Basic local state caching could go here if needed.
}

// Render active batch jobs left sidebar cards
function updateBatchesList(batchId, status, progressVal, jobData = null) {
    const list = document.getElementById("batches-list");
    if (!list) return;
    
    // Clear empty state
    const empty = list.querySelector(".empty-state");
    if (empty) empty.remove();
    
    let card = document.getElementById(`batch-card-${batchId}`);
    if (!card) {
        card = document.createElement("div");
        card.id = `batch-card-${batchId}`;
        card.className = "batch-card active";
        list.prepend(card);
    }
    
    const count = jobData ? jobData.slides.length : 1;
    const ratingText = jobData ? `Conf: ${(jobData.slides[0]?.average_confidence * 100).toFixed(0)}%` : "Analysing...";
    const routeClass = jobData ? getRouteClass(jobData.slides[0]?.routing_status) : "route-review";
    const routeName = jobData ? jobData.slides[0]?.routing_status : "Checking";
    
    card.innerHTML = `
        <div class="batch-meta">
            <span class="batch-title">Batch: ${batchId.slice(0, 8)}</span>
            <span class="batch-badge badge-${status}">${status}</span>
        </div>
        <div class="batch-progress">
            <div class="batch-progress-bar ${status === "processing" ? "animated" : ""}" style="width: ${progressVal}%"></div>
        </div>
        <div class="batch-stats">
            <span>Slides: ${count}</span>
            <span class="route-badge ${routeClass}">${routeName}</span>
        </div>
    `;
    
    card.onclick = () => {
        if (status === "completed" && jobData) {
            batchData = jobData;
            activeBatchId = batchId;
            // Set active states
            document.querySelectorAll(".batch-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            selectSlide(0);
            
            document.getElementById("export-ppt-btn").disabled = false;
            document.getElementById("save-slide-btn").disabled = false;
        }
    };
}

function getRouteClass(status) {
    if (status === "Auto Export") return "route-auto";
    if (status === "Warning") return "route-warning";
    return "route-review";
}

// ----------------- Slide Viewport Rendering -----------------
function drawRelations(slide) {
    const container = document.getElementById("canvas-container");
    let svg = document.getElementById("relations-svg");
    if (!svg) {
        svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.id = "relations-svg";
        svg.style.position = "absolute";
        svg.style.top = "0";
        svg.style.left = "0";
        svg.style.width = "100%";
        svg.style.height = "100%";
        svg.style.pointerEvents = "none";
        svg.style.zIndex = "1000";
        svg.style.overflow = "visible";
        container.appendChild(svg);
    }
    svg.innerHTML = "";

    const toggleRelations = document.getElementById("toggle-relations");
    const showRelations = toggleRelations ? toggleRelations.checked : true;
    
    if (showRelations && slide.relationships && slide.relationships.length > 0) {
        let defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
        defs.innerHTML = `
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#38bdf8" />
            </marker>
        `;
        svg.appendChild(defs);
        
        slide.relationships.forEach(rel => {
            const labelComp = slide.components.find(c => c.id === rel.label_id);
            const targetComp = slide.components.find(c => c.id === rel.target_id);
            
            if (labelComp && targetComp && labelComp.visible && targetComp.visible) {
                // Compute centers
                const x1 = ((labelComp.box[0] + labelComp.box[2] / 2) / slide.width) * 100;
                const y1 = ((labelComp.box[1] + labelComp.box[3] / 2) / slide.height) * 100;
                
                const x2 = ((targetComp.box[0] + targetComp.box[2] / 2) / slide.width) * 100;
                const y2 = ((targetComp.box[1] + targetComp.box[3] / 2) / slide.height) * 100;
                
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", `${x1}%`);
                line.setAttribute("y1", `${y1}%`);
                line.setAttribute("x2", `${x2}%`);
                line.setAttribute("y2", `${y2}%`);
                line.setAttribute("stroke", "#38bdf8");
                line.setAttribute("stroke-width", "2");
                line.setAttribute("stroke-dasharray", "4 4");
                line.setAttribute("marker-end", "url(#arrowhead)");
                line.setAttribute("opacity", "0.85");
                
                svg.appendChild(line);
            }
        });
    }
}

function selectSlide(idx) {
    if (!batchData || idx < 0 || idx >= batchData.slides.length) return;
    
    activeSlideIdx = idx;
    const slide = batchData.slides[idx];
    
    logConsole(`Selected Slide ${idx + 1}: ${slide.original_filename}`, "info");
    
    // Toggle slide navigation controls
    const nav = document.getElementById("slide-nav");
    if (batchData.slides.length > 1) {
        nav.style.display = "flex";
        document.getElementById("slide-counter").innerText = `${idx + 1} / ${batchData.slides.length}`;
    } else {
        nav.style.display = "none";
    }
    
    document.getElementById("slide-indicator").innerText = `Slide ${idx + 1}`;
    
    // Hide empty viewport state
    document.getElementById("canvas-empty").style.display = "none";
    const container = document.getElementById("canvas-container");
    container.style.display = "block";
    
    // 1. Set background image source
    // Use background.png if it's a raster slide, otherwise fall back to original file
    const isRaster = ["PNG", "JPG", "JPEG", "WEBP"].includes(slide.file_type.toUpperCase());
    const bgFilename = isRaster ? "background.png" : `original.${slide.file_type.toLowerCase()}`;
    const originalSrc = `${API_URL}/storage/batches/${batchData.batch_id}/slides/slide_${idx}/${bgFilename}?t=${Date.now()}`;
    
    const bgImg = document.getElementById("canvas-bg");
    bgImg.src = originalSrc;
    
    // Reset canvas overlays
    const overlays = document.getElementById("canvas-overlays");
    overlays.innerHTML = "";
    
    // Render layers
    renderCanvasLayers(slide, overlays);
    renderLayerTree(slide);
    updateInfoTab(slide);
    
    // Fetch logs immediately for the slide
    fetchLogs(batchData.batch_id, idx);
    
    // Draw relationship lines
    drawRelations(slide);
}

function renderCanvasLayers(slide, overlaysContainer) {
    slide.components.forEach(comp => {
        if (!comp.visible) return;
        
        const box = document.createElement("div");
        box.id = `canvas-box-${comp.id}`;
        box.className = `overlay-box type-${comp.type === "text_label" ? "label" : comp.type}`;
        box.style.zIndex = comp.z_index;
        
        // Calculate coordinate percentages relative to original image size
        const leftPct = (comp.box[0] / slide.width) * 100;
        const topPct = (comp.box[1] / slide.height) * 100;
        const widthPct = (comp.box[2] / slide.width) * 100;
        const heightPct = (comp.box[3] / slide.height) * 100;
        
        box.style.left = `${leftPct}%`;
        box.style.top = `${topPct}%`;
        box.style.width = `${widthPct}%`;
        box.style.height = `${heightPct}%`;
        
        // Add overlay visual text tag
        const tag = document.createElement("span");
        tag.className = "overlay-tag";
        tag.innerText = comp.type === "text_label" ? "Editable Text" : comp.type === "arrow" ? "Arrow" : "Object";
        box.appendChild(tag);
        
        // Add specific content rendering
        if ((comp.type === "image_object" || comp.type === "arrow") && comp.mask_path) {
            // Render transparent crop image overlay
            const img = document.createElement("img");
            img.src = `${API_URL}/storage/batches/${batchData.batch_id}/slides/slide_${slide.slide_index}/${comp.mask_path}`;
            img.className = "overlay-mask-img";
            box.appendChild(img);
            
        } else if (comp.type === "text_label" && comp.text) {
            // Render inline editable text field
            const textInput = document.createElement("input");
            textInput.type = "text";
            textInput.className = "overlay-text-input";
            textInput.value = comp.text;
            
            // Text change updates coordinates / structure in-memory
            textInput.addEventListener("input", (e) => {
                comp.text = e.target.value;
                // Sync to layer tree input
                const treeInput = document.getElementById(`layer-input-${comp.id}`);
                if (treeInput) treeInput.value = e.target.value;
            });
            
            box.appendChild(textInput);
        }
        
        // Setup Drag & Drop coordinates
        setupDragElement(box, comp, slide);
        
        // Hover highlight syncing
        box.addEventListener("mouseenter", () => {
            const treeItem = document.getElementById(`layer-item-${comp.id}`);
            if (treeItem) treeItem.classList.add("active");
        });
        
        box.addEventListener("mouseleave", () => {
            const treeItem = document.getElementById(`layer-item-${comp.id}`);
            if (treeItem && activeLayerId !== comp.id) treeItem.classList.remove("active");
        });
        
        // Select Layer on click
        box.addEventListener("mousedown", () => {
            selectLayer(comp.id);
        });
        
        overlaysContainer.appendChild(box);
    });
}

// Coordinate Drag and Drop Logic (Percentage-based responsive viewport mapping)
function setupDragElement(element, component, slide) {
    element.addEventListener("mousedown", (e) => {
        if (e.target.tagName === "INPUT") return; // Allow text editing selection
        
        e.preventDefault();
        isDragging = true;
        draggedElement = element;
        activeLayerId = component.id;
        
        const rect = element.getBoundingClientRect();
        const viewport = document.getElementById("canvas-container").getBoundingClientRect();
        
        // Mouse click coordinates relative to target box
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        
        // Initial box bounds in pixels relative to viewport
        initialElementLeft = rect.left - viewport.left;
        initialElementTop = rect.top - viewport.top;
        
        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("mouseup", onMouseUp);
    });

    function onMouseMove(e) {
        if (!isDragging || draggedElement !== element) return;
        
        const viewport = document.getElementById("canvas-container").getBoundingClientRect();
        
        // Compute delta movement
        const deltaX = e.clientX - dragStartX;
        const deltaY = e.clientY - dragStartY;
        
        // New position in pixels relative to viewport boundaries
        let newLeft = initialElementLeft + deltaX;
        let newTop = initialElementTop + deltaY;
        
        // Keep inside viewport limits
        const widthVal = element.getBoundingClientRect().width;
        const heightVal = element.getBoundingClientRect().height;
        newLeft = Math.max(0, Math.min(newLeft, viewport.width - widthVal));
        newTop = Math.max(0, Math.min(newTop, viewport.height - heightVal));
        
        // Convert back to percentages to scale correctly
        const leftPct = (newLeft / viewport.width) * 100;
        const topPct = (newTop / viewport.height) * 100;
        
        element.style.left = `${leftPct}%`;
        element.style.top = `${topPct}%`;
        
        // Update temporary coordinates in memory so lines draw dynamically!
        const imgLeft = Math.round((newLeft / viewport.width) * slide.width);
        const imgTop = Math.round((newTop / viewport.height) * slide.height);
        component.box[0] = imgLeft;
        component.box[1] = imgTop;
        
        drawRelations(slide);
    }

    function onMouseUp() {
        if (isDragging) {
            isDragging = false;
            document.removeEventListener("mousemove", onMouseMove);
            document.removeEventListener("mouseup", onMouseUp);
            
            // Recalculate original image coordinates based on current percentage position
            const viewport = document.getElementById("canvas-container").getBoundingClientRect();
            const rect = element.getBoundingClientRect();
            
            const pxLeft = rect.left - viewport.left;
            const pxTop = rect.top - viewport.top;
            
            // Convert to image dimensions scale
            const imgLeft = Math.round((pxLeft / viewport.width) * slide.width);
            const imgTop = Math.round((pxTop / viewport.height) * slide.height);
            
            // Save to memory
            component.box[0] = imgLeft;
            component.box[1] = imgTop;
            
            logConsole(`Moved layer ${component.id} to coordinate: [${imgLeft}, ${imgTop}]`, "info");
            
            // Final redraw for relationship lines
            drawRelations(slide);
        }
    }
    
    function rectWidth() { return element.getBoundingClientRect().width; }
    function rectHeight() { return element.getBoundingClientRect().height; }
}

function selectLayer(id) {
    activeLayerId = id;
    
    // Canvas highlight
    document.querySelectorAll(".overlay-box").forEach(b => b.classList.remove("active"));
    const box = document.getElementById(`canvas-box-${id}`);
    if (box) box.classList.add("active");
    
    // Layer tree highlight
    document.querySelectorAll(".layer-item").forEach(item => item.classList.remove("active"));
    const item = document.getElementById(`layer-item-${id}`);
    if (item) {
        item.classList.add("active");
        item.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

// ----------------- Right Panel Tree & Tabs -----------------
function renderLayerTree(slide) {
    const tree = document.getElementById("layers-tree");
    tree.innerHTML = "";
    
    // Sort reverse Z-index to show top layers at the top of list
    const sorted = [...slide.components].sort((a, b) => b.z_index - a.z_index);
    
    if (sorted.length === 0) {
        tree.innerHTML = `<div class="empty-state"><p>No layers detected on this slide.</p></div>`;
        return;
    }
    
    sorted.forEach(comp => {
        const item = document.createElement("div");
        item.id = `layer-item-${comp.id}`;
        item.className = `layer-item ${activeLayerId === comp.id ? "active" : ""}`;
        
        const dotType = comp.type === "text_label" ? "label" : comp.type;
        const displayName = comp.type === "text_label" ? comp.text : (comp.semantic_name ? comp.semantic_name : `${comp.type.replace("_", " ")} (${comp.id.split("_").pop()})`);
        
        item.innerHTML = `
            <div class="layer-left">
                <i class="fa-solid fa-grip-vertical layer-drag-handle"></i>
                <span class="layer-color-dot type-${dotType}"></span>
                <input type="text" class="layer-name-input" id="layer-input-${comp.id}" value="${displayName}">
            </div>
            <div class="layer-right" style="display: flex; align-items: center; gap: 8px;">
                <span class="layer-conf" style="font-size: 10px; color: var(--text-secondary); margin-right: 4px;">(${(comp.confidence * 100).toFixed(0)}%)</span>
                <button class="layer-action-btn move-up-btn" title="Move Up" style="padding: 2px; background: transparent; border: none; cursor: pointer; color: #64748B;"><i class="fa-solid fa-arrow-up" style="font-size: 10px;"></i></button>
                <button class="layer-action-btn move-down-btn" title="Move Down" style="padding: 2px; background: transparent; border: none; cursor: pointer; color: #64748B;"><i class="fa-solid fa-arrow-down" style="font-size: 10px;"></i></button>
                <button class="layer-action-btn" id="layer-vis-${comp.id}">
                    <i class="fa-solid ${comp.visible ? "fa-eye" : "fa-eye-slash"}"></i>
                </button>
            </div>
        `;
        
        // Input text edit sync
        const inp = item.querySelector(".layer-name-input");
        inp.addEventListener("input", (e) => {
            if (comp.type === "text_label") {
                comp.text = e.target.value;
                const canvasInput = document.querySelector(`#canvas-box-${comp.id} .overlay-text-input`);
                if (canvasInput) canvasInput.value = e.target.value;
            } else {
                comp.semantic_name = e.target.value;
            }
        });
        
        // Move Up / Down handlers
        const upBtn = item.querySelector(`.move-up-btn`);
        if (upBtn) {
            upBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                const currentIndex = sorted.findIndex(c => c.id === comp.id);
                if (currentIndex > 0) {
                    // Swap z_index
                    const temp = sorted[currentIndex].z_index;
                    sorted[currentIndex].z_index = sorted[currentIndex - 1].z_index;
                    sorted[currentIndex - 1].z_index = temp;
                    slide.components.sort((a, b) => a.z_index - b.z_index);
                    logConsole(`Moved layer ${comp.id} up in depth.`, "info");
                    selectSlide(activeSlideIdx); // redraw
                }
            });
        }
        
        const downBtn = item.querySelector(`.move-down-btn`);
        if (downBtn) {
            downBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                const currentIndex = sorted.findIndex(c => c.id === comp.id);
                if (currentIndex < sorted.length - 1) {
                    // Swap z_index
                    const temp = sorted[currentIndex].z_index;
                    sorted[currentIndex].z_index = sorted[currentIndex + 1].z_index;
                    sorted[currentIndex + 1].z_index = temp;
                    slide.components.sort((a, b) => a.z_index - b.z_index);
                    logConsole(`Moved layer ${comp.id} down in depth.`, "info");
                    selectSlide(activeSlideIdx); // redraw
                }
            });
        }
        
        // Visibility toggle
        const visBtn = item.querySelector(`#layer-vis-${comp.id}`);
        visBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            comp.visible = !comp.visible;
            logConsole(`Toggled layer ${comp.id} visibility to ${comp.visible}`, "info");
            selectSlide(activeSlideIdx); // redraw
        });
        
        // Select Layer on item click
        item.addEventListener("click", () => {
            selectLayer(comp.id);
        });
        
        tree.appendChild(item);
    });
}

function updateInfoTab(slide) {
    document.getElementById("info-content-type").innerText = slide.content_type;
    document.getElementById("info-file-type").innerText = slide.file_type;
    document.getElementById("info-resolution").innerText = `${slide.width} x ${slide.height} px`;
    document.getElementById("info-confidence").innerText = `${(slide.average_confidence * 100).toFixed(0)}%`;
    
    // Update Latencies Dashboard
    const metrics = slide.performance_metrics || {};
    document.getElementById("metric-classify").innerText = metrics.classification_time_ms ? `${metrics.classification_time_ms} ms` : "-";
    document.getElementById("metric-detect").innerText = metrics.detection_time_ms ? `${metrics.detection_time_ms} ms` : "-";
    document.getElementById("metric-segment").innerText = metrics.segmentation_time_ms ? `${metrics.segmentation_time_ms} ms` : "-";
    document.getElementById("metric-ocr").innerText = metrics.ocr_time_ms ? `${metrics.ocr_time_ms} ms` : "-";
    document.getElementById("metric-understand").innerText = metrics.understanding_time_ms ? `${metrics.understanding_time_ms} ms` : "-";
    document.getElementById("metric-ppt").innerText = metrics.ppt_compile_time_ms ? `${metrics.ppt_compile_time_ms} ms` : "-";
    
    // Update workflow routing details
    const alertBox = document.getElementById("routing-alert");
    const title = document.getElementById("routing-title");
    const desc = document.getElementById("routing-desc");
    
    // Reset classes
    alertBox.className = "routing-alert";
    
    if (slide.routing_status === "Auto Export") {
        alertBox.classList.add("alert-auto");
        title.innerText = "Auto-Export Approved";
        desc.innerText = `Average confidence (${(slide.average_confidence * 100).toFixed(0)}%) is high. The pipeline has automatically validated these slide layers. PowerPoint export is ready.`;
    } else if (slide.routing_status === "Warning") {
        alertBox.classList.add("alert-warning");
        title.innerText = "Validation Warning";
        desc.innerText = `Average confidence (${(slide.average_confidence * 100).toFixed(0)}%) is moderate (70-90%). Some text labels or line boundaries may require minor positioning tweaks.`;
    } else {
        alertBox.classList.add("alert-review");
        title.innerText = "Manual Review Required";
        desc.innerText = `Attention Required: Confidence (${(slide.average_confidence * 100).toFixed(0)}%) is low (<70%). We recommend reviewing and manually aligning labels and organs before exporting the slide.`;
    }
}

// ----------------- Tab Navigation -----------------
function initTabs() {
    const tabLayers = document.getElementById("tab-layers");
    const tabMeta = document.getElementById("tab-meta");
    const layersContent = document.getElementById("layers-content");
    const metaContent = document.getElementById("meta-content");
    
    tabLayers.addEventListener("click", () => {
        tabLayers.classList.add("active");
        tabMeta.classList.remove("active");
        layersContent.style.display = "flex";
        metaContent.style.display = "none";
    });
    
    tabMeta.addEventListener("click", () => {
        tabMeta.classList.add("active");
        tabLayers.classList.remove("active");
        metaContent.style.display = "flex";
        layersContent.style.display = "none";
    });
}

// ----------------- Action Buttons -----------------
function initButtons() {
    // Save button
    const saveBtn = document.getElementById("save-slide-btn");
    saveBtn.addEventListener("click", async () => {
        if (!batchData) return;
        
        logConsole(`Saving modifications for Slide ${activeSlideIdx + 1}...`, "info");
        saveBtn.disabled = true;
        
        try {
            const slide = batchData.slides[activeSlideIdx];
            const response = await fetch(`${API_URL}/api/batches/${batchData.batch_id}/slides/${activeSlideIdx}/components`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ components: slide.components })
            });
            
            if (!response.ok) throw new Error("Failed to save slide modifications");
            
            logConsole(`Slide modifications saved! PowerPoint rebuilt successfully.`, "success");
            
            // Reload the slide to refresh the background image erasures
            selectSlide(activeSlideIdx);
        } catch (error) {
            logConsole(`Save failed: ${error.message}`, "error");
        } finally {
            saveBtn.disabled = false;
        }
    });

    // Export PPT button
    const exportBtn = document.getElementById("export-ppt-btn");
    exportBtn.addEventListener("click", () => {
        if (!batchData) return;
        
        logConsole("Downloading compiled PowerPoint slides...", "success");
        // Trigger browser download by routing to file stream endpoint
        window.location.href = `${API_URL}/api/download/${batchData.batch_id}`;
    });
    
    // Slide Navigation Buttons
    document.getElementById("prev-slide-btn").addEventListener("click", () => {
        if (activeSlideIdx > 0) {
            selectSlide(activeSlideIdx - 1);
        }
    });
    
    document.getElementById("next-slide-btn").addEventListener("click", () => {
        if (batchData && activeSlideIdx < batchData.slides.length - 1) {
            selectSlide(activeSlideIdx + 1);
        }
    });
    
    const toggleRelations = document.getElementById("toggle-relations");
    if (toggleRelations) {
        toggleRelations.addEventListener("change", () => {
            if (batchData && batchData.slides[activeSlideIdx]) {
                drawRelations(batchData.slides[activeSlideIdx]);
            }
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const foundryStatusPill = document.getElementById("foundryStatusPill");
    const vectorEngineText = document.getElementById("vectorEngineText");
    const indexedDocsCount = document.getElementById("indexedDocsCount");
    
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const uploadProgress = document.getElementById("uploadProgress");
    
    const docList = document.getElementById("docList");
    const btnRefreshDocs = document.getElementById("btnRefreshDocs");
    const topKSlider = document.getElementById("topKSlider");
    const topKVal = document.getElementById("topKVal");
    const sourceFilterSelect = document.getElementById("sourceFilterSelect");
    
    const chatMessages = document.getElementById("chatMessages");
    const userInput = document.getElementById("userInput");
    const btnSend = document.getElementById("btnSend");
    
    const inspectorDrawer = document.getElementById("inspectorDrawer");
    const inspectorContent = document.getElementById("inspectorContent");
    const btnToggleInspector = document.getElementById("btnToggleInspector");
    const btnCloseInspector = document.getElementById("btnCloseInspector");

    const connectionModal = document.getElementById("connectionModal");
    const btnCloseModal = document.getElementById("btnCloseModal");
    const modalFoundryMode = document.getElementById("modalFoundryMode");
    const modalFoundryModel = document.getElementById("modalFoundryModel");
    const modalSqlitePath = document.getElementById("modalSqlitePath");
    const modalSqliteCount = document.getElementById("modalSqliteCount");

    let lastQueryResult = null;

    // --- 1. INITIAL SYSTEM STATUS & DOCS FETCH ---
    async function fetchSystemStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            
            // Update Foundry Pill
            const fl = data.foundry_local;
            const chatModel = fl.chat_model || "phi-3.5-mini";
            if (fl.available) {
                foundryStatusPill.innerHTML = `<span class="status-dot success"></span><span class="status-text">Foundry Local: Aktif (${chatModel})</span>`;
            } else {
                foundryStatusPill.innerHTML = `<span class="status-dot warning"></span><span class="status-text">Foundry Local: Local Extractive Mode</span>`;
            }

            // Update Engine & Counts
            vectorEngineText.textContent = data.storage_engine || "SQLite Vector DB";
            indexedDocsCount.textContent = `${data.indexed_docs} Doküman (${data.indexed_chunks} Chunk)`;

            // Update Modal Information
            if (modalFoundryMode) modalFoundryMode.textContent = `Mod: ${fl.mode || 'Foundry Local SDK'} (Local Inference)`;
            if (modalFoundryModel) modalFoundryModel.textContent = `${chatModel} (Aktif)`;
            if (modalSqliteCount) modalSqliteCount.textContent = `${data.indexed_chunks} Chunks (${data.indexed_docs} Doküman)`;
        } catch (err) {
            console.error("Status fetch error:", err);
            foundryStatusPill.innerHTML = `<span class="status-dot warning"></span><span class="status-text">Offline / Bağlanamadı</span>`;
        }
    }

    // Modal Events
    foundryStatusPill.addEventListener("click", () => {
        connectionModal.classList.remove("hidden");
    });

    btnCloseModal.addEventListener("click", () => {
        connectionModal.classList.add("hidden");
    });

    connectionModal.addEventListener("click", (e) => {
        if (e.target === connectionModal) {
            connectionModal.classList.add("hidden");
        }
    });

    async function fetchDocuments() {
        try {
            const res = await fetch("/api/documents");
            const data = await res.json();
            
            docList.innerHTML = "";
            sourceFilterSelect.innerHTML = `<option value="">Tüm SQLite Dokümanlarında Ara</option>`;

            if (!data.documents || data.documents.length === 0) {
                docList.innerHTML = `<div class="doc-item"><span class="doc-meta">SQLite veritabanında doküman yok.</span></div>`;
                return;
            }

            data.documents.forEach(doc => {
                const isPdf = doc.filename.toLowerCase().endsWith(".pdf");
                const iconClass = isPdf ? "fa-file-pdf" : "fa-file-word";
                const iconColor = isPdf ? "#ef4444" : "var(--accent-cyan)";
                const item = document.createElement("div");
                item.className = "doc-item";
                item.innerHTML = `
                    <div class="doc-info">
                        <i class="fa-solid ${iconClass}" style="color: ${iconColor};"></i>
                        <div>
                            <div class="doc-name">${doc.filename}</div>
                            <div class="doc-meta">${(doc.file_size / 1024).toFixed(1)} KB • ${doc.chunk_count} SQLite Chunk</div>
                        </div>
                    </div>
                    <i class="fa-solid fa-circle-check" style="color: #10b981;"></i>
                `;
                docList.appendChild(item);

                // Option in filter select
                const opt = document.createElement("option");
                opt.value = doc.filename;
                opt.textContent = doc.filename;
                sourceFilterSelect.appendChild(opt);
            });
        } catch (err) {
            console.error("Fetch docs error:", err);
        }
    }

    // --- 2. UPLOAD HANDLERS ---
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        const nameLower = (file.name || "").toLowerCase();
        const validExts = [".pdf", ".docx", ".md", ".txt"];
        const isValid = validExts.some(ext => nameLower.endsWith(ext));

        if (!isValid) {
            alert(`"${file.name}" desteklenmeyen bir dosya türü. Lütfen sadece .pdf, .docx, .md veya .txt dosyası yükleyin.`);
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        uploadProgress.classList.remove("hidden");

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                await fetchDocuments();
                await fetchSystemStatus();
            } else {
                alert(`Hata: ${data.detail || "Dosya yüklenemedi."}`);
            }
        } catch (err) {
            alert("Yükleme sırasında ağ hatası oluştu.");
        } finally {
            uploadProgress.classList.add("hidden");
        }
    }

    btnRefreshDocs.addEventListener("click", () => {
        fetchDocuments();
        fetchSystemStatus();
    });

    // --- 3. RAG CONTROLS & QUICK PROMPTS ---
    topKSlider.addEventListener("input", (e) => {
        topKVal.textContent = e.target.value;
    });

    document.querySelectorAll(".chip-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const q = btn.getAttribute("data-query");
            userInput.value = q;
            sendQuery();
        });
    });

    // --- 4. CHAT & RAG QUERY PROCESSING ---
    btnSend.addEventListener("click", sendQuery);

    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    async function sendQuery() {
        const queryText = userInput.value.trim();
        if (!queryText) return;

        // Remove Welcome Card if present
        const welcomeCard = document.querySelector(".welcome-card");
        if (welcomeCard) welcomeCard.remove();

        // 1. Append User Message Bubble
        appendUserMessage(queryText);
        userInput.value = "";

        // 2. Append Loading Assistant Bubble
        const loadingId = appendLoadingMessage();

        // Prepare request
        const payload = {
            question: queryText,
            top_k: parseInt(topKSlider.value, 10),
            filter_source: sourceFilterSelect.value || null
        };

        try {
            const res = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            removeLoadingMessage(loadingId);

            if (res.ok) {
                lastQueryResult = data;
                appendAssistantMessage(data);
                updateInspector(data);
            } else {
                appendErrorMessage(data.detail || "Sorgu işlenirken hata oluştu.");
            }
        } catch (err) {
            removeLoadingMessage(loadingId);
            appendErrorMessage("Sunucuya erişilemiyor.");
        }
    }

    function appendUserMessage(text) {
        const bubble = document.createElement("div");
        bubble.className = "message-bubble user";
        bubble.innerHTML = `
            <div class="msg-header"><i class="fa-solid fa-user"></i> Kullanıcı</div>
            <div class="msg-content">${escapeHtml(text)}</div>
        `;
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendLoadingMessage() {
        const id = "loading_" + Date.now();
        const bubble = document.createElement("div");
        bubble.id = id;
        bubble.className = "message-bubble assistant";
        bubble.innerHTML = `
            <div class="msg-header"><i class="fa-solid fa-robot"></i> Local RAG Engine</div>
            <div class="msg-content">
                <div class="upload-progress">
                    <div class="spinner"></div>
                    <span>SQLite vektör veritabanı taranıyor ve yanıt üretiliyor...</span>
                </div>
            </div>
        `;
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    function removeLoadingMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendAssistantMessage(data) {
        const bubble = document.createElement("div");
        bubble.className = "message-bubble assistant";
        
        // Parse Markdown for answer
        const parsedMarkdown = marked.parse(data.answer);

        // Build Citations Cards
        let citationsHtml = "";
        if (data.context_chunks && data.context_chunks.length > 0) {
            const cards = data.context_chunks.map((c, idx) => `
                <div class="citation-card">
                    <div class="cit-header">
                        <span>📄 ${c.source} (SQLite Chunk #${c.chunk_id + 1})</span>
                        <span class="cit-score">%${(c.score * 100).toFixed(0)} (Kalibre) • Yalın: %${((c.raw_score !== undefined ? c.raw_score : c.score) * 100).toFixed(0)}</span>
                    </div>
                    <div class="cit-text">${escapeHtml(c.text)}</div>
                </div>
            `).join("");

            citationsHtml = `
                <div class="citation-box">
                    <div class="citation-title"><i class="fa-solid fa-database"></i> SQLite Veritabanından Getirilen Dayanak Parçaları (${data.context_chunks.length} Chunk)</div>
                    <div class="citation-cards">${cards}</div>
                </div>
            `;
        }

        bubble.innerHTML = `
            <div class="msg-header">
                <i class="fa-solid fa-robot"></i> ${data.engine}
            </div>
            <div class="msg-content">
                ${parsedMarkdown}
                ${citationsHtml}
            </div>
        `;

        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendErrorMessage(msg) {
        const bubble = document.createElement("div");
        bubble.className = "message-bubble assistant";
        bubble.innerHTML = `
            <div class="msg-header" style="color: #ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> Hata</div>
            <div class="msg-content" style="border-color: #ef4444;">${escapeHtml(msg)}</div>
        `;
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // --- 5. PROMPT INSPECTOR ---
    btnToggleInspector.addEventListener("click", () => {
        inspectorDrawer.classList.toggle("hidden");
    });

    btnCloseInspector.addEventListener("click", () => {
        inspectorDrawer.classList.add("hidden");
    });

    function updateInspector(data) {
        if (!data || !data.prompt_inspector) return;

        const pi = data.prompt_inspector;
        inspectorContent.innerHTML = `
<div style="color: var(--accent-cyan); font-weight: bold; margin-bottom: 4px;">// SYSTEM PROMPT</div>
<div style="color: var(--text-muted); margin-bottom: 12px;">${escapeHtml(pi.system_prompt)}</div>

<div style="color: var(--accent-cyan); font-weight: bold; margin-bottom: 4px;">// INJECTED CONTEXT CHUNKS FROM SQLITE (${pi.injected_context_count})</div>
${pi.chunks_used.map(c => `
<div style="border-left: 2px solid var(--accent-cyan); padding-left: 8px; margin-bottom: 8px;">
    <div style="color: #10b981;">[Source: ${c.source} | Similarity Score: ${c.score}]</div>
    <div style="color: var(--text-dark);">${escapeHtml(c.snippet)}</div>
</div>
`).join("")}
        `;
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initial Load
    fetchSystemStatus();
    fetchDocuments();
});

const $ = id => document.getElementById(id);

async function api(path, opts = {}) {
    const r = await fetch(path, { ...opts, headers: { "Content-Type": "application/json", ...(opts.headers || {}) } });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
}

const ROW = "flex items-center justify-between border border-neutral-200 rounded-lg p-3 gap-4";
const BTN_RED = "text-xs px-3 py-1 rounded-md border border-neutral-300 text-red-600 hover:bg-red-50 transition-colors";
const BTN_GO = "text-xs px-3 py-1 rounded-md border border-neutral-300 hover:bg-neutral-50 transition-colors";

let providers = [];
let models = [];
let chain = [];
let detectState = { providerId: null, providerName: "", models: [], checked: new Set() };

function esc(value) {
    return String(value ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function js(value) {
    return esc(JSON.stringify(String(value ?? "")));
}

async function renderProviders() {
    const res = await api("/api/models/providers");
    providers = res.providers;
    $("providersList").innerHTML = providers.length === 0
        ? '<p class="text-neutral-400 text-sm">No providers yet.</p>'
        : providers.map(p => `
            <div class="${ROW}">
                <div>
                    <div class="font-medium text-sm">${esc(p.name)} <span class="text-xs text-neutral-400">${esc(p.provider_type)}</span></div>
                    <div class="text-xs text-neutral-400">${esc(p.base_url || "default endpoint")}</div>
                </div>
                <div class="flex gap-2">
                    <button onclick="detectModels(${js(p.id)}, ${js(p.name)})" class="${BTN_GO}">Detect models</button>
                    <button onclick="deleteProvider(${js(p.id)})" class="${BTN_RED}">Delete</button>
                </div>
            </div>`).join("");

    $("modelProvider").innerHTML = providers.length === 0
        ? '<option value="">Add provider first</option>'
        : providers.map(p => `<option value="${p.id}">${esc(p.name)}</option>`).join("");
}

async function deleteProvider(id) {
    if (!confirm("Delete this provider and all its models?")) return;
    await api(`/api/models/providers/${id}`, { method: "DELETE" });
    await renderAll();
}

async function detectModels(providerId, providerName) {
    try {
        const { models } = await api(`/api/models/providers/${providerId}/detect`);
        detectState = { providerId, providerName, models, checked: new Set() };
        $("modalTitle").textContent = `Models from ${providerName}`;
        $("modelFilter").value = "";
        $("modalSelectAll").checked = false;
        renderModalList();
        $("modelModal").classList.remove("hidden");
    } catch (e) {
        alert(`Detection failed: ${e.message}`);
    }
}

function filteredModels() {
    const q = $("modelFilter").value.trim().toLowerCase();
    return detectState.models.filter(m => !q || m.id.toLowerCase().includes(q) || (m.display_name || "").toLowerCase().includes(q));
}

function renderModalList() {
    const rows = filteredModels();
    $("modalList").innerHTML = rows.length === 0
        ? '<p class="text-neutral-400 text-sm py-2">No models match.</p>'
        : rows.map(m => `
            <label class="flex items-center gap-2.5 py-1.5 px-2 rounded-md hover:bg-neutral-50 cursor-pointer text-sm">
                <input type="checkbox" ${detectState.checked.has(m.id) ? "checked" : ""} onchange="toggleModel(${js(m.id)}, this.checked)">
                <span class="font-mono">${esc(m.id)}</span>
            </label>`).join("");
    $("modalCount").textContent = `${detectState.checked.size} selected · ${detectState.models.length} available`;
}

function toggleModel(id, on) {
    on ? detectState.checked.add(id) : detectState.checked.delete(id);
    renderModalList();
}

function toggleSelectAll(on) {
    filteredModels().map(m => m.id).forEach(id => on ? detectState.checked.add(id) : detectState.checked.delete(id));
    renderModalList();
}

function closeModal() {
    $("modelModal").classList.add("hidden");
}

async function addSelectedModels() {
    const picked = detectState.models.filter(m => detectState.checked.has(m.id));
    if (picked.length === 0) return closeModal();
    for (const m of picked) {
        await api("/api/models/", {
            method: "POST",
            body: JSON.stringify({ provider_id: detectState.providerId, display_name: m.display_name, model_id: m.id }),
        });
    }
    closeModal();
    await renderAll();
}

function closeCodexModal() {
    $("codexModal").classList.add("hidden");
}

async function openCodexModal() {
    $("codexCallbackUrl").value = "";
    $("codexOpenLogin").disabled = true;
    $("codexOpenLogin").onclick = null;
    $("codexModal").classList.remove("hidden");
    try {
        const { auth_url } = await api("/api/codex/auth-url");
        $("codexOpenLogin").disabled = false;
        $("codexOpenLogin").onclick = () => window.open(auth_url, "_blank");
    } catch (err) {
        closeCodexModal();
        alert(`Failed to start Codex auth: ${err.message}`);
    }
}

async function connectCodex() {
    const callbackUrl = $("codexCallbackUrl").value.trim();
    if (!callbackUrl) return alert("Paste the callback URL first.");
    const btn = $("codexConnect");
    const original = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Connecting…";
    try {
        const provider = await api("/api/codex/exchange", { method: "POST", body: JSON.stringify({ callback_url: callbackUrl }) });
        closeCodexModal();
        await renderAll();
        await detectModels(provider.id, provider.name);
    } catch (err) {
        alert(`Failed to connect Codex: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = original;
    }
}

$("connectCodex").addEventListener("click", openCodexModal);
$("codexConnect").addEventListener("click", connectCodex);

$("providerForm").addEventListener("submit", async e => {
    e.preventDefault();
    const body = Object.fromEntries(new FormData(e.target).entries());
    if (!body.base_url) body.base_url = null;
    try {
        await api("/api/models/providers", { method: "POST", body: JSON.stringify(body) });
        e.target.reset();
        await renderAll();
    } catch (err) {
        alert(`Failed: ${err.message}`);
    }
});

async function renderModels() {
    const res = await api("/api/models/");
    models = res.models;
    $("modelsList").innerHTML = models.length === 0
        ? '<p class="text-neutral-400 text-sm">No models yet. Use “Detect models” on a provider above.</p>'
        : models.map(m => `
            <div class="${ROW}">
                <div>
                    <div class="font-medium text-sm">${esc(m.display_name)} ${m.is_default ? '<span class="text-xs text-neutral-400">default</span>' : ''}</div>
                    <div class="text-xs text-neutral-400">${esc(m.provider_name)} · ${esc(m.model_id)}</div>
                </div>
                <div class="flex gap-2">
                    <button onclick="testModel(${js(m.id)})" class="${BTN_GO}">Test</button>
                    <button onclick="deleteModel(${js(m.id)})" class="${BTN_RED}">Delete</button>
                </div>
            </div>`).join("");
}

$("modelForm").addEventListener("submit", async e => {
    e.preventDefault();
    const form = new FormData(e.target);
    const body = {
        provider_id: form.get("provider_id"),
        display_name: form.get("display_name"),
        model_id: form.get("model_id"),
        enabled: form.get("enabled") === "on",
        is_default: form.get("is_default") === "on",
    };
    try {
        await api("/api/models/", { method: "POST", body: JSON.stringify(body) });
        e.target.reset();
        await renderAll();
    } catch (err) {
        alert(`Failed: ${err.message}`);
    }
});

async function testModel(id) {
    try {
        await api(`/api/models/${id}/test`, { method: "POST" });
        alert("Model replied pong.");
    } catch (err) {
        alert(`Test failed: ${err.message}`);
    }
}

async function deleteModel(id) {
    if (!confirm("Delete this model?")) return;
    await api(`/api/models/${id}`, { method: "DELETE" });
    chain = chain.filter(x => x.model_id !== id);
    await api("/api/models/chain", { method: "PUT", body: JSON.stringify(chain) });
    await renderAll();
}

function chainModel(entry) {
    return models.find(x => x.id === entry.model_id);
}

function modelProviderBadge(model) {
    return `${esc(model.provider_name)} · ${esc(model.provider_type)}`;
}

function setChainStatus(text, tone = "neutral") {
    const el = $("chainStatus");
    if (!el) return;
    el.textContent = text;
    el.className = `text-xs ${tone === "error" ? "text-red-500" : tone === "saving" ? "text-neutral-900" : "text-neutral-400"}`;
}

function normalizeChainEntry(entry) {
    return {
        model_id: entry.model_id,
        timeout: Math.max(1, parseInt(entry.timeout) || 120),
        retries: Math.max(0, parseInt(entry.retries) || 0),
    };
}

async function saveChainNow() {
    chain = chain.map(normalizeChainEntry);
    setChainStatus("Saving…", "saving");
    try {
        await api("/api/models/chain", { method: "PUT", body: JSON.stringify(chain) });
        setChainStatus("Saved");
    } catch (err) {
        setChainStatus("Save failed", "error");
        alert(`Failed to save chain: ${err.message}`);
    }
}

async function loadChain() {
    const res = await api("/api/models/chain");
    chain = res.chain.map(normalizeChainEntry);
    renderChain();
}

function renderChain() {
    const activeIds = new Set(chain.map(e => e.model_id));
    const available = models.filter(m => m.enabled && !activeIds.has(m.id));

    $("chainList").innerHTML = chain.length === 0
        ? '<div class="border border-dashed border-neutral-300 rounded-lg p-4 text-sm text-neutral-400">No fallback chain yet. Add one model from the right.</div>'
        : chain.map((entry, i) => {
            const m = chainModel(entry);
            return `
                <div class="border border-neutral-200 rounded-lg p-3">
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="text-xs text-neutral-400 font-mono mb-1">${i + 1}</div>
                            <div class="font-medium text-sm truncate">${esc(m ? m.display_name : entry.model_id)}</div>
                            <div class="text-xs text-neutral-400 truncate">${m ? modelProviderBadge(m) : "missing model"}</div>
                        </div>
                        <div class="flex gap-1 shrink-0">
                            <button onclick="moveChain(${i}, -1)" class="${BTN_GO}" ${i === 0 ? "disabled" : ""}>↑</button>
                            <button onclick="moveChain(${i}, 1)" class="${BTN_GO}" ${i === chain.length - 1 ? "disabled" : ""}>↓</button>
                            <button onclick="removeChain(${i})" class="${BTN_RED}">Remove</button>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-2 mt-3 text-xs text-neutral-500">
                        <label>timeout seconds
                            <input value="${entry.timeout}" type="number" min="1" onchange="setChainField(${i}, 'timeout', this.value)" class="mt-1 w-full px-2 py-1.5 rounded border border-neutral-300 text-neutral-900">
                        </label>
                        <label>retries
                            <input value="${entry.retries}" type="number" min="0" onchange="setChainField(${i}, 'retries', this.value)" class="mt-1 w-full px-2 py-1.5 rounded border border-neutral-300 text-neutral-900">
                        </label>
                    </div>
                </div>`;
        }).join("");

    $("chainAvailable").innerHTML = available.length === 0
        ? '<p class="text-neutral-400 text-sm">All enabled models are already in the chain.</p>'
        : available.map(m => `
            <button onclick="addToChain(${js(m.id)})" class="w-full text-left border border-neutral-200 rounded-lg p-3 hover:border-neutral-900 hover:bg-neutral-50 transition-colors">
                <div class="font-medium text-sm truncate">${esc(m.display_name)}</div>
                <div class="text-xs text-neutral-400 truncate">${modelProviderBadge(m)}</div>
                <div class="text-xs font-mono text-neutral-400 truncate mt-1">${esc(m.model_id)}</div>
            </button>`).join("");
}

async function addToChain(modelId) {
    if (chain.some(e => e.model_id === modelId)) return;
    chain.push({ model_id: modelId, timeout: 120, retries: 1 });
    renderChain();
    await saveChainNow();
}

async function setChainField(i, field, value) {
    chain[i][field] = Math.max(field === "retries" ? 0 : 1, parseInt(value) || 0);
    renderChain();
    await saveChainNow();
}

async function moveChain(i, direction) {
    const j = i + direction;
    if (j < 0 || j >= chain.length) return;
    [chain[i], chain[j]] = [chain[j], chain[i]];
    renderChain();
    await saveChainNow();
}

async function removeChain(i) {
    chain.splice(i, 1);
    renderChain();
    await saveChainNow();
}

async function renderAll() {
    await renderProviders();
    await renderModels();
    await loadChain();
}

renderAll();
window.detectModels = detectModels;
window.deleteProvider = deleteProvider;
window.renderModalList = renderModalList;
window.toggleModel = toggleModel;
window.toggleSelectAll = toggleSelectAll;
window.closeModal = closeModal;
window.addSelectedModels = addSelectedModels;
window.closeCodexModal = closeCodexModal;
window.testModel = testModel;
window.deleteModel = deleteModel;
window.addToChain = addToChain;
window.setChainField = setChainField;
window.moveChain = moveChain;
window.removeChain = removeChain;

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
                    <button onclick="detectModels('${p.id}', '${esc(p.name)}')" class="${BTN_GO}">Detect models</button>
                    <button onclick="deleteProvider('${p.id}')" class="${BTN_RED}">Delete</button>
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
                <input type="checkbox" ${detectState.checked.has(m.id) ? "checked" : ""} onchange="toggleModel('${esc(m.id)}', this.checked)">
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
                    <button onclick="testModel('${m.id}')" class="${BTN_GO}">Test</button>
                    <button onclick="deleteModel('${m.id}')" class="${BTN_RED}">Delete</button>
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

async function renderChain() {
    const res = await api("/api/models/chain");
    chain = res.chain;
    $("chainAdd").innerHTML = '<option value="">+ Add model to chain</option>' + models
        .filter(m => !chain.some(e => e.model_id === m.id))
        .map(m => `<option value="${m.id}">${esc(m.display_name)} (${esc(m.provider_name)})</option>`)
        .join("");

    $("chainList").innerHTML = chain.length === 0
        ? '<p class="text-neutral-400 text-sm">No fallback chain. Add one model or mark a model as default.</p>'
        : chain.map((entry, i) => {
            const m = models.find(x => x.id === entry.model_id);
            return `
                <div class="${ROW}">
                    <div class="text-sm flex-1"><span class="text-neutral-400 font-mono">${i + 1}.</span> ${esc(m ? m.display_name : entry.model_id)}</div>
                    <div class="flex items-center gap-2 text-xs text-neutral-500">
                        <label>timeout <input value="${entry.timeout}" type="number" min="1" onchange="setChainField(${i}, 'timeout', this.value)" class="w-16 px-2 py-1 rounded border border-neutral-300"></label>
                        <label>retries <input value="${entry.retries}" type="number" min="0" onchange="setChainField(${i}, 'retries', this.value)" class="w-14 px-2 py-1 rounded border border-neutral-300"></label>
                        <button onclick="moveChain(${i}, -1)" class="${BTN_GO}">↑</button>
                        <button onclick="moveChain(${i}, 1)" class="${BTN_GO}">↓</button>
                        <button onclick="removeChain(${i})" class="${BTN_RED}">Remove</button>
                    </div>
                </div>`;
        }).join("");
}

$("chainAdd").addEventListener("change", e => {
    if (!e.target.value) return;
    chain.push({ model_id: e.target.value, timeout: 120, retries: 1 });
    renderChain();
});

function setChainField(i, field, value) {
    chain[i][field] = Math.max(field === "retries" ? 0 : 1, parseInt(value) || 0);
}

function moveChain(i, direction) {
    const j = i + direction;
    if (j < 0 || j >= chain.length) return;
    [chain[i], chain[j]] = [chain[j], chain[i]];
    renderChain();
}

function removeChain(i) {
    chain.splice(i, 1);
    renderChain();
}

$("saveChain").addEventListener("click", async () => {
    await api("/api/models/chain", { method: "PUT", body: JSON.stringify(chain) });
    alert("Chain saved.");
});

async function renderAll() {
    await renderProviders();
    await renderModels();
    await renderChain();
}

renderAll();
window.detectModels = detectModels;
window.deleteProvider = deleteProvider;
window.renderModalList = renderModalList;
window.toggleModel = toggleModel;
window.toggleSelectAll = toggleSelectAll;
window.closeModal = closeModal;
window.addSelectedModels = addSelectedModels;
window.testModel = testModel;
window.deleteModel = deleteModel;
window.setChainField = setChainField;
window.moveChain = moveChain;
window.removeChain = removeChain;

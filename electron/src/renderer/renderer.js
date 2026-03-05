const baseUrl = window.researchBotApi.baseUrl;

const state = {
  history: [],
  controller: null,
  settings: {
    enable_search: true,
    enable_thinking_mode: true,
    show_thinking: true,
    temperature: 0.6,
    max_tokens: 8192,
  },
  activeModel: '',
  models: [],
};

const el = {
  chatLog: document.getElementById('chatLog'),
  thinking: document.getElementById('thinking'),
  status: document.getElementById('status'),
  contextUsage: document.getElementById('contextUsage'),
  messageInput: document.getElementById('messageInput'),
  sendBtn: document.getElementById('sendBtn'),
  stopBtn: document.getElementById('stopBtn'),
  clearBtn: document.getElementById('clearBtn'),
  enableSearch: document.getElementById('enableSearch'),
  enableThinkingMode: document.getElementById('enableThinkingMode'),
  showThinking: document.getElementById('showThinking'),
  temperature: document.getElementById('temperature'),
  temperatureValue: document.getElementById('temperatureValue'),
  maxTokens: document.getElementById('maxTokens'),
  maxTokensValue: document.getElementById('maxTokensValue'),
  saveSettingsBtn: document.getElementById('saveSettingsBtn'),
  tabChatBtn: document.getElementById('tabChatBtn'),
  tabModelsBtn: document.getElementById('tabModelsBtn'),
  chatTabPanel: document.getElementById('chatTabPanel'),
  modelsTabPanel: document.getElementById('modelsTabPanel'),
  modelList: document.getElementById('modelList'),
  refreshModelsBtn: document.getElementById('refreshModelsBtn'),
  unloadModelBtn: document.getElementById('unloadModelBtn'),
  switchModelSelect: document.getElementById('switchModelSelect'),
  switchModelBtn: document.getElementById('switchModelBtn'),
  downloadModelSelect: document.getElementById('downloadModelSelect'),
  downloadModelBtn: document.getElementById('downloadModelBtn'),
  downloadProgress: document.getElementById('downloadProgress'),
};

function appendMessage(role, content) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = content;
  el.chatLog.appendChild(div);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
  return div;
}

function replaceOrAppendAssistant(content) {
  const last = el.chatLog.lastElementChild;
  if (last && last.classList.contains('assistant')) {
    last.textContent = content;
    el.chatLog.scrollTop = el.chatLog.scrollHeight;
    return;
  }
  appendMessage('assistant', content);
}

function updateStatus(status, contextUsage) {
  if (status) el.status.textContent = status;
  if (contextUsage) el.contextUsage.textContent = contextUsage;
}

function syncSettingsToUi() {
  el.enableSearch.checked = state.settings.enable_search;
  el.enableThinkingMode.checked = state.settings.enable_thinking_mode;
  el.showThinking.checked = state.settings.show_thinking;
  el.temperature.value = String(state.settings.temperature);
  el.temperatureValue.textContent = String(state.settings.temperature);
  el.maxTokens.value = String(state.settings.max_tokens);
  el.maxTokensValue.textContent = String(state.settings.max_tokens);
}

function readSettingsFromUi() {
  state.settings = {
    enable_search: el.enableSearch.checked,
    enable_thinking_mode: el.enableThinkingMode.checked,
    show_thinking: el.showThinking.checked,
    temperature: Number(el.temperature.value),
    max_tokens: Number(el.maxTokens.value),
  };
}

async function saveSettings() {
  readSettingsFromUi();
  const res = await fetch(`${baseUrl}/api/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state.settings),
  });
  if (!res.ok) {
    throw new Error('設定保存に失敗しました');
  }
}

function refreshModelSelectors() {
  el.switchModelSelect.innerHTML = '';
  el.downloadModelSelect.innerHTML = '';

  for (const model of state.models) {
    const switchOpt = document.createElement('option');
    switchOpt.value = model.key;
    switchOpt.textContent = `${model.key}${model.downloaded ? '' : ' (未DL)'}`;
    if (model.active) switchOpt.selected = true;
    el.switchModelSelect.appendChild(switchOpt);

    const downloadOpt = document.createElement('option');
    downloadOpt.value = model.key;
    downloadOpt.textContent = model.key;
    el.downloadModelSelect.appendChild(downloadOpt);
  }
}

function renderModels() {
  const rows = state.models.map((m) => {
    const flags = [m.downloaded ? 'DL済み' : '未DL', m.active ? '使用中' : ''];
    return `• ${m.key} | ${m.size_gb}GB | ${m.vram_gb}GB | ${flags.filter(Boolean).join(' / ')}\n  ${m.description}`;
  });
  el.modelList.textContent = rows.join('\n\n') || 'モデル情報なし';
  refreshModelSelectors();
}

function setActiveTab(tabName) {
  const isChat = tabName === 'chat';
  el.tabChatBtn.classList.toggle('is-active', isChat);
  el.tabModelsBtn.classList.toggle('is-active', !isChat);
  el.chatTabPanel.classList.toggle('is-active', isChat);
  el.modelsTabPanel.classList.toggle('is-active', !isChat);
  el.tabChatBtn.setAttribute('aria-selected', String(isChat));
  el.tabModelsBtn.setAttribute('aria-selected', String(!isChat));
}

async function loadBootstrap() {
  const res = await fetch(`${baseUrl}/api/bootstrap`);
  if (!res.ok) {
    throw new Error('初期情報の取得に失敗しました');
  }
  const data = await res.json();

  state.settings = {
    enable_search: data.settings.enable_search ?? data.defaults.enable_search,
    enable_thinking_mode: data.settings.enable_thinking_mode ?? true,
    show_thinking: data.settings.show_thinking ?? data.defaults.show_thinking,
    temperature: data.settings.temperature ?? data.defaults.temperature,
    max_tokens: data.settings.max_tokens ?? data.defaults.max_tokens,
  };
  state.activeModel = data.active_model_key || '';
  state.models = data.models || [];

  syncSettingsToUi();
  renderModels();
}

async function loadModels() {
  const res = await fetch(`${baseUrl}/api/models`);
  if (!res.ok) {
    throw new Error('モデル一覧取得に失敗しました');
  }
  const data = await res.json();
  state.models = data.models || [];
  renderModels();
}

async function streamJsonPost(url, payload, onData) {
  const controller = new AbortController();
  state.controller = controller;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: controller.signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';

    for (const part of parts) {
      const line = part
        .split('\n')
        .map((v) => v.trim())
        .find((v) => v.startsWith('data:'));
      if (!line) continue;
      const json = line.slice(5).trim();
      if (!json) continue;
      onData(JSON.parse(json));
    }
  }

  state.controller = null;
}

async function sendMessage() {
  const message = el.messageInput.value.trim();
  if (!message) return;

  appendMessage('user', message);
  replaceOrAppendAssistant('');
  el.thinking.textContent = '';
  el.messageInput.value = '';

  const payload = {
    message,
    history: state.history,
    enable_search: el.enableSearch.checked,
    show_thinking: el.showThinking.checked,
    enable_thinking_mode: el.enableThinkingMode.checked,
    temperature: Number(el.temperature.value),
    max_tokens: Number(el.maxTokens.value),
  };

  try {
    await streamJsonPost(`${baseUrl}/api/chat/stream`, payload, (event) => {
      updateStatus(event.status, event.context_usage);

      if (typeof event.answer === 'string') {
        replaceOrAppendAssistant(event.answer);
      }

      if (typeof event.thinking === 'string') {
        // details内のmarkdown/HTMLタグはそのまま表示せずテキストで扱う
        el.thinking.textContent = event.thinking.replace(/<[^>]+>/g, '');
      }

      if (event.event === 'final') {
        state.history.push({ role: 'user', content: message });
        state.history.push({ role: 'assistant', content: event.answer || '' });
      }
    });
  } catch (err) {
    if (err.name === 'AbortError') {
      updateStatus('⏹ 停止しました');
      return;
    }
    replaceOrAppendAssistant(`エラー: ${err.message}`);
    updateStatus('❌ エラー');
  }
}

async function switchModel() {
  const model_key = el.switchModelSelect.value;
  if (!model_key) return;

  updateStatus('🔄 モデル切り替え中...');
  const res = await fetch(`${baseUrl}/api/models/switch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_key }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'モデル切り替え失敗');

  updateStatus(`✅ ${data.message}`);
  await loadModels();
}

async function unloadModel() {
  const res = await fetch(`${baseUrl}/api/models/unload`, { method: 'POST' });
  const data = await res.json();
  updateStatus(data.ok ? `✅ ${data.message}` : `⚠ ${data.message}`);
  await loadModels();
}

async function downloadModel() {
  const model_key = el.downloadModelSelect.value;
  if (!model_key) return;

  el.downloadProgress.textContent = '';

  await streamJsonPost(
    `${baseUrl}/api/models/download/stream`,
    { model_key },
    (event) => {
      if (event.event === 'progress') {
        el.downloadProgress.textContent += `${event.message}\n`;
        el.downloadProgress.scrollTop = el.downloadProgress.scrollHeight;
      }
      if (event.event === 'done') {
        state.models = event.models || state.models;
        renderModels();
      }
    }
  );
}

function bindEvents() {
  el.tabChatBtn.addEventListener('click', () => setActiveTab('chat'));
  el.tabModelsBtn.addEventListener('click', () => setActiveTab('models'));

  el.sendBtn.addEventListener('click', sendMessage);
  el.messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  el.stopBtn.addEventListener('click', () => {
    if (state.controller) {
      state.controller.abort();
      state.controller = null;
    }
  });

  el.clearBtn.addEventListener('click', () => {
    state.history = [];
    el.chatLog.innerHTML = '';
    el.thinking.textContent = '';
    updateStatus('待機中', '計算待ち');
  });

  el.temperature.addEventListener('input', () => {
    el.temperatureValue.textContent = el.temperature.value;
  });
  el.maxTokens.addEventListener('input', () => {
    el.maxTokensValue.textContent = el.maxTokens.value;
  });

  el.saveSettingsBtn.addEventListener('click', async () => {
    try {
      await saveSettings();
      updateStatus('✅ 設定を保存しました');
    } catch (err) {
      updateStatus(`❌ ${err.message}`);
    }
  });

  el.refreshModelsBtn.addEventListener('click', async () => {
    try {
      await loadModels();
      updateStatus('✅ モデル一覧更新');
    } catch (err) {
      updateStatus(`❌ ${err.message}`);
    }
  });

  el.switchModelBtn.addEventListener('click', async () => {
    try {
      await switchModel();
    } catch (err) {
      updateStatus(`❌ ${err.message}`);
    }
  });

  el.unloadModelBtn.addEventListener('click', async () => {
    try {
      await unloadModel();
    } catch (err) {
      updateStatus(`❌ ${err.message}`);
    }
  });

  el.downloadModelBtn.addEventListener('click', async () => {
    try {
      await downloadModel();
      updateStatus('✅ ダウンロード処理終了');
    } catch (err) {
      updateStatus(`❌ ${err.message}`);
    }
  });
}

async function init() {
  bindEvents();
  setActiveTab('chat');
  await loadBootstrap();
  updateStatus('待機中', '計算待ち');
}

init().catch((err) => {
  updateStatus(`❌ 初期化失敗: ${err.message}`);
});

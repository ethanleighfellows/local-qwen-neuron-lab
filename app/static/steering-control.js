// Small local profile control added alongside the unmodified llama.cpp UI.
(() => {
  const host = document.createElement('div');
  host.id = 'neuron-lab-control';
  const shadow = host.attachShadow({mode: 'open'});
  shadow.innerHTML = `
    <style>
      :host { position: fixed; z-index: 999; left: 12px; bottom: 12px; font: 14px system-ui, sans-serif; color-scheme: light dark; }
      .toggle, .panel { border: 1px solid #a5a5a5; border-radius: 12px; background: #fff; color: #202020; box-shadow: 0 3px 18px #0002; }
      .toggle { padding: 9px 13px; cursor: pointer; font: inherit; }
      .panel { width: min(310px, calc(100vw - 40px)); margin-bottom: 8px; padding: 14px; }
      .panel[hidden] { display: none; }
      h2 { font-size: 15px; margin: 0 0 12px; }
      label { display: block; margin: 10px 0; }
      select { width: 100%; font: inherit; color: inherit; background: transparent; padding: 7px; border: 1px solid #aaa; border-radius: 7px; }
      option { color: #202020; background: #fff; }
      .enabled { display: flex; gap: 8px; align-items: center; }
      ul { margin: 8px 0; padding-left: 20px; font: 12px ui-monospace, monospace; }
      p { margin: 8px 0; font-size: 12px; line-height: 1.4; }
      a { color: #146b5a; font-size: 13px; }
      .error { color: #b42525; }
      :host-context(.dark) .toggle, :host-context(.dark) .panel { background: #252525; color: #f4f4f4; border-color: #555; box-shadow: 0 3px 18px #0008; }
      :host-context(.dark) a { color: #8bdfca; }
      :host-context(.dark) .error { color: #ff9b9b; }
      @media (max-width: 700px) { :host { bottom: 80px; } }
    </style>
    <section class="panel" id="panel" role="region" aria-label="Neuron steering" hidden>
      <h2>Model and steering</h2>
      <label for="model">Model</label>
      <select id="model"></select>
      <p id="model-status"></p>
      <label for="profile">Global profile</label>
      <select id="profile"></select>
      <label class="enabled"><input type="checkbox" id="enabled"> Enable steering globally</label>
      <p id="status"></p><ul id="codes"></ul>
      <p class="error" id="error" role="alert"></p>
      <a href="/steering" target="_blank" rel="noopener">Manage profiles and neuron doses ↗</a>
    </section>
    <button class="toggle" id="toggle" aria-expanded="false" aria-controls="panel">Steering</button>`;
  document.body.append(host);
  const get = id => shadow.getElementById(id);
  let state = null, modelState = null, saving = false;

  async function fetchState() {
    const response = await fetch('/api/profiles');
    if (!response.ok) throw new Error(`Could not load profiles (${response.status})`);
    return response.json();
  }
  function render() {
    const profile = state.profiles[state.active_profile];
    const chosen = modelState?.models.find(m => m.id === modelState.selected);
    if (modelState) {
      get('model').replaceChildren(...modelState.models.map(m => {
        const label = m.steering ? 'Qwen3.5-4B (steering available)' : 'Qwen3.5-4B Uncensored (no steering)';
        const option = new Option(label + (m.available ? '' : ' ? not installed'), m.id);
        option.disabled = !m.available;
        return option;
      }));
      get('model').value = modelState.selected;
    }
    const steerable = chosen?.steering !== false;
    get('model-status').textContent = steerable ? 'Steering applies to this model.' : 'Steering is always off for this model.';
    get('profile').disabled = !steerable;
    get('enabled').disabled = !steerable;
    const select = get('profile');
    select.replaceChildren(...Object.keys(state.profiles).map(name => new Option(name, name)));
    select.value = state.active_profile;
    get('enabled').checked = state.enabled;
    const active = steerable && state.enabled && profile.enabled;
    get('status').textContent = active ? 'Active for web chat and VS Code' : 'Baseline: steering is off';
    get('toggle').textContent = `Steering · ${active ? state.active_profile : 'off'}`;
    get('codes').replaceChildren(...profile.steering.map(code => {
      const item = document.createElement('li');
      const dose = code.dose > 0 ? `+${code.dose}` : String(code.dose);
      item.textContent = `L${code.layer} ${String(code.neuron).padStart(5, '0')} ${dose}${code.enabled === false ? ' (off)' : ''}`;
      return item;
    }));
  }
  async function refresh() {
    if (saving) return;
    try {
      const [next, modelResponse] = await Promise.all([fetchState(), fetch('/api/models')]);
      if (!modelResponse.ok) throw new Error('Could not load models');
      const nextModels = await modelResponse.json();
      if (!state || next.version !== state.version || JSON.stringify(nextModels) !== JSON.stringify(modelState)) {
        state = next; modelState = nextModels; render();
      }
    } catch (error) { get('error').textContent = error.message; }
  }
  async function save(update) {
    if (!state || saving) return;
    saving = true;
    try {
      const next = structuredClone(state);
      update(next);
      const response = await fetch('/api/profiles', {method: 'PUT',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify(next)});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error?.message || 'Could not save profile');
      state = result;
      get('error').textContent = '';
      render();
    } catch (error) {
      get('error').textContent = error.message;
      try { state = await fetchState(); render(); } catch { /* keep the error visible */ }
    } finally { saving = false; }
  }
  get('toggle').addEventListener('click', () => {
    const opening = get('panel').hidden;
    get('panel').hidden = !opening;
    get('toggle').setAttribute('aria-expanded', String(opening));
    if (opening) refresh();
  });
  get('model').addEventListener('change', async event => {
    try {
      const response = await fetch('/api/model', {method: 'PUT', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model: event.target.value})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error?.message || 'Could not select model');
      location.reload();
    } catch (error) { get('error').textContent = error.message; }
  });
  get('profile').addEventListener('change', event => save(s => { s.active_profile = event.target.value; }));
  get('enabled').addEventListener('change', event => save(s => { s.enabled = event.target.checked; }));
  refresh();
  setInterval(refresh, 2500);
})();

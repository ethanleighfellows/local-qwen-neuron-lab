/* Integration test in the actual VS Code extension host using installed Continue. */
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
exports.run = async function () {
  const reportPath = path.resolve(__dirname, '../../data/continue-acceptance.json');
  const report = {success: false, client: 'Continue in VS Code extension host'};
  try {
    process.env.NODE_ENV = 'test';
    const extension = vscode.extensions.getExtension('Continue.continue');
    if (!extension) throw Error('Continue is not installed in this extension host');
    const api = await extension.activate();
    if (!api?.extension?.core) throw Error('Continue test-mode core API unavailable');
    const core = api.extension.core;
    const loaded = await core.configHandler.loadConfig();
    const config = loaded.config;
    if (!config) throw Error(JSON.stringify(loaded));
    const model = config.selectedModelByRole.chat;
    if (!model || model.model !== 'Qwen/Qwen3.5-4B') throw Error('Local model is not selected');
    const tool = config.tools.find(t => t.function.name === 'read_file');
    if (!tool?.readonly) throw Error('Expected Continue read-only file tool');
    const messages = [{role:'user',content:'Use read_file to read tests/vscode/fixture.txt. After the tool returns, tell me the acceptance marker.'}];
    const options = {temperature:0,maxTokens:180,tools:[tool]};
    const signal = new AbortController().signal;
    const chunks = [];
    for await (const chunk of model.streamChat(messages, signal, options)) chunks.push(chunk);
    report.firstChunks = chunks;
    const calls = chunks.flatMap(c => c.toolCalls || []);
    if (calls.length !== 1) throw Error('Expected exactly one native tool call');
    const call = calls[0];
    const args = JSON.parse(call.function.arguments);
    if (call.function.name !== 'read_file' || args.filepath !== 'tests/vscode/fixture.txt') throw Error('Unexpected tool target');
    // This read-only in-workspace tool uses Continue's normal default policy.
    const result = await core.handleToolCall(call);
    report.toolResult = result;
    const content = JSON.stringify(result);
    if (!content.includes('persistent local neuron steering')) throw Error('Continue did not return the fixture');
    messages.push({role:'assistant',content:chunks.map(c=>c.content||'').join(''),toolCalls:[call]},
      {role:'tool',toolCallId:call.id,content});
    const continuation = [];
    for await (const chunk of model.streamChat(messages,signal,options)) continuation.push(chunk);
    report.continuation = continuation;
    if (!continuation.some(c => c.content)) throw Error('Model did not continue after tool result');
    report.success = true;
  } catch (error) {
    report.error = error.stack || String(error);
    throw error;
  } finally {
    fs.writeFileSync(reportPath, JSON.stringify(report,null,2));
  }
};

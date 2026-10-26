// Run the original teacher worker against local assets, with JSON lines on stdio.
import { readFile, readdir } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

const assets = new URL('../../battlefield_page/az/', import.meta.url);
const workerDirectory = new URL('_app/immutable/workers/', assets);
const workers = (await readdir(workerDirectory)).filter(name => /^worker-.*\.js$/.test(name));
if (workers.length !== 1) throw new Error('Expected exactly one bundled teacher worker');

const allowedFiles = new Set(['weights.json', 'weights.bin', 'matmul.wasm']);
globalThis.fetch = async input => {
  const url = new URL(input);
  const name = fileURLToPath(url).slice(fileURLToPath(assets).length);
  if (!url.href.startsWith(assets.href) || !allowedFiles.has(name)) {
    throw new Error(`Teacher asset outside the bundled model: ${url.href}`);
  }
  return new Response(await readFile(url), {
    headers: { 'Content-Type': name.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream' },
  });
};

let events = [];
globalThis.self = { postMessage: event => events.push(event) };
console.log = console.error;
await import(new URL(workers[0], workerDirectory).href);

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of input) {
  let id = null;
  let response;
  try {
    const request = JSON.parse(line);
    id = request.id;
    if (!['init', 'notify', 'choose', 'reset'].includes(request.type)) {
      throw new Error(`Unsupported teacher request: ${request.type}`);
    }
    if (request.type === 'init') request.baseUrl = assets.href.replace(/\/$/, '');
    events = [];
    await self.onmessage({ data: request });
    const error = events.find(event => event.type === 'error');
    if (error) throw new Error(error.message);
    const result = events.find(event => event.type === 'choice' || event.type === 'ready');
    response = { id, ok: true, ...(result ?? { type: 'ack' }) };
  } catch (error) {
    response = { id, ok: false, error: error.message ?? String(error) };
  }
  // Include the Node process's cumulative CPU, which the Python referee cannot see.
  const cpu = process.cpuUsage();
  response.cpu_seconds = (cpu.user + cpu.system) / 1e6;
  process.stdout.write(JSON.stringify(response) + '\n');
}

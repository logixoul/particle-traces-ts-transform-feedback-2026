// Drive a real-WebGPU headless Chromium over raw CDP. No dependencies: Node 24
// ships a WebSocket client, and Playwright's Chromium is already on disk.
//
// As a library:
//   import { withPage } from './tools/cdp.mjs';
//   await withPage('http://localhost:5173/', async ({ evaluate }) => evaluate('1+1'));
//
// As a script (screenshot + console dump):
//   node tools/cdp.mjs http://localhost:5173/ shot.png
//
// See CLAUDE.md for why the flag list is so short -- the SwiftShader/ANGLE flags
// all *remove* the WebGPU adapter.

import { spawn } from 'node:child_process';
import { mkdtempSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

function findChrome() {
	if (process.env.CHROME_BIN) return process.env.CHROME_BIN;

	const root = join(process.env.USERPROFILE || process.env.HOME, 'AppData/Local/ms-playwright');
	const dirs = readdirSync(root)
		.filter((d) => /^chromium-\d+$/.test(d))
		.sort((a, b) => Number(b.split('-')[1]) - Number(a.split('-')[1]));

	if (dirs.length === 0) throw new Error(`no Playwright chromium under ${root}`);

	return join(root, dirs[0], 'chrome-win64/chrome.exe');
}

/**
 * Opens `url` in a fresh headless Chromium and hands your callback the page tools.
 * The browser is always torn down afterwards.
 *
 * @param {string} url
 * @param {(page: { evaluate: (expr: string) => Promise<any>, screenshot: () => Promise<string>, logs: string[] }) => Promise<any>} fn
 * @param {{ flags?: string[], settleMs?: number }} [options] settleMs waits for the render loop to spin up.
 */
export async function withPage(url, fn, { flags = [], settleMs = 4000 } = {}) {
	const port = 9500 + Math.floor(Math.random() * 400);
	const proc = spawn(findChrome(), [
		'--headless=new',
		`--remote-debugging-port=${port}`,
		`--user-data-dir=${mkdtempSync(join(tmpdir(), 'cdp-'))}`,
		'--no-first-run',
		'--no-sandbox',
		'--window-size=800,600',
		...flags,
		'about:blank',
	], { stdio: ['ignore', 'ignore', 'pipe'] });

	let stderr = '';
	proc.stderr.on('data', (d) => { stderr += d; });

	try {
		let version = null;
		for (let i = 0; i < 100 && !version; i++) {
			try {
				version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
			} catch { await new Promise((r) => setTimeout(r, 100)); }
		}
		if (!version) throw new Error(`chrome never came up:\n${stderr}`);

		const ws = new WebSocket(version.webSocketDebuggerUrl);
		await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

		let id = 0;
		const pending = new Map();
		const logs = [];

		ws.onmessage = (ev) => {
			const msg = JSON.parse(ev.data);
			if (msg.id !== undefined) {
				const p = pending.get(msg.id);
				pending.delete(msg.id);
				msg.error ? p.rej(new Error(JSON.stringify(msg.error))) : p.res(msg.result);
			} else if (msg.method === 'Runtime.consoleAPICalled') {
				logs.push(`[${msg.params.type}] `
					+ msg.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
			} else if (msg.method === 'Runtime.exceptionThrown') {
				logs.push('[exception] ' + (msg.params.exceptionDetails.exception?.description
					?? msg.params.exceptionDetails.text));
			}
		};

		const send = (method, params, sessionId) => new Promise((res, rej) => {
			const msgId = ++id;
			pending.set(msgId, { res, rej });
			ws.send(JSON.stringify({ id: msgId, method, params, sessionId }));
		});

		const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
		const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
		await send('Runtime.enable', {}, sessionId);
		await send('Page.enable', {}, sessionId);
		await send('Page.navigate', { url }, sessionId);
		await new Promise((r) => setTimeout(r, settleMs));

		const evaluate = async (expression) => {
			const res = await send('Runtime.evaluate', {
				expression, awaitPromise: true, returnByValue: true,
			}, sessionId);
			if (res.exceptionDetails) {
				throw new Error(res.exceptionDetails.exception?.description ?? res.exceptionDetails.text);
			}
			return res.result.value;
		};

		const screenshot = async () =>
			(await send('Page.captureScreenshot', { format: 'png' }, sessionId)).data;

		return await fn({ evaluate, screenshot, logs });
	} finally {
		proc.kill();
	}
}

// Run directly: screenshot a URL and print anything the page logged.
if (import.meta.filename === process.argv[1]) {
	const [url = 'http://localhost:5173/', out = 'shot.png'] = process.argv.slice(2);

	const logs = await withPage(url, async ({ evaluate, screenshot, logs }) => {
		writeFileSync(out, Buffer.from(await screenshot(), 'base64'));
		console.log(await evaluate(`document.getElementById('info')?.textContent ?? document.title`));
		return logs;
	});

	console.log(`wrote ${out}`);
	for (const line of logs) console.log(line);
}

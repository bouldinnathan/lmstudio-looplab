// SPDX-License-Identifier: AGPL-3.0-or-later
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {events, localAddress, call} = require('../dist/transport.js');
const {messages} = require('../dist/generator.js');
const {Chat} = require('@lmstudio/sdk');

test('local addresses cannot redirect credentials to remote hosts', () => {
  assert.equal(localAddress('http://127.0.0.1:7341/'), 'http://127.0.0.1:7341');
  for (const value of ['https://example.com','http://127.0.0.1.evil.test','http://user@localhost:7341','file:///tmp/x','http://localhost:7341/x']) {
    assert.throws(() => localAddress(value));
  }
});

test('SSE handles split UTF-8, CRLF, comments and done markers', async () => {
  const bytes = new TextEncoder().encode(': keepalive\r\n\r\ndata: {"x":"é"}\r\n\r\ndata: {"x":2}\n\ndata: [DONE]\n\n');
  const stream = new ReadableStream({start(c) { for (let i=0;i<bytes.length;i+=3) c.enqueue(bytes.slice(i,i+3)); c.close(); }});
  const result=[];
  for await (const event of events(stream)) result.push(event);
  assert.deepEqual(result,[{x:'é'},{x:2}]);
});

test('SSE propagates errors and rejects incomplete frames', async () => {
  for (const input of ['data: {"error":"failed"}\n\n','data: {"x":1}']) {
    const stream=new ReadableStream({start(c){c.enqueue(new TextEncoder().encode(input));c.close();}});
    await assert.rejects(async()=>{for await(const e of events(stream)) { void e; }});
  }
});

test('chat conversion retains system and user messages', () => {
  assert.deepEqual(messages(Chat.from([{role:'system',content:'Be precise'},{role:'user',content:'Hi'}])),
    [{role:'system',content:'Be precise'},{role:'user',content:'Hi'}]);
});

test('HTTP errors reach the user without silently returning empty output', async () => {
  const previous=global.fetch;
  global.fetch=async()=>new Response(JSON.stringify({error:'Unsupported model'}),{status:400});
  try { await assert.rejects(call('http://127.0.0.1:7341','test','/looplab/models',new AbortController().signal),/Unsupported model/); }
  finally {global.fetch=previous;}
});

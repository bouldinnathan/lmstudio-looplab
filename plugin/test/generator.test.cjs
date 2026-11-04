// SPDX-License-Identifier: AGPL-3.0-or-later
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {createServer} = require('node:http');
const {Chat} = require('@lmstudio/sdk');
const {generate} = require('../dist/generator.js');

async function companion(t, respond) {
  const requests=[];
  const server=createServer(async(req,res)=>{
    let text='';
    for await(const chunk of req) text+=chunk;
    const item={path:req.url,method:req.method,token:req.headers.authorization,body:text?JSON.parse(text):null};
    requests.push(item);
    try {await respond(item,res);}
    catch(error){res.writeHead(500);res.end(JSON.stringify({error:error.message}));}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  t.after(()=>new Promise(resolve=>server.close(resolve)));
  return {base:`http://127.0.0.1:${server.address().port}`,requests};
}

function json(res,value){res.setHeader('Content-Type','application/json');res.end(JSON.stringify(value));}
function controller(base,settings={},onFragment=()=>{}) {
  const config={action:'chat',model:'tiny',profile:'recommended',autoTest:true,quick:false,maxTokens:128,...settings};
  const abort=new AbortController();
  const fragments=[],tools=[];
  return {abort,fragments,tools,
    getPluginConfig:()=>({get:k=>config[k]}),
    getGlobalPluginConfig:()=>({get:k=>({baseUrl:base,accessToken:'test-token'}[k])}),
    abortSignal:abort.signal,
    fragmentGenerated:(text,opts)=>{fragments.push({text,opts});onFragment(text,abort);},
    getToolDefinitions:()=>[{type:'function',function:{name:'sum',description:'Add',parameters:{type:'object'}}}],
    toolCallGenerationStarted:()=>{},toolCallGenerationNameReceived:()=>{},toolCallGenerationArgumentFragmentGenerated:()=>{},
    toolCallGenerationEnded:value=>tools.push(value),toolCallGenerationFailed:error=>{throw error;}
  };
}

test('blank selection discovers models without launching inference',async t=>{
  const fake=await companion(t,(req,res)=>{
    assert.equal(req.path,'/looplab/refresh');
    json(res,{models:[{id:'tiny-id',name:'tiny',eligible:true,reasons:[]}]});
  });
  const ctl=controller(fake.base,{model:''});
  await generate(ctl,Chat.from([{role:'user',content:'Hello'}]));
  assert.match(ctl.fragments.map(f=>f.text).join(''),/tiny-id/);
  assert.equal(fake.requests.length,1);
  assert.equal(fake.requests[0].token,'Bearer test-token');
});

test('first use tests, reads the report and streams text, reasoning and tool calls',async t=>{
  let reports=0;
  const report={status:'tested',recommended_profile:'baseline',model:{name:'tiny'},profiles:[]};
  const fake=await companion(t,(req,res)=>{
    if(req.path==='/looplab/reports/tiny')return json(res,reports++?report:{status:'untested'});
    if(req.path==='/looplab/jobs')return json(res,{id:'job1'});
    if(req.path==='/looplab/jobs/job1')return json(res,{status:'complete',reports:[report]});
    assert.equal(req.path,'/v1/chat/completions');
    assert.equal(req.body.looplab_profile,'recommended');
    assert.equal(req.body.messages[0].content,'Use a tool');
    assert.equal(req.body.tools[0].function.name,'sum');
    res.setHeader('Content-Type','text/event-stream');
    for(const delta of [
      {reasoning_content:'Checking.'},{content:'Answer: '},
      {tool_calls:[{index:0,id:'call-1',function:{name:'sum',arguments:'{"a":'}}]},
      {tool_calls:[{index:0,function:{arguments:'2}'}}]},
    ])res.write(`data: ${JSON.stringify({choices:[{delta}]})}\n\n`);
    res.end('data: [DONE]\n\n');
  });
  const ctl=controller(fake.base);
  await generate(ctl,Chat.from([{role:'user',content:'Use a tool'}]));
  assert.ok(ctl.fragments.some(f=>f.text==='Checking.'&&f.opts.reasoningType==='reasoning'));
  assert.ok(ctl.fragments.some(f=>f.text==='Answer: '));
  assert.deepEqual(ctl.tools,[{type:'function',id:'call-1',name:'sum',arguments:{a:2}}]);
  assert.equal(fake.requests.filter(r=>r.path==='/looplab/jobs').length,1);
});

test('aborting a test also cancels the companion job',async t=>{
  const fake=await companion(t,(req,res)=>{
    if(req.path==='/looplab/reports/tiny')return json(res,{status:'untested'});
    if(req.path==='/looplab/jobs')return json(res,{id:'job2'});
    if(req.path==='/looplab/jobs/job2')return json(res,{status:'running',progress:'tiny: baseline',reports:[]});
    assert.equal(req.path,'/looplab/jobs/job2/cancel');
    json(res,{status:'cancellation_requested'});
  });
  const ctl=controller(fake.base,{action:'test-model'},(text,abort)=>{
    if(text.startsWith('tiny:'))abort.abort();
  });
  await assert.rejects(generate(ctl,Chat.from([{role:'user',content:'Test'}])),{name:'AbortError'});
  assert.ok(fake.requests.some(r=>r.path.endsWith('/cancel')));
  assert.ok(!fake.requests.some(r=>r.path==='/v1/chat/completions'));
});

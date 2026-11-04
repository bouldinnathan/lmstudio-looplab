#!/usr/bin/env python3
"""Compare complete tiny GGUF networks with an independent NumPy implementation.

Creates random F32 fixtures locally. No model downloads or training checkpoints.
Tests causal attention, RoPE, RMSNorm, SwiGLU, MoE route reuse, prefill and decode.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
import argparse
import json
import sys
import tempfile

import numpy as np
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looplab.models import inspect, Profile, atomic_json
from looplab.worker import Worker

D, H, HD, FF, VOCAB, LAYERS = 32, 2, 16, 48, 32, 8
EPS = 1e-5


def fixture(path, arch):
    rng = np.random.default_rng(73019)
    def weight(shape, scale=0.10):
        return (rng.standard_normal(shape) * scale).astype(np.float32)
    weights = {"token_embd.weight": weight((VOCAB,D), 0.3), "output.weight": weight((VOCAB,D), 0.2),
               "output_norm.weight": np.ones(D, np.float32)}
    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_name("LoopLab synthetic numerical fixture")
    writer.add_context_length(512)
    writer.add_embedding_length(D)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(FF)
    writer.add_head_count(H)
    writer.add_head_count_kv(H)
    writer.add_rope_dimension_count(HD)
    writer.add_layer_norm_rms_eps(EPS)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(["<unk>","<s>","</s>"] + [f"t{i}" for i in range(3,VOCAB)])
    writer.add_token_types([2,3,3] + [1]*(VOCAB-3))
    writer.add_token_scores([0.0]*VOCAB)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(False)
    if arch == "qwen3moe":
        writer.add_uint32("qwen3moe.expert_count", 4)
        writer.add_uint32("qwen3moe.expert_used_count", 2)
        writer.add_uint32("qwen3moe.expert_feed_forward_length", FF)
    for il in range(LAYERS):
        prefix = f"blk.{il}."
        for name in ["attn_norm", "ffn_norm"]:
            weights[prefix+name+".weight"] = np.ones(D, np.float32)
        for name in ["attn_q", "attn_k", "attn_v", "attn_output"]:
            weights[prefix+name+".weight"] = weight((D,D))
        if arch in {"qwen3", "qwen3moe"}:
            for name in ["attn_q_norm", "attn_k_norm"]:
                weights[prefix+name+".weight"] = np.ones(HD, np.float32)
        if arch == "qwen3moe":
            weights[prefix+"ffn_gate_inp.weight"] = weight((4,D), 0.2)
            for name in ["ffn_gate_exps", "ffn_up_exps"]:
                weights[prefix+name+".weight"] = weight((4,FF,D))
            weights[prefix+"ffn_down_exps.weight"] = weight((4,D,FF))
        else:
            for name in ["ffn_gate", "ffn_up"]:
                weights[prefix+name+".weight"] = weight((FF,D))
            weights[prefix+"ffn_down.weight"] = weight((D,FF))
    for name, value in weights.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {k: v.astype(np.float64) for k,v in weights.items()}


def norm(x):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+EPS)


def softmax(x):
    e=np.exp(x-np.max(x, axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)


def rope(x, positions, neox):
    angles = positions[:,None,None] / (10000.0 ** (np.arange(0, HD, 2)/HD))[None,None,:]
    cos, sin = np.cos(angles), np.sin(angles)
    y=x.copy()
    if neox:
        a,b=x[...,:HD//2],x[...,HD//2:]
        y[...,:HD//2]=a*cos-b*sin
        y[...,HD//2:]=a*sin+b*cos
    else:
        a,b=x[...,::2],x[...,1::2]
        y[...,::2]=a*cos-b*sin
        y[...,1::2]=a*sin+b*cos
    return y


class Reference:
    def __init__(self, weights, arch, profile):
        self.w,self.arch,self.profile=weights,arch,profile
        self.cache={}
        self.past=0

    def forward(self, tokens):
        count=len(tokens)
        positions=np.arange(self.past,self.past+count)
        prefix={i:(k[:self.past].copy(),v[:self.past].copy()) for i,(k,v) in self.cache.items()}
        x=self.w["token_embd.weight"][tokens].copy()
        def layer(x, il, routes=None):
            p=f"blk.{il}."
            z=norm(x)*self.w[p+"attn_norm.weight"]
            q=(z@self.w[p+"attn_q.weight"].T).reshape(count,H,HD)
            k=(z@self.w[p+"attn_k.weight"].T).reshape(count,H,HD)
            v=(z@self.w[p+"attn_v.weight"].T).reshape(count,H,HD)
            if self.arch in {"qwen3", "qwen3moe"}:
                q=norm(q)*self.w[p+"attn_q_norm.weight"]
                k=norm(k)*self.w[p+"attn_k_norm.weight"]
            q,k=rope(q,positions,self.arch!="llama"),rope(k,positions,self.arch!="llama")
            if il in prefix:
                pk,pv=prefix[il]
                k=np.concatenate([pk,k]);v=np.concatenate([pv,v])
            self.cache[il]=(k.copy(),v.copy())
            scores=np.einsum("thd,shd->hts",q,k)/np.sqrt(HD)
            mask=np.arange(len(k))[None,:] > positions[:,None]
            scores=np.where(mask[None,:,:],-np.inf,scores)
            attn=np.einsum("hts,shd->thd",softmax(scores),v).reshape(count,D)
            x=x+attn@self.w[p+"attn_output.weight"].T
            z=norm(x)*self.w[p+"ffn_norm.weight"]
            def expert(gate,up,down):
                g=z@gate.T
                return ((g/(1+np.exp(-g)))*(z@up.T))@down.T
            if self.arch == "qwen3moe":
                if routes is not None and il in routes:
                    ids,probs=routes[il]
                else:
                    probs=softmax(z@self.w[p+"ffn_gate_inp.weight"].T)
                    ids=np.argsort(-probs,axis=-1)[:,:2]
                    probs=np.take_along_axis(probs,ids,axis=-1)
                    probs=probs/probs.sum(axis=-1,keepdims=True)
                    if routes is not None: routes[il]=(ids,probs)
                out=np.zeros_like(x)
                for e in range(4):
                    value=expert(self.w[p+"ffn_gate_exps.weight"][e],self.w[p+"ffn_up_exps.weight"][e],self.w[p+"ffn_down_exps.weight"][e])
                    mass=np.sum(np.where(ids==e,probs,0),axis=1)
                    out+=value*mass[:,None]
                return x+out
            return x+expert(self.w[p+"ffn_gate.weight"],self.w[p+"ffn_up.weight"],self.w[p+"ffn_down.weight"])
        def group(x,begin,end,routes=None):
            for i in range(begin,end): x=layer(x,i,routes)
            return x
        c=self.profile
        if c.passes==1:
            x=group(x,0,LAYERS)
        else:
            x=group(x,0,c.start)
            groups=[(i,i+1) for i in range(c.start,c.end)] if c.mode=="layer" else [(c.start,c.end)]
            for begin,end in groups:
                routes={} if self.arch=="qwen3moe" else None
                natural=group(x,begin,end,routes)
                first={i:tuple(a.copy() for a in self.cache[i]) for i in range(begin,end)}
                refined=x+(natural-x)/c.passes
                for _ in range(1,c.passes):
                    y=group(refined,begin,end,routes)
                    refined=refined+(y-refined)/c.passes
                x=c.anchor*natural+(1-c.anchor)*refined
                if c.cache=="first": self.cache.update(first)
                else: group(x,begin,end)
            x=group(x,c.end,LAYERS)
        self.past+=count
        return (norm(x)*self.w["output_norm.weight"])@self.w["output.weight"].T


def validate(binary=None, gpu_layers=0, output=None):
    results=[]
    with tempfile.TemporaryDirectory(prefix="looplab-numerics-") as directory:
        root=Path(directory)
        worker=Worker(root/"logs",binary,context=512,gpu_layers=gpu_layers)
        try:
            for arch in ("llama","qwen2","qwen3","qwen3moe"):
                folder=root/arch
                folder.mkdir()
                path=folder/"fixture.gguf"
                weights=fixture(path,arch)
                model=inspect(path)
                for passes,beta,cache in [(1,0.5,"first"),(2,0.5,"first"),(3,0,"first"),(4,0.5,"first"),(2,1,"first"),(2,0.5,"last")]:
                    profile=Profile.for_model(model,passes,beta,cache)
                    ref=Reference(weights,arch,profile)
                    with worker.use(model,profile):
                        data=worker.json("/completion",{"prompt":[3,7,5],"n_predict":4,"temperature":-1,
                            "return_tokens":True,"n_probs":VOCAB,"cache_prompt":False,"seed":17})
                    predictions=data["completion_probabilities"]
                    logits=ref.forward([3,7,5])[-1]
                    errors=[]
                    for prediction in predictions:
                        expected=np.log(softmax(logits))
                        by_id={p["id"]:p["logprob"] for p in prediction["top_logprobs"]}
                        if len(by_id)!=VOCAB:
                            raise AssertionError(f"Incomplete distribution: {len(by_id)}/{VOCAB}")
                        error=max(abs(expected[i]-by_id[i]) for i in range(VOCAB))
                        errors.append(float(error))
                        if prediction["id"] != int(np.argmax(logits)):
                            raise AssertionError(f"Argmax differs: {arch} {profile.id}")
                        logits=ref.forward([prediction["id"]])[-1]
                    row={"architecture":arch,"profile":profile.id,"decode_steps":len(predictions),
                         "max_logprob_error":max(errors),"passed":max(errors)<0.005}
                    print(json.dumps(row),flush=True)
                    results.append(row)
                    if not row["passed"]:
                        raise AssertionError(f"Numerical reference mismatch: {row}")
        finally:
            worker.close()
    report={"reference":"independent_numpy_f32_v1","gpu_layers":gpu_layers,
            "runtime":worker.build_info(),"passed":all(r["passed"] for r in results),"checks":results}
    if output: atomic_json(output,report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path)
    p.add_argument("--gpu-layers",type=int,default=0)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    validate(args.binary,args.gpu_layers,args.output)

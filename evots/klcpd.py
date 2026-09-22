"""Modern research adaptation of KL-CPD (Chang et al., ICLR 2019).

Generator/critic architecture and objective follow OctoberChang/klcpd_code.
BSD notice is in THIRD_PARTY_LICENSES.md. Differences: exact RBF MMD instead
of random Fourier features, CPU determinism, training-only preprocessing,
fixed training epochs and robust peak extraction instead of ROC evaluation.
This is NOT asserted to reproduce that repository's numerical results.
"""
from __future__ import annotations

import numpy as np


def klcpd_scores(train, x, c, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    w = c["window"]
    if min(len(train),len(x)) <= 2*w:
        raise ValueError("KL-CPD needs more than twice window samples in train and evaluation splits")
    d,h = train.shape[1],c["hidden"]

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.GRU(d,h,batch_first=True)
            self.decoder=nn.GRU(d,h,batch_first=True)
            self.projection=nn.Linear(h,d)
        def forward(self,past,future):
            _,state=self.encoder(past)
            shifted=torch.cat([torch.zeros_like(future[:,:1]),future[:,:-1]],dim=1)
            result,_=self.decoder(shifted,state+torch.randn_like(state))
            return self.projection(result)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.GRU(d,h,batch_first=True)
            self.decoder=nn.GRU(h,d,batch_first=True)
        def forward(self,values):
            latent,_=self.encoder(values)
            reconstruction,_=self.decoder(latent)
            return latent,reconstruction

    subset=train[rng.choice(len(train),min(512,len(train)),replace=False)]
    sq=((subset[:,None]-subset[None,:])**2).sum(-1)
    positive=sq[sq>0]
    median=float(np.median(positive)) if len(positive) else 1.0
    variances=torch.tensor([median*f for f in [0.25,0.5,1,2,4]],dtype=torch.float32).clamp_min(1e-6)

    def kernel(a,b):
        distances=(a[:,:,None,:]-b[:,None,:,:]).square().sum(-1)
        return torch.exp(-distances[...,None]/(2*variances)).mean(-1)
    def mmd(a,b):
        return kernel(a,a).mean((1,2))+kernel(b,b).mean((1,2))-2*kernel(a,b).mean((1,2))
    def batch(a,centers):
        past=np.stack([a[t-w:t] for t in centers])
        future=np.stack([a[t:t+w] for t in centers])
        return torch.tensor(past,dtype=torch.float32),torch.tensor(future,dtype=torch.float32)

    generator,critic=Generator(),Critic()
    go=torch.optim.Adam(generator.parameters(),lr=c["learning_rate"])
    co=torch.optim.Adam(critic.parameters(),lr=c["learning_rate"])
    positions=np.arange(w,len(train)-w+1)
    steps=max(1,int(np.ceil(len(positions)/c["batch_size"])))
    for _ in range(c["epochs"]):
        for _ in range(steps):
            for param in critic.parameters(): param.requires_grad_(True)
            for _ in range(c["critic_steps"]):
                with torch.no_grad():
                    for param in critic.encoder.parameters(): param.clamp_(-0.1,0.1)
                past,future=batch(train,rng.choice(positions,c["batch_size"]))
                with torch.no_grad(): fake=generator(past,future)
                ep,_=critic(past); ef,rf=critic(future); eg,rg=critic(fake)
                objective=mmd(ef,eg).mean()-c["lambda_real"]*mmd(ep,ef).mean()-c["lambda_ae"]*((future-rf).square().mean()+(fake-rg).square().mean())
                co.zero_grad(); (-objective).backward()
                nn.utils.clip_grad_norm_(critic.parameters(),10.0); co.step()
            for param in critic.parameters(): param.requires_grad_(False)
            past,future=batch(train,rng.choice(positions,c["batch_size"]))
            with torch.no_grad(): ef,_=critic(future)
            eg,_=critic(generator(past,future))
            loss=mmd(ef,eg).mean()
            go.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(),10.0); go.step()
    critic.eval()
    valid=np.arange(w,len(x)-w+1)
    scores=np.zeros(len(x))
    with torch.no_grad():
        for start in range(0,len(valid),c["batch_size"]):
            centers=valid[start:start+c["batch_size"]]
            past,future=batch(x,centers)
            ep,_=critic(past); ef,_=critic(future)
            scores[centers]=mmd(ep,ef).numpy()
    if not np.isfinite(scores).all(): raise ValueError("KL-CPD produced nonfinite scores")
    return scores,valid

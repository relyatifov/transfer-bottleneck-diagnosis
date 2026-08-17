"""
Knowledge vs Integration Deficit: diagnosing the transfer bottleneck
in a model-based agent.

FINAL VERSION (addresses reviewer comments):
  R1: threshold theta is now selected on a TRAIN split and evaluated on a
      held-out TEST split (no more optimistic in-sample estimate).
  R1: K(X) fully specified; sensitivity to the coefficient c reported.
  R2: reproducibility - fixed seeds, all hyperparameters listed, code public.
  R2: ablation over p_open and population size.

Runtime: ~2-4 min on a laptop CPU. No GPU required.
"""
import numpy as np, copy, json

# =========================== HYPERPARAMETERS ===========================
HP = dict(
    grid_rows=5, grid_cols=7, start=(2,0), lever=(4,2), gap=(2,3),
    p_open=0.7,           # probability the lever opens the bridge
    gamma=0.97, horizon=90,
    eps_explore=0.25,     # epsilon during optimistic exploration
    c_coef=1.0,           # uncertainty penalty in K(X)
    n_agents=320,
    data_budgets=[2,3,4,6,10,16,24,40,70,120],   # exploration episodes
    plan_budgets=[1,2,3,4,6,10,25,300],          # value-iteration sweeps
    data_intervention_episodes=90,
    full_sweeps=300,
    rollouts_per_goal=12,
    fail_threshold=0.5,   # base success below this = deployment failure
    recover_threshold=0.7,# success above this after a fix = recovered
    train_frac=0.5,       # split for threshold selection
    n_bootstrap=2000,
    master_seed=0,
)
R,C = HP["grid_rows"], HP["grid_cols"]
START,LEVER,GAP = HP["start"],HP["lever"],HP["gap"]
WALLS = {(r,3) for r in range(R)} - {GAP}
FAR_GOALS = [(0,5),(1,4),(0,6),(4,4),(4,6),(2,6),(1,6),(3,5)]
ACTIONS = [(-1,0),(1,0),(0,1),(0,-1)]; A=4
GAMMA, HORIZON = HP["gamma"], HP["horizon"]

def states(): return [(r,c,b) for r in range(R) for c in range(C) for b in (0,1) if (r,c) not in WALLS]
S=states(); sidx={s:i for i,s in enumerate(S)}; NS=len(S)

def mv(s,a):
    r,c,b=s; dr,dc=ACTIONS[a]; nr,nc=r+dr,c+dc
    if not(0<=nr<R and 0<=nc<C): nr,nc=r,c
    elif (nr,nc) in WALLS: nr,nc=r,c
    elif (nr,nc)==GAP and b==0: nr,nc=r,c
    return nr,nc
def step(s,a,goal,rng,p_open):
    nr,nc=mv(s,a); b=s[2]; nb=b
    if (nr,nc)==LEVER and b==0: nb = 1 if rng.random()<p_open else 0
    ns=(nr,nc,nb); done=(nr,nc)==goal
    return ns,(1.0 if done else -0.01),done
def is_lever_sa(s,a):
    nr,nc=mv(s,a); return (nr,nc)==LEVER and s[2]==0

class Model:
    def __init__(self): self.Nsas=np.zeros((NS,A,NS)); self.Nsa=np.zeros((NS,A)); self.Rsum=np.zeros((NS,A))
    def upd(self,i,a,j,r): self.Nsas[i,a,j]+=1; self.Nsa[i,a]+=1; self.Rsum[i,a]+=r
def That(m): return m.Nsas/np.maximum(m.Nsa,1)[...,None]
def Rhat(m): return m.Rsum/np.maximum(m.Nsa,1)

def VI(m,goal,sweeps,optimistic=False):
    gidx=[sidx[(goal[0],goal[1],b)] for b in (0,1)]
    T=That(m); Rh=Rhat(m); known=m.Nsa>=1; VMAX=1/(1-GAMMA); V=np.zeros(NS)
    for _ in range(sweeps):
        Q=Rh+GAMMA*np.einsum('ijk,k->ij',T,V)
        Q=np.where(known,Q,VMAX) if optimistic else np.where(known,Q,-1e9)
        Vn=Q.max(1)
        for gi in gidx: Vn[gi]=0.0
        if np.max(np.abs(Vn-V))<1e-5: V=Vn; break
        V=Vn
    Q=Rh+GAMMA*np.einsum('ijk,k->ij',T,V)
    Q=np.where(known,Q,VMAX) if optimistic else np.where(known,Q,-1e9)
    return Q.argmax(1)

def explore(m,episodes,rng,p_open):
    for _ in range(episodes):
        pol=VI(m,FAR_GOALS[0],40,optimistic=True)
        s=(START[0],START[1],0)
        for _ in range(HORIZON):
            i=sidx[s]; a=pol[i] if rng.random()>HP["eps_explore"] else rng.integers(A)
            ns,r,done=step(s,a,FAR_GOALS[0],rng,p_open); m.upd(i,a,sidx[ns],r); s=ns
            if done: break

def K_of_X(m,c=None):
    """Lower-confidence estimate of the learned effect of X.
    Averaged over lever-approach transitions (s,a) with at least one observation:
        K = mean_{(s,a) in D(X)} max(0, p_hat_sa - c/sqrt(n_sa))
    where p_hat_sa = model's predicted prob. that the bridge is open after (s,a)."""
    if c is None: c=HP["c_coef"]
    T=That(m); vals=[]
    for s in S:
        for a in range(A):
            n=m.Nsa[sidx[s],a]
            if n>=1 and is_lever_sa(s,a):
                i=sidx[s]; phat=sum(T[i,a,sidx[t]] for t in S if t[2]==1)
                vals.append(max(0.0, phat - c/np.sqrt(n)))
    return float(np.mean(vals)) if vals else 0.0

def far_success(m,sweeps,rng,p_open):
    tot=0; n=0
    for g in FAR_GOALS:
        pol=VI(m,g,sweeps)
        for _ in range(HP["rollouts_per_goal"]):
            s=(START[0],START[1],0); ok=False
            for _ in range(HORIZON):
                a=pol[sidx[s]]; ns,r,done=step(s,a,g,rng,p_open); s=ns
                if done: ok=True; break
            tot+=ok; n+=1
    return tot/n

def make_agent(seed,p_open=None,c=None):
    if p_open is None: p_open=HP["p_open"]
    rng=np.random.default_rng(seed)
    db=int(rng.choice(HP["data_budgets"])); pb=int(rng.choice(HP["plan_budgets"]))
    m=Model(); explore(m,db,rng,p_open)
    Ks={cc: K_of_X(m,cc) for cc in [0.5,1.0,1.5,2.0]}
    base=far_success(m,pb,rng,p_open)
    rep =far_success(m,HP["full_sweeps"],rng,p_open)                       # +Compute
    m2=copy.deepcopy(m); explore(m2,HP["data_intervention_episodes"],rng,p_open)
    dat=far_success(m2,HP["full_sweeps"],rng,p_open)                       # +Data
    return dict(seed=seed,data_budget=db,plan_budget=pb,Ks=Ks,
                K=Ks[c if c else HP["c_coef"]],base=base,replan=rep,data=dat)

# =========================== MAIN RUN ===========================
def roc(scores,labels):
    order=np.argsort(-scores); s=scores[order]; l=labels[order]
    P=l.sum(); N=len(l)-P; tp=fp=0; tpr=[0]; fpr=[0]
    for i in range(len(l)):
        if l[i]==1: tp+=1
        else: fp+=1
        tpr.append(tp/P if P else 0); fpr.append(fp/N if N else 0)
    return np.array(fpr),np.array(tpr)
def auc(f,t): return float(np.sum((f[1:]-f[:-1])*(t[1:]+t[:-1])/2))

def analyse(agents,c=None,verbose=True,seed=0):
    key=(lambda a: a['Ks'][c]) if c else (lambda a: a['K'])
    fail=[a for a in agents if a['base']<HP["fail_threshold"]]
    K=np.array([key(a) for a in fail])
    y=np.array([1 if a['replan']>=HP["recover_threshold"] else 0 for a in fail])  # replan suffices
    # ---- honest split: choose theta on TRAIN, report on TEST ----
    rs=np.random.default_rng(seed); idx=rs.permutation(len(fail))
    ntr=int(HP["train_frac"]*len(fail)); tr,te=idx[:ntr],idx[ntr:]
    cands=np.linspace(0,1,201)
    def youden(Ks,ys,th):
        pred=(Ks>=th).astype(int)
        tp=np.sum((pred==1)&(ys==1)); tn=np.sum((pred==0)&(ys==0))
        fp=np.sum((pred==1)&(ys==0)); fn=np.sum((pred==0)&(ys==1))
        sens=tp/(tp+fn) if (tp+fn) else 0; spec=tn/(tn+fp) if (tn+fp) else 0
        return sens+spec-1,(tp+tn)/len(pred),sens,spec
    Js=[youden(K[tr],y[tr],t)[0] for t in cands]
    theta=float(cands[int(np.argmax(Js))])
    _,acc_te,sens_te,spec_te=youden(K[te],y[te],theta)
    f_te,t_te=roc(K[te],y[te]); auc_te=auc(f_te,t_te)
    f_all,t_all=roc(K,y); auc_all=auc(f_all,t_all)
    # bootstrap CIs on the TEST split
    rb=np.random.default_rng(1); aucs=[]; accs=[]
    for _ in range(HP["n_bootstrap"]):
        bi=rb.integers(0,len(te),len(te)); ks=K[te][bi]; ys=y[te][bi]
        if ys.sum() in (0,len(ys)): continue
        fb,tb=roc(ks,ys); aucs.append(auc(fb,tb))
        accs.append(float(np.mean((ks>=theta).astype(int)==ys)))
    ci=lambda x:(float(np.percentile(x,2.5)),float(np.percentile(x,97.5)))
    # prescription outcome on TEST
    pred=(K[te]>=theta).astype(int); ft=[fail[i] for i in te]
    pi=[a for p,a in zip(pred,ft) if p==1]; pk=[a for p,a in zip(pred,ft) if p==0]
    rec_i=float(np.mean([a['replan']>=HP["recover_threshold"] for a in pi])) if pi else 0
    rec_k=float(np.mean([a['data']  >=HP["recover_threshold"] for a in pk])) if pk else 0
    overall=float(np.mean([ (a['replan'] if p==1 else a['data'])>=HP["recover_threshold"]
                            for p,a in zip(pred,ft)]))
    out=dict(n_agents=len(agents),n_fail=len(fail),
             frac_replan_suffices=float(y.mean()),
             theta=theta,auc_test=auc_te,auc_all=auc_all,
             acc_test=acc_te,sens_test=sens_te,spec_test=spec_te,
             auc_ci=ci(aucs),acc_ci=ci(accs),
             rec_integration=rec_i,rec_knowledge=rec_k,overall=overall,
             n_train=len(tr),n_test=len(te))
    if verbose:
        print(f"agents={out['n_agents']} failing={out['n_fail']} "
              f"(train {out['n_train']} / test {out['n_test']})")
        print(f"  replan suffices in {out['frac_replan_suffices']*100:.0f}% of failures")
        print(f"  theta*(train) = {theta:.3f}")
        print(f"  TEST: AUC {auc_te:.3f} CI{tuple(round(v,3) for v in out['auc_ci'])} | "
              f"acc {acc_te:.3f} CI{tuple(round(v,3) for v in out['acc_ci'])} | "
              f"sens {sens_te:.2f} spec {spec_te:.2f}")
        print(f"  prescribed fix recovers {overall*100:.0f}% of test failures "
              f"(integration {rec_i*100:.0f}%, knowledge {rec_k*100:.0f}%)")
    return out,K,y,tr,te,theta

if __name__=="__main__":
    print("="*70); print("MAIN RUN (p_open=%.2f, c=%.1f, n=%d)"%(HP['p_open'],HP['c_coef'],HP['n_agents']))
    agents=[make_agent(s) for s in range(HP["n_agents"])]
    main,K,y,tr,te,theta=analyse(agents,seed=HP["master_seed"])

    print("\n"+"="*70); print("ABLATION: uncertainty coefficient c")
    abl_c={}
    for c in [0.5,1.0,1.5,2.0]:
        r,_,_,_,_,_=analyse(agents,c=c,verbose=False)
        abl_c[c]=r; print(f"  c={c}: theta*={r['theta']:.3f}  TEST AUC={r['auc_test']:.3f}  acc={r['acc_test']:.3f}")

    print("\n"+"="*70); print("ABLATION: stochasticity p_open")
    abl_p={}
    for p in [0.5,0.7,0.9,1.0]:
        ag=[make_agent(10000+s,p_open=p) for s in range(160)]
        r,_,_,_,_,_=analyse(ag,verbose=False)
        abl_p[p]=r; print(f"  p_open={p}: failing={r['n_fail']}  theta*={r['theta']:.3f}  "
                          f"TEST AUC={r['auc_test']:.3f}  acc={r['acc_test']:.3f}")

    print("\n"+"="*70); print("ABLATION: population size (stability of estimates)")
    abl_n={}
    for n in [80,160,320]:
        r,_,_,_,_,_=analyse(agents[:n],verbose=False)
        abl_n[n]=r; print(f"  n={n}: TEST AUC={r['auc_test']:.3f}  acc={r['acc_test']:.3f}")

    # save everything for figures / paper
    np.savez("final.npz",
        K=K,y=y,tr=tr,te=te,theta=theta,
        Kall=[a['K'] for a in agents], base=[a['base'] for a in agents],
        replan=[a['replan'] for a in agents], data=[a['data'] for a in agents])
    json.dump(dict(HP={k:(v if not isinstance(v,tuple) else list(v)) for k,v in HP.items()},
                   main=main, abl_c={str(k):v for k,v in abl_c.items()},
                   abl_p={str(k):v for k,v in abl_p.items()},
                   abl_n={str(k):v for k,v in abl_n.items()}),
              open("results.json","w"), indent=2, ensure_ascii=False)
    print("\nsaved final.npz and results.json")

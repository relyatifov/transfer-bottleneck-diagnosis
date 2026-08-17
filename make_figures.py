import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13})
d=np.load("final.npz"); R=json.load(open("results.json"))
K,y,te,theta=d["K"],d["y"],d["te"],float(d["theta"])
m=R["main"]

def roc(s,l):
    o=np.argsort(-s); s,l=s[o],l[o]; P=l.sum(); N=len(l)-P; tp=fp=0; T=[0]; F=[0]
    for i in range(len(l)):
        if l[i]==1: tp+=1
        else: fp+=1
        T.append(tp/P); F.append(fp/N)
    return np.array(F),np.array(T)

# ---- Fig 1: K distribution by true regime (all failing agents) ----
fig,ax=plt.subplots(figsize=(6.4,4.0),dpi=200)
bins=np.linspace(0,0.75,16)
ax.hist(K[y==1],bins=bins,color="0.55",edgecolor="black",label="replanning suffices\n(integration deficit)")
ax.hist(K[y==0],bins=bins,color="white",edgecolor="black",hatch="///",label="data needed\n(knowledge deficit)")
ax.axvline(theta,color="0.1",lw=1.8,ls=":")
ax.text(theta+0.015,ax.get_ylim()[1]*0.82,f"$\\theta$* = {theta:.2f}",fontsize=14)
ax.set_xlabel("Diagnostic criterion $K(X)$",fontsize=14)
ax.set_ylabel("Number of agents",fontsize=14)
ax.tick_params(labelsize=12)
ax.legend(frameon=False,fontsize=11,loc="upper right")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("figures/f1_hist.png",dpi=200); plt.close(fig)

# ---- Fig 2: ROC on held-out test split ----
F,T=roc(K[te],y[te])
fig,ax=plt.subplots(figsize=(4.9,4.6),dpi=200)
ax.plot([0,1],[0,1],ls=":",color="0.6",lw=1.2)
ax.plot(F,T,color="0.1",lw=2.4); ax.fill_between(F,T,0,color="0.88")
ax.set_xlabel("False positive rate",fontsize=14); ax.set_ylabel("True positive rate",fontsize=14)
ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.tick_params(labelsize=12)
ax.text(0.34,0.14,f"AUC = {m['auc_test']:.3f}\n[{m['auc_ci'][0]:.3f}; {m['auc_ci'][1]:.3f}]",fontsize=14)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("figures/f2_roc.png",dpi=200); plt.close(fig)

# ---- Fig 3: threshold sensitivity on test split ----
cands=np.linspace(0,1,201); Kt,yt=K[te],y[te]
def acc_at(t):
    p=(Kt>=t).astype(int); return float(np.mean(p==yt))
def bal_at(t):
    p=(Kt>=t).astype(int)
    tp=np.sum((p==1)&(yt==1)); tn=np.sum((p==0)&(yt==0))
    fp=np.sum((p==1)&(yt==0)); fn=np.sum((p==0)&(yt==1))
    se=tp/(tp+fn) if (tp+fn) else 0; sp=tn/(tn+fp) if (tn+fp) else 0
    return (se+sp)/2
fig,ax=plt.subplots(figsize=(6.0,3.9),dpi=200)
ax.plot(cands,[acc_at(t) for t in cands],color="0.1",lw=2.2,label="accuracy")
ax.plot(cands,[bal_at(t) for t in cands],color="0.55",lw=2.2,ls="--",label="balanced accuracy")
ax.axvline(theta,color="0.1",lw=1.6,ls=":")
ax.text(theta+0.02,0.56,f"$\\theta$* = {theta:.2f}\n(chosen on train)",fontsize=12)
ax.set_xlabel("Threshold $\\theta$",fontsize=14); ax.set_ylabel("Classification quality",fontsize=14)
ax.set_ylim(0.45,1.02); ax.set_xlim(0,1); ax.tick_params(labelsize=12)
ax.legend(frameon=False,fontsize=12,loc="upper right")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("figures/f3_theta.png",dpi=200); plt.close(fig)

# ---- Fig 4: environment schematic ----
from matplotlib.patches import Rectangle, RegularPolygon, Circle
Rr,Cc=5,7; START=(2,0); LEVER=(4,2); GAP=(2,3); GOAL=(1,5)
WALLS={(r,3) for r in range(Rr)}-{GAP}
fig,ax=plt.subplots(figsize=(5.4,3.9),dpi=200)
ax.set_xlim(-0.1,Cc+0.1); ax.set_ylim(-0.1,Rr+0.1); ax.set_aspect("equal"); ax.axis("off"); ax.invert_yaxis()
for r in range(Rr+1): ax.plot([0,Cc],[r,r],color="0.6",lw=1)
for c in range(Cc+1): ax.plot([c,c],[0,Rr],color="0.6",lw=1)
for (wr,wc) in WALLS: ax.add_patch(Rectangle((wc,wr),1,1,facecolor="0.15"))
ax.add_patch(Rectangle((GAP[1],GAP[0]),1,1,facecolor="white",edgecolor="0.15",hatch="xx",lw=1.2))
ax.text(GAP[1]+0.5,GAP[0]-0.16,"bridge (p=0.7)",ha="center",fontsize=10)
ax.add_patch(RegularPolygon((LEVER[1]+0.5,LEVER[0]+0.5),numVertices=3,radius=0.30,facecolor="white",edgecolor="0.1",lw=1.8))
ax.text(LEVER[1]+0.5,LEVER[0]+0.92,"lever $X$",ha="center",fontsize=11)
ax.add_patch(RegularPolygon((GOAL[1]+0.5,GOAL[0]+0.5),numVertices=5,radius=0.32,orientation=np.pi,facecolor="white",edgecolor="0.1",lw=1.8))
ax.text(GOAL[1]+0.5,GOAL[0]-0.16,"goal",ha="center",fontsize=11)
ax.add_patch(Circle((START[1]+0.5,START[0]+0.5),0.28,facecolor="0.1"))
ax.text(START[1]+0.5,START[0]+0.92,"start",ha="center",fontsize=11)
fig.subplots_adjust(left=0.02,right=0.98,top=0.97,bottom=0.03)
fig.savefig("figures/f4_env.png",dpi=200); plt.close(fig)
print("figures saved")

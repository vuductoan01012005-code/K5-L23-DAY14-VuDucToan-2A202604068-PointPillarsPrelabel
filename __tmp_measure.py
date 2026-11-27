import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vf_pp.core import correction_cost, needs_edit, VF_NAMES, within_range
from scripts.vf_common import match_boxes

base = Path(__file__).resolve().parent / "d14_2A202604068"
pred = json.loads((base/"pred_A.json").read_text(encoding="utf-8-sig"))["frames"]
ref = json.loads((base/"ref_A.json").read_text(encoding="utf-8-sig"))["frames"]

def label_name(b):
    v=b.get("label")
    return VF_NAMES.get(v, v) if isinstance(v, int) else v

total={"gt":0,"pred":0,"match":0,"delete":0,"add":0,"edit":0}
unmatched=[]
matched_edit=[]
for fr in ["s03f32","s04f50","s05f32"]:
    g=ref[fr]; p=pred[fr]
    c=correction_cost(g,p)
    print(fr, c)
    for k in total: total[k]+=c[k]
    gt_w=within_range(g); pr_w=within_range(p)
    pairs=match_boxes(gt_w, pr_w, 0.3)
    matched_pred={pj for _,pj,_ in pairs}
    for j,b in enumerate(pr_w):
        if j not in matched_pred:
            unmatched.append((fr, label_name(b), round(float(b.get("score",0)),3), b.get("npts"), round(math.hypot(b["x"],b["y"]),1), b["x"], b["y"]))
    for gi,pj,iou in pairs:
        gg=gt_w[gi]; pp=pr_w[pj]
        if needs_edit(gg,pp):
            dyaw=abs((gg["yaw"]-pp["yaw"]+math.pi/2)%math.pi-math.pi/2)
            reasons=[]
            if gg["label"]!=label_name(pp): reasons.append(f"label {gg['label']}!={label_name(pp)}")
            if math.hypot(gg["x"]-pp["x"],gg["y"]-pp["y"])>0.3: reasons.append(f"pos{math.hypot(gg['x']-pp['x'],gg['y']-pp['y']):.2f}")
            if dyaw>math.radians(10): reasons.append(f"yaw{math.degrees(dyaw):.0f}deg")
            for k in ("length","width","height"):
                if abs(gg[k]-pp[k])>0.2: reasons.append(f"{k}:{gg[k]:.2f}vs{pp[k]:.2f}")
            matched_edit.append((fr, label_name(pp), round(float(pp.get("score",0)),3), pp.get("npts"), round(iou,2), ",".join(reasons)))

cost=round(0.3*total["delete"]+total["add"]+0.5*total["edit"],2)
print("TOTAL",total,"cost",cost)
print(f"unmatched_pred={len(unmatched)} matched_edit={len(matched_edit)}")
# distributions of unmatched
from collections import Counter
def bucket_score(s):
    if s<0.2: return "<0.2"
    if s<0.3: return "0.2-0.3"
    if s<0.4: return "0.3-0.4"
    if s<0.5: return "0.4-0.5"
    return ">=0.5"
print("unmatched by score", Counter(bucket_score(s) for _,_,s,_,_,_,_ in unmatched))
print("unmatched by label", Counter(l for _,l,_,_,_,_,_ in unmatched))
print("unmatched npts<5", sum(1 for *_,n,_,_,_ in [(u[0],u[1],u[2],u[3],u[4],u[5],u[6]) for u in unmatched] if n is not None and n<5),
      "npts<10", sum(1 for u in unmatched if u[3] is not None and u[3]<10),
      "npts<20", sum(1 for u in unmatched if u[3] is not None and u[3]<20),
      "npts>=50", sum(1 for u in unmatched if u[3] is not None and u[3]>=50))
print("unmatched dist>40", sum(1 for u in unmatched if u[4]>40), "dist>30", sum(1 for u in unmatched if u[4]>30), "dist<=30", sum(1 for u in unmatched if u[4]<=30))
print("unmatched list (frame,label,score,npts,dist):")
for u in unmatched: print(" ",u[:5])
print("matched_edit list:")
for u in matched_edit: print(" ",u)

# simulate single-param filtering on pred_A boxes (approx postprocess)
def simulate(name, filt):
    tot={"delete":0,"add":0,"edit":0}
    for fr in ["s03f32","s04f50","s05f32"]:
        p=[b for b in pred[fr] if filt(b)]
        c=correction_cost(ref[fr],p)
        for k in tot: tot[k]+=c[k]
    cst=round(0.3*tot["delete"]+tot["add"]+0.5*tot["edit"],2)
    print(f"SIM {name}: delete={tot['delete']} add={tot['add']} edit={tot['edit']} cost={cst}")
    return cst

for thr in [0.2,0.3,0.4,0.5]:
    simulate(f"score_thr={thr}", lambda b,thr=thr: float(b.get("score",0))>=thr)
for mp in [5,10,20]:
    simulate(f"min_points={mp}", lambda b,mp=mp: int(b.get("npts",0))>=mp)
for r in [40,30]:
    simulate(f"range_m={r}", lambda b,r=r: math.hypot(b["x"],b["y"])<=r)
# counts pred within 50
for fr in ["s03f32","s04f50","s05f32"]:
    print(fr, "pred_all",len(pred[fr]),"pred_le50",len(within_range(pred[fr])),"gt_all",len(ref[fr]),"gt_le50",len(within_range(ref[fr])))

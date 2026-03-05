import torch, torch.nn as nn, numpy as np, glob, os, json
from torchvision import transforms, models
from PIL import Image
from scipy.stats import spearmanr, pearsonr

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
KP = {'OHK':(3,'prime','stopper',1),'F8K':(4,'prime','stopper',1),
      'BK':(4,'loop','loop',1),'RK':(6,'composite','binding',2),
      'FSK':(6,'composite','bend',2),'FMB':(8,'composite','bend',2),
      'F8L':(4,'loop','loop',1),'CH':(2,'hitch','hitch',1),
      'SK':(3,'slip','stopper',1),'ABK':(4,'loop','loop',1)}
DR = {('OHK','SK'):0.1,('F8K','FMB'):0.15,('RK','FSK'):0.1,('F8K','F8L'):0.1}
def td(a,b):
    c1,t1,f1,n1=KP[a];c2,t2,f2,n2=KP[b]
    return 0.25*abs(c1-c2)/8+0.25*(0 if f1==f2 else 1)+0.15*(0 if t1==t2 else .5)+.1*abs(n1-n2)+.25*DR.get(tuple(sorted([a,b])),.5)
tD = np.array([[td(CLASSES[i],CLASSES[j]) if i!=j else 0 for j in range(10)] for i in range(10)])

c2i = {c:i for i,c in enumerate(CLASSES)}
paths, labels = [], []
for p in sorted(glob.glob('train/*.jpg')):
    cn = os.path.basename(p).split('_')[0]
    if cn in c2i: paths.append(p); labels.append(c2i[cn])
labels = np.array(labels)

tf = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def get_feats(model, hook_target):
    ho = {}
    def hk(m,i,o): ho['f']=o.detach()
    h = hook_target.register_forward_hook(hk)
    all_f = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), 32):
            imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[start:start+32]])
            _ = model(imgs)
            f = ho['f'].squeeze()
            if f.ndim == 1: f = f.unsqueeze(0)
            all_f.append(f.numpy())
    h.remove()
    return np.vstack(all_f)

class TopoModel(nn.Module):
    def __init__(self, bb, dim):
        super().__init__()
        self.backbone = bb
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 10))
    def forward(self, x): return self.classifier(self.backbone(x))

configs = [
    ('resnet18_baseline','checkpoints/resnet18_best.pth','resnet18',512,False),
    ('resnet18_topo','checkpoints/resnet18_topo_guided.pth','resnet18',512,True),
    ('resnet50_baseline','checkpoints/resnet50_best.pth','resnet50',2048,False),
    ('resnet50_topo','checkpoints/resnet50_topo_guided.pth','resnet50',2048,True),
]

results = {}
idx = np.triu_indices(10,k=1)
for name, ckpt, arch, dim, is_topo in configs:
    if arch == 'resnet18': base = models.resnet18(weights=None)
    else: base = models.resnet50(weights=None)
    if is_topo:
        base.fc = nn.Identity()
        m = TopoModel(base, dim)
        hook_t = m.backbone.avgpool
    else:
        base.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 10))
        m = base
        hook_t = m.avgpool
    m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False))
    feats = get_feats(m, hook_t)
    centroids = np.array([feats[labels==c].mean(0) for c in range(10)])
    eD = np.array([[np.linalg.norm(centroids[i]-centroids[j]) for j in range(10)] for i in range(10)])
    rs,ps = spearmanr(tD[idx],eD[idx])
    rp,pp = pearsonr(tD[idx],eD[idx])
    results[name] = {'spearman_rho':float(rs),'spearman_p':float(ps),'pearson_r':float(rp),'pearson_p':float(pp)}
    with open('results/embed_progress.txt','a') as f:
        f.write(f'{name}: Spearman={rs:.4f}(p={ps:.4f}) Pearson={rp:.4f}(p={pp:.4f})\n')

with open('results/embedding_alignment.json','w') as f:
    json.dump(results, f, indent=2)
with open('results/embed_done.txt','w') as f:
    f.write('DONE\n')

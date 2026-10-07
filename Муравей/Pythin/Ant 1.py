import itertools, sys
DIRS = [(-1,0),(0,1),(1,0),(0,-1)]            # 0=N,1=E,2=S,3=W
TURN = {'L':3,'R':1,'U':2,'F':0}              # приращение направления mod 4

class Ant:
    def __init__(self, rule):
        self.rule, self.q = rule, len(rule)
        self.field = {}; self.pos, self.d = (0,0), 1
        self.t = 0; self.nvis = 1; self.seen = {(0,0)}; self.bb = [0,0,0,0]
    def step(self):
        i,j = self.pos
        s = self.field.get((i,j), 0)
        self.d = (self.d + TURN[self.rule[s]]) % 4
        s2 = (s+1) % self.q
        if s2: self.field[(i,j)] = s2
        else:  self.field.pop((i,j), None)
        di,dj = DIRS[self.d]; self.pos = (i+di, j+dj)
        if self.pos not in self.seen: self.seen.add(self.pos); self.nvis += 1
        b = self.bb
        b[0]=min(b[0],self.pos[0]); b[1]=max(b[1],self.pos[0])
        b[2]=min(b[2],self.pos[1]); b[3]=max(b[3],self.pos[1])
        self.t += 1
    def window(self, r=3):
        i,j = self.pos; g = self.field.get
        return (self.d, tuple(g((i+a,j+b),0) for a in range(-r,r+1) for b in range(-r,r+1)))
    def stats(self):
        i,j = self.pos
        return dict(t=self.t, dist=abs(i)+abs(j), nvis=self.nvis,
                    bbox=(self.bb[1]-self.bb[0]+1)*(self.bb[3]-self.bb[2]+1),
                    ncol=len(self.field))

def symmetry(a):
    f = a.field
    if not f: return 1.0
    m1 = sum(1 for (i,j),s in f.items() if f.get((i,-j),-1)==s)/len(f)
    m2 = sum(1 for (i,j),s in f.items() if f.get((-i,j),-1)==s)/len(f)
    return max(m1, m2)

def classify(rule, T=20000, wait=4000, sym_thr=0.98, log=None, every=500):
    a = Ant(rule); seen = {}; prev = None; conf = 0
    last_grow, last_area = 0, 1
    fh = open(log,'w') if log else None
    if fh: fh.write('t,dist,nvis,bbox,ncol\n')
    while a.t < T:
        a.step()
        area = (a.bb[1]-a.bb[0]+1)*(a.bb[3]-a.bb[2]+1)
        if area != last_area: last_area, last_grow = area, a.t
        if fh and a.t % every == 0:
            s = a.stats(); fh.write(f"{s['t']},{s['dist']},{s['nvis']},{s['bbox']},{s['ncol']}\n")
        h = a.window()
        if h in seen:
            t0,p0 = seen[h]; tau = a.t-t0
            d = (a.pos[0]-p0[0], a.pos[1]-p0[1])
            conf = conf+1 if prev==(tau,d) else 1; prev = (tau,d)
            if conf >= 3:
                if fh: fh.close()
                return dict(rule=rule, cls=2 if d==(0,0) else 3, tau=tau, d=d, **a.stats())
        else: conf, prev = 0, None
        seen[h] = (a.t, a.pos)
    if fh: fh.close()
    if a.t - last_grow >= wait: return dict(rule=rule, cls=1, **a.stats())
    return dict(rule=rule, cls=5 if symmetry(a)>=sym_thr else 4, **a.stats())

TRACE = [((-1,0),0),((-1,-1),3),((0,-1),2),((0,0),1),((1,0),2),((1,1),1),
         ((0,1),0),((0,0),3),((1,0),2),((1,-1),3),((2,-1),2),((2,0),1)]
def selftest():
    a = Ant('LR')
    for k,(p,d) in enumerate(TRACE,1):
        a.step(); assert (a.pos,a.d)==(p,d), f'step {k}: {(a.pos,a.d)}!={(p,d)}'
    for rule,cls in {'LL':2,'RR':2,'UU':2,'FF':3}.items():
        r = classify(rule, T=500); assert r['cls']==cls, (rule,r)
    print('selftest OK: трасса 12 шагов + 4 леммы')

def enum(q, T=20000):
    from collections import Counter
    cnt = Counter()
    for tup in itertools.product('RLFU', repeat=q):
        r = classify(''.join(tup), T=T); cnt[r['cls']] += 1
        if r['cls'] in (2,3): print(r)
    print('q=',q,'распределение классов:',dict(cnt))

if __name__ == '__main__':
    c = sys.argv[1] if len(sys.argv)>1 else 'selftest'
    if c=='selftest': selftest()
    elif c=='series1': print(classify('LR', T=200000, log='series1.csv'))
    elif c=='enum': enum(int(sys.argv[2]))
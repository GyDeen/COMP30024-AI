# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Alpha-Beta Agent with TD-Leaf Learned Weights

import os, time, random, math
import numpy as np
from referee.game import PlayerColor, Coord, Direction, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from Agents.common.board import BoardState, RED_FLAG, BLUE_FLAG, HEIGHT_MASK
from Agents.common.action import place, move, eat, cascade, is_valid_coord, \
    BOARD_N, UP, DOWN, LEFT, RIGHT, MOVE, EAT, CASCADE
from .features import extract_features, evaluate_linear, NUM_FEATURES, DEFAULT_WEIGHTS

OFFSETS = (UP, DOWN, LEFT, RIGHT)
DIR_MAP = {UP: Direction.Up, DOWN: Direction.Down, LEFT: Direction.Left, RIGHT: Direction.Right}
INV_DIR_MAP = {v: k for k, v in DIR_MAP.items()}
PLACEMENT_TURNS = 8; INF = 999999
TT_EXACT=0; TT_LOWER=1; TT_UPPER=2

_rng = random.Random(42)
ZOBRIST_TABLE = [[_rng.getrandbits(64) for _ in range(64)] for _ in range(64)]
ZOBRIST_TURN = _rng.getrandbits(64)

EDGE_DIST = [min(r,7-r,c,7-c) for r in range(8) for c in range(8)]
MANHATTAN = [[abs((a>>3)-(b>>3))+abs((a&7)-(b&7)) for b in range(64)] for a in range(64)]

def zobrist_hash(board, is_red_turn):
    h=0
    for idx in range(64):
        v=board.state[idx]
        if v: h^=ZOBRIST_TABLE[idx][v&63]
    if not is_red_turn: h^=ZOBRIST_TURN
    return h

def generate_actions(board, is_red):
    my_f=RED_FLAG if is_red else BLUE_FLAG; op_f=BLUE_FLAG if is_red else RED_FLAG
    acts=[]
    for idx in range(64):
        v=board.state[idx]
        if not(v&my_f):continue
        h=v&HEIGHT_MASK
        for off in OFFSETS:
            if not is_valid_coord(idx,off):continue
            tv=board.state[idx+off]
            if tv==0:acts.append((MOVE,idx,off,0))
            elif tv&my_f:acts.append((MOVE,idx,off,1))
            elif tv&op_f:
                th=tv&HEIGHT_MASK
                if h>=th:acts.append((EAT,idx,off,10+th))
        if h>=2:
            for off in OFFSETS:acts.append((CASCADE,idx,off,5))
    acts.sort(key=lambda x:x[3],reverse=True)
    return acts

def generate_placement_actions(board, is_red, pc):
    op_f=BLUE_FLAG if is_red else RED_FLAG; acts=[]
    for idx in range(64):
        if board.state[idx]!=0:continue
        if pc>0:
            adj=False
            for off in OFFSETS:
                if is_valid_coord(idx,off) and(board.state[idx+off]&op_f):adj=True;break
            if adj:continue
        acts.append(idx)
    return acts

def apply_action(board, at, si, off):
    if at==MOVE:return move(board,si,off)
    elif at==EAT:return eat(board,si,off)
    elif at==CASCADE:return cascade(board,si,off)
    return board

def tactical_decision(board, is_red):
    my_f=RED_FLAG if is_red else BLUE_FLAG;op_f=BLUE_FLAG if is_red else RED_FLAG
    state=board.state
    my_b=board.red_tokens if is_red else board.blue_tokens
    op_b=board.blue_tokens if is_red else board.red_tokens
    best_eat=None;beg=0
    for idx in range(64):
        v=state[idx]
        if not(v&my_f):continue
        h=v&HEIGHT_MASK
        for off in OFFSETS:
            if not is_valid_coord(idx,off):continue
            tv=state[idx+off]
            if not(tv&op_f):continue
            th=tv&HEIGHT_MASK
            if h<th:continue
            nb=apply_action(board,EAT,idx,off)
            ol=nb.blue_tokens if is_red else nb.red_tokens
            if ol==0:return(EAT,idx,off)
            dest=idx+off;safe=True
            for no in OFFSETS:
                if not is_valid_coord(dest,no):continue
                nv=nb.state[dest+no]
                if nv and(nv&op_f)and(nv&HEIGHT_MASK)>=h:safe=False;break
            if safe and th>beg:beg=th;best_eat=(EAT,idx,off)
    # Strict cascade check
    one=any(state[i]and(state[i]&op_f)and EDGE_DIST[i]<=2 for i in range(64))
    best_cas=None;bcg=0
    if one:
        for idx in range(64):
            v=state[idx]
            if not(v&my_f):continue
            h=v&HEIGHT_MASK
            if h<2:continue
            for off in OFFSETS:
                nb=apply_action(board,CASCADE,idx,off)
                ol=nb.blue_tokens if is_red else nb.red_tokens
                ml=nb.red_tokens if is_red else nb.blue_tokens
                if ol==0 and ml>0:return(CASCADE,idx,off)
                od=op_b-ol
                if od==0:continue
                mld=my_b-ml;risk=0
                for ci in range(64):
                    cv=nb.state[ci]
                    if not cv or not(cv&my_f):continue
                    ch=cv&HEIGHT_MASK
                    for oi in range(64):
                        ov=nb.state[oi]
                        if not ov or not(ov&op_f):continue
                        if(ov&HEIGHT_MASK)>=ch and MANHATTAN[ci][oi]<=2:risk+=ch;break
                adj=od-mld-risk
                if adj>=3 and adj>bcg:bcg=adj;best_cas=(CASCADE,idx,off)
    if best_eat:return best_eat
    return best_cas

def filter_cascades(board, is_red, actions):
    my_f=RED_FLAG if is_red else BLUE_FLAG;op_f=BLUE_FLAG if is_red else RED_FLAG
    mb=board.red_tokens if is_red else board.blue_tokens
    ob=board.blue_tokens if is_red else board.red_tokens
    filt=[]
    for act in actions:
        if act[0]!=CASCADE:filt.append(act);continue
        nb=apply_action(board,act[0],act[1],act[2])
        ma=nb.red_tokens if is_red else nb.blue_tokens
        oa=nb.blue_tokens if is_red else nb.red_tokens
        net=(ob-oa)-(mb-ma)
        if net<=0:continue
        if net<3:
            bad=False
            for ci in range(64):
                cv=nb.state[ci]
                if cv and(cv&my_f)and(cv&HEIGHT_MASK)==1:
                    for oi in range(64):
                        ov=nb.state[oi]
                        if ov and(ov&op_f)and MANHATTAN[ci][oi]<=3:bad=True;break
                    if bad:break
            if bad:continue
        filt.append(act)
    return filt if filt else actions

class AlphaBetaSearcher:
    def __init__(self, weights):
        self.weights=weights;self.tt={};self.nodes=0;self.tl=0;self.t0=0;self.to=False
    def ev(self,b,rm):return evaluate_linear(b,rm,self.weights)
    def tu(self):
        if self.nodes%1024==0 and time.process_time()-self.t0>self.tl:self.to=True
        return self.to
    def ab(self,b,d,a,bt,mx,rt,rm):
        self.nodes+=1
        if self.tu():return 0.0
        if b.red_tokens==0 or b.blue_tokens==0:return self.ev(b,rm)
        if d<=0:return self.ev(b,rm)
        bh=zobrist_hash(b,rt);te=self.tt.get(bh)
        if te:
            td,tf,tv,tb=te
            if td>=d:
                if tf==TT_EXACT:return tv
                elif tf==TT_LOWER and tv>=bt:return tv
                elif tf==TT_UPPER and tv<=a:return tv
        acts=generate_actions(b,rt)
        if not acts:return 0.0
        if te and te[3]:
            tb=te[3]
            for i,ac in enumerate(acts):
                if ac[0]==tb[0]and ac[1]==tb[1]and ac[2]==tb[2]:acts[0],acts[i]=acts[i],acts[0];break
        ba=acts[0];oa=a
        if mx:
            me=-INF
            for ac in acts:
                nb=apply_action(b,ac[0],ac[1],ac[2])
                v=self.ab(nb,d-1,a,bt,False,not rt,rm)
                if self.to:return me if me>-INF else 0.0
                if v>me:me=v;ba=ac
                a=max(a,v)
                if a>=bt:break
            if not self.to:
                f=TT_UPPER if me<=oa else(TT_LOWER if me>=bt else TT_EXACT)
                self.tt[bh]=(d,f,me,ba)
            return me
        else:
            me=INF
            for ac in acts:
                nb=apply_action(b,ac[0],ac[1],ac[2])
                v=self.ab(nb,d-1,a,bt,True,not rt,rm)
                if self.to:return me if me<INF else 0.0
                if v<me:me=v;ba=ac
                bt=min(bt,v)
                if a>=bt:break
            if not self.to:
                f=TT_LOWER if me>=bt else(TT_UPPER if me<=oa else TT_EXACT)
                self.tt[bh]=(d,f,me,ba)
            return me
    def search(self,board,is_red,is_red_max,budget):
        self.t0=time.process_time();self.tl=budget;self.to=False
        if len(self.tt)>500000:self.tt.clear()
        acts=generate_actions(board,is_red)
        if not acts:return None,0.0,0
        if len(acts)==1:return acts[0],0.0,0
        acts=filter_cascades(board,is_red,acts)
        best=acts[0];bv=-INF
        for depth in range(1,30):
            self.nodes=0;self.to=False;v=-INF;cur=acts[0]
            for ac in acts:
                nb=apply_action(board,ac[0],ac[1],ac[2])
                s=self.ab(nb,depth-1,v,INF,False,not is_red,is_red_max)
                if self.to:break
                if s>v:v=s;cur=ac
            if not self.to:
                best=cur;bv=v
                if bv>=INF-100:break
                for i,a in enumerate(acts):
                    if a[0]==best[0]and a[1]==best[1]and a[2]==best[2]:acts[0],acts[i]=acts[i],acts[0];break
            else:break
            if time.process_time()-self.t0>budget*0.6:break
        return best,bv,depth

def choose_placement(board,is_red,pc):
    legal=generate_placement_actions(board,is_red,pc)
    if not legal:raise RuntimeError("No legal placements")
    my_f=RED_FLAG if is_red else BLUE_FLAG
    my_pos=[i for i in range(64) if board.state[i]and(board.state[i]&my_f)]
    bi=legal[0];bs=-999
    for idx in legal:
        s=EDGE_DIST[idx]*3.0
        for mi in my_pos:
            d=abs((idx>>3)-(mi>>3))+abs((idx&7)-(mi&7))
            if d==2 or d==3:s+=2.0
            elif d==1:s+=0.5
            elif d==4:s+=1.0
        r,c=idx>>3,idx&7;s+=max(0,4-abs(r-3.5)-abs(c-3.5))*0.5
        if s>bs:bs=s;bi=idx
    return bi

def idx_to_coord(idx):return Coord(idx//8,idx%8)
def coord_to_idx(coord):return coord.r*8+coord.c
def action_tuple_to_action(at,si,off):
    co=idx_to_coord(si);di=DIR_MAP[off]
    if at==MOVE:return MoveAction(co,di)
    elif at==EAT:return EatAction(co,di)
    elif at==CASCADE:return CascadeAction(co,di)

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color=color;self._is_red=(color==PlayerColor.RED)
        self._board=BoardState();self._pc=0;self._tc=0
        wp=os.path.join(os.path.dirname(__file__),'learned_weights.npy')
        self._w=np.load(wp) if os.path.exists(wp) else DEFAULT_WEIGHTS.copy()
        self._searcher=AlphaBetaSearcher(self._w)
        self._hist=[];self._time=0.0

    def action(self, **referee: dict) -> Action:
        t0=time.process_time();tr=referee.get("time_remaining",None)
        if self._pc<PLACEMENT_TURNS:
            idx=choose_placement(self._board,self._is_red,self._pc)
            return PlaceAction(idx_to_coord(idx))
        forced=tactical_decision(self._board,self._is_red)
        if forced:
            self._time+=time.process_time()-t0
            return action_tuple_to_action(forced[0],forced[1],forced[2])
        rem=tr if tr else 180.0-self._time
        if rem<5:bg=0.2
        elif rem<15:bg=0.5
        elif rem<40:bg=1.0
        elif rem<80:bg=1.5
        else:bg=2.0
        best,val,depth=self._searcher.search(self._board,self._is_red,self._is_red,max(0.1,bg))
        if best is None:
            acts=generate_actions(self._board,self._is_red)
            if acts:best=acts[0]
            else:raise RuntimeError("No legal actions")
        tb=apply_action(self._board,best[0],best[1],best[2])
        th=bytes(tb.state)
        if self._hist.count(th)>=2:
            for alt in generate_actions(self._board,self._is_red):
                if alt[0]==best[0]and alt[1]==best[1]and alt[2]==best[2]:continue
                ab=apply_action(self._board,alt[0],alt[1],alt[2])
                if self._hist.count(bytes(ab.state))<2:best=alt;break
        self._time+=time.process_time()-t0
        return action_tuple_to_action(best[0],best[1],best[2])

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        ir=(color==PlayerColor.RED)
        match action:
            case PlaceAction(coord):self._board=place(self._board,coord_to_idx(coord),ir);self._pc+=1
            case MoveAction(coord,direction):self._board=apply_action(self._board,MOVE,coord_to_idx(coord),INV_DIR_MAP[direction])
            case EatAction(coord,direction):self._board=apply_action(self._board,EAT,coord_to_idx(coord),INV_DIR_MAP[direction])
            case CascadeAction(coord,direction):self._board=apply_action(self._board,CASCADE,coord_to_idx(coord),INV_DIR_MAP[direction])
        if color==self._color:self._tc+=1
        if self._pc>=PLACEMENT_TURNS:self._hist.append(bytes(self._board.state))


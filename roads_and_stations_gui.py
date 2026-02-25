"""
roads_and_stations_gui.py
Tkinter GUI wrapper for the road/station route builder.
All user interaction through the window; computation runs in a background thread.
"""

import json, math, os, re, glob, time, random, warnings, threading, queue
from datetime import datetime
from collections import defaultdict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.figure as mplfig
import matplotlib.cm as mpl_cm
import matplotlib.lines as mlines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from shapely.ops import unary_union
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

# ================================================================
# CONSTANTS
# ================================================================
SNAP_PRECISION     = 5
BRIDGE_RADIUS_M    = 500
TJUNCTION_RADIUS_M = 200
BUFFER_M           = 50
MEAN_LAT           = 61.0
COS_LAT            = math.cos(math.radians(MEAN_LAT))
DEG_TO_M           = 111_000
COORD_PRECISION    = 7
N_ROUNDS           = 3
STATION_SNAP_M     = 500
N_SELECT           = 10
ROUTE_OFFSET_M     = 40

ROUTE_COLORS = ['#E53935','#43A047','#FB8C00','#8E24AA','#00ACC1',
                '#F4511E','#7CB342','#D81B60','#FFB300','#5E35B1']

BG_STYLES = {
    'water':     {'line_color':'#2196F3','fill_color':'#BBDEFB','edge_color':'#1976D2','lw':0.6,'alpha':0.6,'fill_alpha':0.35},
    'coastline': {'line_color':'#2196F3','fill_color':'#BBDEFB','edge_color':'#1976D2','lw':0.8,'alpha':0.7,'fill_alpha':0.1},
    'railway':   {'line_color':'#000000','fill_color':'#000000','edge_color':'#000000','lw':1.2,'alpha':0.8,'fill_alpha':0.3},
    'other':     {'line_color':'#aaaaaa','fill_color':'#e0e0e0','edge_color':'#bbbbbb','lw':0.4,'alpha':0.5,'fill_alpha':0.4},
}

WATER_KEYS = ('water','river','lake','sea','ocean','bay','strait','reservoir','pond',
              'вод','река','озер','море','залив','пруд','водохр','ручей','канал')
COAST_KEYS = ('coastline','coast','берег','побереж')
RAIL_KEYS  = ('railway','rail','жд','жел_дор','железн')

# ================================================================
# COMPUTATION FUNCTIONS
# ================================================================
_node_counter = 0

def snap_coord(coord, prec=SNAP_PRECISION):
    return (round(coord[0], prec), round(coord[1], prec))

def make_nid(sc, prec=SNAP_PRECISION):
    return f"n_{sc[0]:.{prec}f}_{sc[1]:.{prec}f}"

def deg_to_m(lon, lat):
    return lon * DEG_TO_M * COS_LAT, lat * DEG_TO_M

def edge_length_m(c1, c2):
    dx = (c1[0] - c2[0]) * DEG_TO_M * COS_LAT
    dy = (c1[1] - c2[1]) * DEG_TO_M
    return math.hypot(dx, dy)

def linestring_length_m(coords):
    return sum(edge_length_m(coords[i], coords[i+1]) for i in range(len(coords)-1))

def safe_filename(s):
    s = re.sub(r'[\\/*?:"<>|]', '_', s)
    s = re.sub(r'\s+', '_', s)
    return s.strip('._') or 'route'

def offset_segment(p1, p2, offset_m):
    dx_m = (p2[0]-p1[0]) * DEG_TO_M * COS_LAT
    dy_m = (p2[1]-p1[1]) * DEG_TO_M
    len_m = math.hypot(dx_m, dy_m)
    if len_m < 1e-9: return p1, p2
    off_lon = offset_m * (-dy_m/len_m) / (DEG_TO_M * COS_LAT)
    off_lat = offset_m * (dx_m/len_m) / DEG_TO_M
    return (p1[0]+off_lon, p1[1]+off_lat), (p2[0]+off_lon, p2[1]+off_lat)

class UF:
    def __init__(self, n): self.p = list(range(n)); self.r = [0]*n
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.r[px] < self.r[py]: px, py = py, px
        self.p[py] = px
        if self.r[px] == self.r[py]: self.r[px] += 1
        return True

def build_comp_map(G):
    comps = list(nx.connected_components(G))
    return {n: ci for ci, c in enumerate(comps) for n in c}, comps

def graph_coords_m(G):
    nodes = list(G.nodes())
    arr = np.array([deg_to_m(G.nodes[n]['x'], G.nodes[n]['y']) for n in nodes])
    return nodes, arr

def route_elevation_stats(path, G):
    gain = loss = 0.0; has_data = False
    for k in range(len(path)-1):
        elevs = G[path[k]][path[k+1]].get('elevations')
        if not elevs: continue
        valid = [e for e in elevs if e is not None]
        if len(valid) < 2: continue
        has_data = True
        for j in range(len(valid)-1):
            diff = valid[j+1] - valid[j]
            if diff > 0: gain += diff
            else: loss += abs(diff)
    return (round(gain), round(loss)) if has_data else (None, None)

def load_geojson_as_edges(filepath):
    global _node_counter
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    G_temp = nx.Graph()
    src = os.path.basename(filepath)
    if data.get("type") != "FeatureCollection" or "features" not in data:
        return G_temp
    for feature in data["features"]:
        geom = feature.get("geometry", {}); props = feature.get("properties", {})
        def process_ls(coords):
            global _node_counter
            if len(coords) < 2: return
            c2d = [(round(c[0], COORD_PRECISION), round(c[1], COORD_PRECISION)) for c in coords]
            elevs = [c[2] if len(c) > 2 else None for c in coords]
            if c2d[0] == c2d[-1]: return
            sn = f"nd_{_node_counter}"; _node_counter += 1
            en = f"nd_{_node_counter}"; _node_counter += 1
            G_temp.add_node(sn, x=c2d[0][0], y=c2d[0][1])
            G_temp.add_node(en, x=c2d[-1][0], y=c2d[-1][1])
            ea = dict(length_m=round(linestring_length_m(c2d), 2),
                      road_type=props.get("highway", props.get("type", "")),
                      source_file=src, geometry=c2d, original_edge=True)
            if any(e is not None for e in elevs): ea['elevations'] = elevs
            G_temp.add_edge(sn, en, **ea)
        gt = geom.get("type")
        if gt == "LineString": process_ls(geom["coordinates"])
        elif gt == "MultiLineString":
            for lc in geom["coordinates"]: process_ls(lc)
    return G_temp

def snap_graph(G):
    snap_map = defaultdict(list)
    for node in G.nodes():
        snap_map[snap_coord((G.nodes[node]['x'], G.nodes[node]['y']))].append(node)
    newG = nx.Graph(); old_to_new = {}
    for snapped, nodes in snap_map.items():
        nn = make_nid(snapped); newG.add_node(nn, x=snapped[0], y=snapped[1])
        for old in nodes: old_to_new[old] = nn
    for u, v, d in G.edges(data=True):
        nu, nv = old_to_new[u], old_to_new[v]
        if nu == nv: continue
        if not newG.has_edge(nu, nv):
            if 'geometry' in d:
                geom = d['geometry'].copy()
                geom[0] = (newG.nodes[nu]['x'], newG.nodes[nu]['y'])
                geom[-1] = (newG.nodes[nv]['x'], newG.nodes[nv]['y'])
                d = dict(d); d['geometry'] = geom
            newG.add_edge(nu, nv, **d)
    return newG

def do_bridge_stitching(G, label=''):
    while True:
        nl, coords_m = graph_coords_m(G)
        n2i = {n: i for i, n in enumerate(nl)}
        tree = cKDTree(coords_m); pairs = tree.query_pairs(BRIDGE_RADIUS_M)
        uf = UF(len(nl))
        for u, v in G.edges(): uf.union(n2i[u], n2i[v])
        cross = sorted([(np.linalg.norm(coords_m[i]-coords_m[j]), i, j)
                        for i, j in pairs if uf.find(i) != uf.find(j)])
        added = 0
        for d_val, i, j in cross:
            if uf.find(i) != uf.find(j):
                uf.union(i, j)
                G.add_edge(nl[i], nl[j], length_m=float(d_val),
                           road_type='bridge', source_file=f'auto_bridge{label}', original_edge=False)
                added += 1
        if added == 0: break

def do_edge_intersections(G):
    edges_info = []; edge_lines = []
    for u, v, d in G.edges(data=True):
        coords = d['geometry'] if ('geometry' in d and d['geometry']) else \
                 [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]
        edges_info.append((u, v, dict(d), coords)); edge_lines.append(LineString(coords))
    strtree_ei = STRtree(edge_lines); split_pts = defaultdict(list)
    eps = 10 ** -(SNAP_PRECISION+1)
    for i, (u1, v1, d1, c1) in enumerate(edges_info):
        for j in strtree_ei.query(edge_lines[i]):
            if j <= i: continue
            u2, v2, d2, c2 = edges_info[j]
            if u1 in (u2,v2) or v1 in (u2,v2): continue
            if not edge_lines[i].intersects(edge_lines[j]): continue
            pt = edge_lines[i].intersection(edge_lines[j])
            if pt.geom_type != 'Point': continue
            if any(abs(pt.x-px)<eps and abs(pt.y-py)<eps for px,py in [c1[0],c1[-1],c2[0],c2[-1]]): continue
            sc = snap_coord((pt.x, pt.y)); split_pts[i].append(sc); split_pts[j].append(sc)
    to_remove = set(); to_add = []
    for ei, pts in split_pts.items():
        u, v, d, coords = edges_info[ei]; to_remove.add((u,v))
        all_pts = [coords[0]] + pts + [coords[-1]]; sx, sy = coords[0]
        all_pts.sort(key=lambda c: (c[0]-sx)**2+(c[1]-sy)**2)
        uniq = [all_pts[0]]
        for p in all_pts[1:]:
            if p != uniq[-1]: uniq.append(p)
        for k in range(len(uniq)-1):
            to_add.append((uniq[k], uniq[k+1], d.get('road_type',''), d.get('source_file','')))
    for u, v in to_remove:
        if G.has_edge(u, v): G.remove_edge(u, v)
    for sc, ec, rt, sf in to_add:
        if sc == ec: continue
        sn, en = make_nid(sc), make_nid(ec)
        if sn not in G: G.add_node(sn, x=sc[0], y=sc[1])
        if en not in G: G.add_node(en, x=ec[0], y=ec[1])
        G.add_edge(sn, en, length_m=edge_length_m(sc,ec), road_type=rt, source_file=sf, original_edge=False)
    G.remove_nodes_from(list(nx.isolates(G)))

def do_tjunction(G):
    comp_map, _ = build_comp_map(G)
    dangling = [n for n in G.nodes() if G.degree(n) == 1]
    tj_edges = []; tj_geoms = []
    for u, v, d in G.edges(data=True):
        coords = d['geometry'] if ('geometry' in d and d['geometry']) else \
                 [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]
        tj_edges.append((u,v,d)); tj_geoms.append(LineString(coords))
    tj_tree = STRtree(tj_geoms); tj_rad_deg = TJUNCTION_RADIUS_M/DEG_TO_M
    edge_projs = defaultdict(list)
    for n in dangling:
        nd = G.nodes[n]; pt = Point(nd['x'], nd['y']); cn = comp_map[n]
        buf = pt.buffer(tj_rad_deg); best_dist = TJUNCTION_RADIUS_M; best_idx = best_proj = None
        for idx in tj_tree.query(buf):
            eu, ev, ed = tj_edges[idx]
            if comp_map.get(eu) == cn: continue
            proj = tj_geoms[idx].interpolate(tj_geoms[idx].project(pt))
            dm = edge_length_m((pt.x,pt.y),(proj.x,proj.y))
            if dm < best_dist: best_dist=dm; best_idx=idx; best_proj=snap_coord((proj.x,proj.y))
        if best_idx is not None: edge_projs[best_idx].append((best_proj, n, best_dist))
    for ei, projs in edge_projs.items():
        u, v, d = tj_edges[ei]
        if not G.has_edge(u, v): continue
        rt = d.get('road_type',''); sf = d.get('source_file','')
        coords = d['geometry'] if ('geometry' in d and d['geometry']) else \
                 [(G.nodes[u]['x'],G.nodes[u]['y']),(G.nodes[v]['x'],G.nodes[v]['y'])]
        start, end = snap_coord(coords[0]), snap_coord(coords[-1])
        all_p = [start]; pm = defaultdict(list)
        for proj_c, dn, dm in projs:
            if proj_c not in (start, end): all_p.append(proj_c)
            pm[proj_c].append((dn, dm))
        all_p.append(end); sx, sy = start
        all_p.sort(key=lambda c: (c[0]-sx)**2+(c[1]-sy)**2)
        uniq = [all_p[0]]
        for p in all_p[1:]:
            if p != uniq[-1]: uniq.append(p)
        G.remove_edge(u, v)
        for k in range(len(uniq)-1):
            sc, ec = uniq[k], uniq[k+1]
            if sc == ec: continue
            sn, en = make_nid(sc), make_nid(ec)
            if sn not in G: G.add_node(sn, x=sc[0], y=sc[1])
            if en not in G: G.add_node(en, x=ec[0], y=ec[1])
            G.add_edge(sn, en, length_m=edge_length_m(sc,ec), road_type=rt, source_file=sf, original_edge=False)
        for proj_c, dl in pm.items():
            pid = make_nid(proj_c)
            if pid not in G: G.add_node(pid, x=proj_c[0], y=proj_c[1])
            for dn, dm in dl:
                G.add_edge(dn, pid, length_m=dm, road_type='t_junction', source_file='auto_tj', original_edge=False)
    G.remove_nodes_from(list(nx.isolates(G)))

def do_buffer_merge(G):
    comp_map2, comps2 = build_comp_map(G)
    buf_deg = BUFFER_M/DEG_TO_M; comp_edges_geom = defaultdict(list)
    for u, v, d in G.edges(data=True):
        ci = comp_map2[u]
        line = LineString(d['geometry']) if ('geometry' in d and d['geometry']) else \
               LineString([(G.nodes[u]['x'],G.nodes[u]['y']),(G.nodes[v]['x'],G.nodes[v]['y'])])
        comp_edges_geom[ci].append(line)
    big_comps = [(ci, gs) for ci, gs in comp_edges_geom.items() if len(gs) >= 3]
    comp_buffers = [unary_union(gs).buffer(buf_deg) for ci, gs in big_comps]
    comp_ids_buf = [ci for ci, _ in big_comps]
    buf_tree = STRtree(comp_buffers); buf_pairs = set()
    for i, geom in enumerate(comp_buffers):
        for j in buf_tree.query(geom):
            if j <= i: continue
            ci, cj = comp_ids_buf[i], comp_ids_buf[j]
            if ci != cj and comp_buffers[i].intersects(comp_buffers[j]): buf_pairs.add((i,j))
    for i, j in buf_pairs:
        ci, cj = comp_ids_buf[i], comp_ids_buf[j]
        ni_list = list(comps2[ci]); nj_list = list(comps2[cj])
        if not ni_list or not nj_list: continue
        ci_arr = np.array([deg_to_m(G.nodes[n]['x'],G.nodes[n]['y']) for n in ni_list])
        cj_arr = np.array([deg_to_m(G.nodes[n]['x'],G.nodes[n]['y']) for n in nj_list])
        ti = cKDTree(ci_arr); dists, idxs = ti.query(cj_arr, k=1)
        best = np.argmin(dists); bi = idxs[best]; dm = dists[best]
        if dm > BRIDGE_RADIUS_M: continue
        ni, nj = ni_list[bi], nj_list[best]
        if not G.has_edge(ni, nj):
            G.add_edge(ni, nj, length_m=float(dm), road_type='buffer_merge',
                       source_file='auto_buffer', original_edge=False)
    G.remove_nodes_from(list(nx.isolates(G)))

def classify_feature(feat, filename):
    props = feat.get('properties', {})
    vals = ' '.join(str(v).lower() for v in props.values() if v)
    keys_present = set(k.lower() for k in props.keys()); fn = filename.lower()
    if 'railway' in keys_present or any(k in fn for k in RAIL_KEYS) or any(k in vals for k in ('rail','railway')):
        return 'railway'
    nat = str(props.get('natural','')).lower()
    if nat == 'coastline' or any(k in fn for k in COAST_KEYS) or any(k in vals for k in COAST_KEYS):
        return 'coastline'
    if ('waterway' in keys_present or 'water' in keys_present or nat in ('water','bay','strait','wetland') or
            any(k in fn for k in WATER_KEYS) or any(k in vals for k in WATER_KEYS)):
        return 'water'
    return 'other'

def draw_bg(ax, features, cat):
    s = BG_STYLES[cat]
    for feat in features:
        geom = feat.get('geometry',{}); gt = geom.get('type',''); crd = geom.get('coordinates',[])
        if gt == 'LineString':
            ax.plot([p[0] for p in crd],[p[1] for p in crd],
                    color=s['line_color'],linewidth=s['lw'],alpha=s['alpha'],zorder=1)
        elif gt == 'MultiLineString':
            for line in crd:
                ax.plot([p[0] for p in line],[p[1] for p in line],
                        color=s['line_color'],linewidth=s['lw'],alpha=s['alpha'],zorder=1)
        elif gt == 'Polygon':
            ax.add_patch(MplPolygon([(p[0],p[1]) for p in crd[0]], closed=True,
                                    fc=s['fill_color'],ec=s['edge_color'],
                                    linewidth=s['lw'],alpha=s['fill_alpha'],zorder=1))
        elif gt == 'MultiPolygon':
            for polygon in crd:
                ax.add_patch(MplPolygon([(p[0],p[1]) for p in polygon[0]], closed=True,
                                        fc=s['fill_color'],ec=s['edge_color'],
                                        linewidth=s['lw'],alpha=s['fill_alpha'],zorder=1))
        elif gt == 'Point':
            ax.plot(crd[0],crd[1],'.',color=s['line_color'],markersize=2,alpha=s['alpha'],zorder=1)

# ================================================================
# GUI APPLICATION
# ================================================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Roads & Stations — Route Builder")
        self.root.geometry("1000x720")
        self.root.minsize(800, 600)

        self.q = queue.Queue()
        self.worker = None
        self._stop_flag = threading.Event()

        self._build_ui()
        self._poll()

    # ----------------------------------------------------------
    # UI LAYOUT
    # ----------------------------------------------------------
    def _build_ui(self):
        # ── top: settings ──────────────────────────────────────
        settings = ttk.LabelFrame(self.root, text=" Settings ", padding=8)
        settings.pack(fill=tk.X, padx=10, pady=(10, 4))

        def path_row(parent, label, var, is_file=False, filetypes=None, row=0):
            ttk.Label(parent, text=label, width=18, anchor='e').grid(
                row=row, column=0, sticky='e', padx=(0,4), pady=3)
            ttk.Entry(parent, textvariable=var, width=52).grid(
                row=row, column=1, sticky='ew', pady=3)
            cmd = (lambda v=var, ft=filetypes: v.set(
                filedialog.askopenfilename(filetypes=ft) or v.get())
            ) if is_file else (
                lambda v=var: v.set(filedialog.askdirectory() or v.get()))
            ttk.Button(parent, text="Browse…", width=8, command=cmd).grid(
                row=row, column=2, padx=(4,0), pady=3)

        self.roads_var    = tk.StringVar()
        self.stations_var = tk.StringVar()
        self.bg_var       = tk.StringVar()
        self.save_var     = tk.StringVar()

        path_row(settings, "Roads folder:",   self.roads_var,    row=0)
        path_row(settings, "Stations file:",  self.stations_var, row=1, is_file=True,
                 filetypes=[("GeoJSON", "*.geojson *.json"), ("All", "*.*")])
        path_row(settings, "Background dir:", self.bg_var,       row=2)
        path_row(settings, "Save folder:",    self.save_var,     row=3)
        settings.columnconfigure(1, weight=1)

        # ── bbox ───────────────────────────────────────────────
        bbox_frame = ttk.LabelFrame(self.root, text=" Bounding box (optional) ", padding=8)
        bbox_frame.pack(fill=tk.X, padx=10, pady=4)

        self.bbox_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bbox_frame, text="Restrict by coordinates",
                        variable=self.bbox_var, command=self._toggle_bbox).grid(
            row=0, column=0, columnspan=8, sticky='w')

        labels = ["North (lat):", "South (lat):", "West (lon):", "East (lon):"]
        self.bbox_entries = []
        for col, lbl in enumerate(labels):
            ttk.Label(bbox_frame, text=lbl).grid(row=1, column=col*2, sticky='e', padx=(8,2))
            var = tk.StringVar()
            e = ttk.Entry(bbox_frame, textvariable=var, width=9, state='disabled')
            e.grid(row=1, column=col*2+1, sticky='w', padx=(0,4))
            self.bbox_entries.append((var, e))

        # ── route length ───────────────────────────────────────
        route_frame = ttk.LabelFrame(self.root, text=" Route length ", padding=8)
        route_frame.pack(fill=tk.X, padx=10, pady=4)

        self.min_km = tk.StringVar(value="50")
        self.max_km = tk.StringVar(value="150")
        for col, (lbl, var) in enumerate([("Min (km):", self.min_km), ("Max (km):", self.max_km)]):
            ttk.Label(route_frame, text=lbl).grid(row=0, column=col*2, sticky='e', padx=(0,4))
            ttk.Entry(route_frame, textvariable=var, width=8).grid(
                row=0, column=col*2+1, sticky='w', padx=(0,20))

        # ── buttons ────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=4)

        self.run_btn = ttk.Button(btn_frame, text="▶  Run", width=12, command=self._run)
        self.run_btn.pack(side=tk.LEFT, padx=(0,6))

        self.stop_btn = ttk.Button(btn_frame, text="■  Stop", width=10,
                                   command=self._stop, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=(0,6))

        ttk.Button(btn_frame, text="Clear log", width=10,
                   command=self._clear_log).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(btn_frame, textvariable=self.status_var,
                  foreground='#555').pack(side=tk.LEFT, padx=12)

        # ── progress ───────────────────────────────────────────
        self.progress = ttk.Progressbar(self.root, mode='determinate', maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=(0,4))

        # ── log ────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text=" Log ", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, state='disabled', wrap='word',
            font=('Consolas', 9), background='#1e1e1e', foreground='#d4d4d4',
            insertbackground='white', relief='flat')
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # colour tags
        self.log_text.tag_config('ok',   foreground='#4ec9b0')
        self.log_text.tag_config('warn', foreground='#ce9178')
        self.log_text.tag_config('head', foreground='#569cd6', font=('Consolas',9,'bold'))
        self.log_text.tag_config('sep',  foreground='#444')

    def _toggle_bbox(self):
        state = 'normal' if self.bbox_var.get() else 'disabled'
        for _, e in self.bbox_entries:
            e.config(state=state)

    # ----------------------------------------------------------
    # QUEUE POLLING
    # ----------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self._dispatch(msg)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _dispatch(self, msg):
        t = msg['type']
        if t == 'log':
            self._log(msg['text'], msg.get('tag'))
        elif t == 'progress':
            self.progress['value'] = msg['pct']
            self.status_var.set(msg.get('label', ''))
        elif t == 'show_fig':
            self._embed_figure(msg['fig'], msg['title'], msg['event'])
        elif t == 'ask':
            result = messagebox.askyesno(msg['title'], msg['question'])
            msg['result'].append(result)
            msg['event'].set()
        elif t == 'done':
            self._on_done(success=True)
        elif t == 'stopped':
            self._on_done(success=False, msg="Stopped.")
        elif t == 'error':
            self._log("ERROR: " + msg['text'], 'warn')
            self._on_done(success=False, msg="Error — see log.")

    # ----------------------------------------------------------
    # FIGURE WINDOW
    # ----------------------------------------------------------
    def _embed_figure(self, fig, title, event):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("960x720")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def on_continue():
            event.set()
            win.destroy()

        btn = ttk.Button(win, text="Continue  ▶", command=on_continue)
        btn.pack(pady=6)
        win.protocol("WM_DELETE_WINDOW", on_continue)

    # ----------------------------------------------------------
    # LOG
    # ----------------------------------------------------------
    def _log(self, text, tag=None):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, text + '\n', tag or '')
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def _clear_log(self):
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')

    # ----------------------------------------------------------
    # RUN / STOP
    # ----------------------------------------------------------
    def _run(self):
        params = self._collect_params()
        if params is None:
            return
        self._stop_flag.clear()
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress['value'] = 0
        self.status_var.set("Running…")
        self.worker = threading.Thread(target=self._worker, args=(params,), daemon=True)
        self.worker.start()

    def _stop(self):
        self._stop_flag.set()
        self.status_var.set("Stopping…")

    def _on_done(self, success=True, msg=None):
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress['value'] = 0
        self.status_var.set(msg or ("Done." if success else ""))

    def _collect_params(self):
        roads_dir = self.roads_var.get().strip().strip('"').strip("'")
        stations_path = self.stations_var.get().strip().strip('"').strip("'")
        bg_dir = self.bg_var.get().strip().strip('"').strip("'")
        save_dir = self.save_var.get().strip().strip('"').strip("'")

        if not os.path.isdir(roads_dir):
            messagebox.showerror("Error", f"Roads folder not found:\n{roads_dir}"); return None
        if not os.path.isfile(stations_path):
            messagebox.showerror("Error", f"Stations file not found:\n{stations_path}"); return None
        if save_dir and not os.path.isdir(save_dir):
            messagebox.showerror("Error", f"Save folder not found:\n{save_dir}"); return None

        try:
            min_km = float(self.min_km.get())
            max_km = float(self.max_km.get())
        except ValueError:
            messagebox.showerror("Error", "Min/Max km must be numbers."); return None

        bbox = None
        if self.bbox_var.get():
            try:
                bbox = {
                    'N': float(self.bbox_entries[0][0].get()),
                    'S': float(self.bbox_entries[1][0].get()),
                    'W': float(self.bbox_entries[2][0].get()),
                    'E': float(self.bbox_entries[3][0].get()),
                }
            except ValueError:
                messagebox.showerror("Error", "Bounding box coordinates must be numbers."); return None

        return dict(roads_dir=roads_dir, stations_path=stations_path,
                    bg_dir=bg_dir, save_dir=save_dir,
                    min_km=min_km, max_km=max_km, bbox=bbox)

    # ----------------------------------------------------------
    # WORKER (background thread)
    # ----------------------------------------------------------
    def _worker(self, params):
        q = self.q
        stop = self._stop_flag

        def log(text, tag=None):
            q.put({'type': 'log', 'text': text, 'tag': tag})

        def prog(pct, label=''):
            q.put({'type': 'progress', 'pct': pct, 'label': label})

        def show_fig(fig, title):
            evt = threading.Event()
            q.put({'type': 'show_fig', 'fig': fig, 'title': title, 'event': evt})
            evt.wait()

        def ask(title, question):
            evt = threading.Event(); result = []
            q.put({'type': 'ask', 'title': title, 'question': question,
                   'event': evt, 'result': result})
            evt.wait()
            return result[0]

        def log_stats(G):
            log(f"      Nodes: {G.number_of_nodes():,}  |  "
                f"Edges: {G.number_of_edges():,}  |  "
                f"Components: {nx.number_connected_components(G):,}")

        try:
            global _node_counter
            _node_counter = 0

            # ── Load roads ─────────────────────────────────────
            log("=" * 60, 'sep')
            log("[1/8] Loading road files…", 'head')
            road_files = sorted(
                glob.glob(os.path.join(params['roads_dir'], '*.geojson')) +
                glob.glob(os.path.join(params['roads_dir'], '*.json'))
            )
            if not road_files:
                raise FileNotFoundError("No .geojson/.json files in roads folder.")

            G = nx.Graph()
            for i, fp in enumerate(road_files):
                if stop.is_set(): q.put({'type':'stopped'}); return
                prog(i/len(road_files)*100, f"Loading {os.path.basename(fp)}")
                G_temp = load_geojson_as_edges(fp)
                for node, attrs in G_temp.nodes(data=True): G.add_node(node, **attrs)
                for u, v, attrs in G_temp.edges(data=True): G.add_edge(u, v, **attrs)
                log(f"  [{i+1}/{len(road_files)}] {os.path.basename(fp)}  "
                    f"(+{G_temp.number_of_edges()} edges)")
            prog(100)
            log_stats(G)
            elev_edges = sum(1 for _,_,d in G.edges(data=True) if d.get('elevations'))
            log(f"  Edges with elevation: {elev_edges:,}" if elev_edges
                else "  No Z-coordinates found — elevation columns will show '—'.", 'warn')

            # ── Bbox filter ────────────────────────────────────
            bbox = params['bbox']
            BBOX_N = BBOX_S = BBOX_W = BBOX_E = None
            if bbox:
                BBOX_N, BBOX_S = bbox['N'], bbox['S']
                BBOX_W, BBOX_E = bbox['W'], bbox['E']
                log(f"\n[1.5/8] Bbox filter: Lat {BBOX_S}°–{BBOX_N}°, Lon {BBOX_W}°–{BBOX_E}°", 'head')
                to_remove = [n for n in G.nodes()
                             if not (BBOX_W <= G.nodes[n]['x'] <= BBOX_E
                                     and BBOX_S <= G.nodes[n]['y'] <= BBOX_N)]
                G.remove_nodes_from(to_remove)
                G.remove_nodes_from(list(nx.isolates(G)))
                log(f"  Removed {len(to_remove):,} nodes outside area.")
                log_stats(G)

            # ── Stitching ──────────────────────────────────────
            log("\n[2/8] Graph stitching…", 'head')
            log("  [SNAP] Coordinate snapping…")
            prog(0, "SNAP")
            t0 = time.time(); G = snap_graph(G)
            log(f"  Done in {time.time()-t0:.1f} s"); log_stats(G)

            log("  [BRIDGE] Initial bridge stitching…")
            prog(10, "BRIDGE")
            t0 = time.time(); do_bridge_stitching(G)
            log(f"  Done in {time.time()-t0:.1f} s"); log_stats(G)

            for r in range(1, N_ROUNDS+1):
                if stop.is_set(): q.put({'type':'stopped'}); return
                log(f"\n  — Round {r}/{N_ROUNDS} —")
                base = 10 + (r-1)/N_ROUNDS*70
                for lbl, fn, dp in [("INTERSECT", do_edge_intersections, 0),
                                    ("T-JUNCTION", do_tjunction, 5),
                                    ("BUFFER",    do_buffer_merge, 10)]:
                    log(f"  [{lbl}]"); prog(base+dp, lbl)
                    t0 = time.time(); fn(G)
                    log(f"  Done in {time.time()-t0:.1f} s"); log_stats(G)
                log("  [BRIDGE extra]"); prog(base+20, "BRIDGE extra")
                t0 = time.time(); do_bridge_stitching(G, label='_r')
                log(f"  Done in {time.time()-t0:.1f} s"); log_stats(G)

            prog(90, "Stitching complete")
            log("\n  STITCHING COMPLETE", 'ok'); log_stats(G)

            # ── Component visualisation ────────────────────────
            log("\n[3/8] Building component map…", 'head')
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            node_comp  = {n: ci for ci, c in enumerate(components) for n in c}
            show_n     = min(10, len(components))
            comp_segs  = [[] for _ in range(show_n)]; other_segs = []
            for u, v, d in G.edges(data=True):
                ci = node_comp[u]
                if 'geometry' in d and d['geometry']:
                    coords = d['geometry']
                    for i in range(len(coords)-1):
                        (comp_segs[ci] if ci < show_n else other_segs).append([coords[i], coords[i+1]])
                else:
                    p1 = (G.nodes[u]['x'],G.nodes[u]['y']); p2 = (G.nodes[v]['x'],G.nodes[v]['y'])
                    (comp_segs[ci] if ci < show_n else other_segs).append([p1, p2])

            fig_comp = mplfig.Figure(figsize=(14, 10))
            ax_c = fig_comp.add_subplot(111)
            cmap_c = mpl_cm.get_cmap('tab10', max(show_n, 1))
            if other_segs:
                ax_c.add_collection(LineCollection(other_segs, colors='#555', linewidths=0.15, alpha=0.3, label='others'))
            for i in range(show_n-1, -1, -1):
                if not comp_segs[i]: continue
                ax_c.add_collection(LineCollection(comp_segs[i], colors=[cmap_c(i)],
                    linewidths=1.5 if i==0 else 0.6, alpha=0.85,
                    label=f'#{i+1} ({len(components[i]):,} nodes)'))
            ax_c.autoscale(); ax_c.set_aspect('equal')
            ax_c.set_xlabel('Longitude'); ax_c.set_ylabel('Latitude')
            bbox_info = (f" | Lat {BBOX_S}°–{BBOX_N}°, Lon {BBOX_W}°–{BBOX_E}°" if bbox else "")
            ax_c.set_title(f"Top-10 components{bbox_info}\n"
                           f"Components: {len(components):,} | Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}",
                           fontsize=11)
            ax_c.legend(loc='upper left', fontsize=7)
            ax_c.grid(True, alpha=0.2)
            fig_comp.tight_layout()

            show_fig(fig_comp, "Top-10 Components — close or click Continue to proceed")
            if stop.is_set(): q.put({'type':'stopped'}); return

            # ── Load stations ──────────────────────────────────
            log("\n[4/8] Loading stations…", 'head')
            with open(params['stations_path'], 'r', encoding='utf-8') as f:
                sdata = json.load(f)
            stations_raw = []; skipped_bbox = 0
            for feat in sdata.get('features', []):
                geom = feat.get('geometry', {}); props = feat.get('properties', {})
                if geom.get('type') == 'Point':
                    lon, lat = geom['coordinates'][:2]
                    in_area = (not bbox) or (BBOX_W <= lon <= BBOX_E and BBOX_S <= lat <= BBOX_N)
                    if not in_area: skipped_bbox += 1; continue
                    name = (props.get('name') or props.get('NAME') or props.get('title') or
                            props.get('TITLE') or f'Station_{len(stations_raw)+1}')
                    stations_raw.append({'name': str(name), 'lon': lon, 'lat': lat})
            log(f"  Found: {len(stations_raw)}" + (f"  (outside area: {skipped_bbox})" if skipped_bbox else ""))

            # ── Project stations ───────────────────────────────
            log("\n[5/8] Projecting stations onto graph…", 'head')
            edges_data = []; edge_lines_proj = []
            for u, v, d in G.edges(data=True):
                coords = d['geometry'] if ('geometry' in d and d['geometry']) else \
                         [(G.nodes[u]['x'],G.nodes[u]['y']),(G.nodes[v]['x'],G.nodes[v]['y'])]
                edges_data.append((u,v,d,coords)); edge_lines_proj.append(LineString(coords))
            strtree_st = STRtree(edge_lines_proj); snap_rad = STATION_SNAP_M/DEG_TO_M
            stations = []
            for si, st in enumerate(stations_raw):
                if stop.is_set(): q.put({'type':'stopped'}); return
                prog(si/max(len(stations_raw),1)*100, "Projecting stations")
                pt = Point(st['lon'], st['lat']); buf = pt.buffer(snap_rad)
                best_dist = float('inf'); best_proj = best_idx = None
                for idx in strtree_st.query(buf):
                    line = edge_lines_proj[idx]
                    proj = line.interpolate(line.project(pt))
                    dm = edge_length_m((pt.x,pt.y),(proj.x,proj.y))
                    if dm < best_dist and dm <= STATION_SNAP_M:
                        best_dist=dm; best_proj=snap_coord((proj.x,proj.y)); best_idx=idx
                if best_proj is not None:
                    nid = f"st_{len(stations)}"
                    u, v, d, _ = edges_data[best_idx]
                    G.add_node(nid, x=best_proj[0], y=best_proj[1], is_station=True, station_name=st['name'])
                    G.add_edge(nid, u, length_m=edge_length_m(best_proj,(G.nodes[u]['x'],G.nodes[u]['y'])), road_type='station_link')
                    G.add_edge(nid, v, length_m=edge_length_m(best_proj,(G.nodes[v]['x'],G.nodes[v]['y'])), road_type='station_link')
                    stations.append({'name':st['name'],'node_id':nid,
                                     'lon':best_proj[0],'lat':best_proj[1],
                                     'orig_lon':st['lon'],'orig_lat':st['lat'],
                                     'snap_dist_m':round(best_dist,1)})
                else:
                    log(f"  ! '{st['name']}' — no edges within {STATION_SNAP_M} m", 'warn')
            prog(100)
            log(f"  Projected: {len(stations)} / {len(stations_raw)}", 'ok')

            # ── Load background ────────────────────────────────
            log("\n[7/8] Loading background files…", 'head')
            bg_classified = {'water':[],'coastline':[],'railway':[],'other':[]}
            bg_dir = params['bg_dir']
            if os.path.isdir(bg_dir):
                bg_files = sorted(glob.glob(os.path.join(bg_dir,'*.geojson')) +
                                  glob.glob(os.path.join(bg_dir,'*.json')))
                log(f"  Files found: {len(bg_files)}")
                for bf in bg_files:
                    fname = os.path.basename(bf)
                    with open(bf,'r',encoding='utf-8') as f: bdata = json.load(f)
                    fc = defaultdict(int)
                    for feat in bdata.get('features',[]):
                        cat = classify_feature(feat, fname); bg_classified[cat].append(feat); fc[cat] += 1
                    log(f"  - {fname}  (" + ", ".join(f"{c}:{n}" for c,n in sorted(fc.items())) + ")")
            else:
                log("  Background folder not found — map will have no background.", 'warn')

            road_segs = []
            for u, v, d in G.edges(data=True):
                if d.get('road_type') == 'station_link': continue
                if 'geometry' in d and d['geometry']:
                    coords = d['geometry']
                    for i in range(len(coords)-1): road_segs.append([coords[i],coords[i+1]])
                else:
                    road_segs.append([(G.nodes[u]['x'],G.nodes[u]['y']),(G.nodes[v]['x'],G.nodes[v]['y'])])

            # ── MAIN LOOP ──────────────────────────────────────
            min_m = params['min_km'] * 1000
            max_m = params['max_km'] * 1000
            iteration = 0

            while True:
                iteration += 1
                if stop.is_set(): q.put({'type':'stopped'}); return
                log(f"\n[6/8] {'Iteration '+str(iteration)+': ' if iteration>1 else ''}"
                    f"Building routes ({params['min_km']}–{params['max_km']} km)…", 'head')

                n_sel = min(N_SELECT, len(stations))
                selected = random.sample(stations, n_sel)
                log(f"  Selected {n_sel} stations:")
                for i, s in enumerate(selected, 1): log(f"    {i}. {s['name']}")

                routes = []
                for si, src in enumerate(selected):
                    if stop.is_set(): q.put({'type':'stopped'}); return
                    prog(si/n_sel*100, f"Route search {si+1}/{n_sel}")
                    found = False
                    for tgt in stations:
                        if tgt['node_id'] == src['node_id']: continue
                        try:
                            path = nx.shortest_path(G, src['node_id'], tgt['node_id'], weight='length_m')
                            length = sum(G[path[k]][path[k+1]]['length_m'] for k in range(len(path)-1))
                            if min_m <= length <= max_m:
                                gain, loss = route_elevation_stats(path, G)
                                routes.append({'from':src['name'],'to':tgt['name'],
                                               'from_id':src['node_id'],'to_id':tgt['node_id'],
                                               'length_m':length,'path':path,
                                               'elev_gain_m':gain,'elev_loss_m':loss})
                                found = True; break
                        except nx.NetworkXNoPath: pass
                    if not found: log(f"  ! No route for '{src['name']}'", 'warn')

                prog(100)
                routes.sort(key=lambda r: r['length_m'])
                log(f"  Routes found: {len(routes)}", 'ok')

                # table
                has_elev = any(r['elev_gain_m'] is not None for r in routes)
                col_w = 88 if has_elev else 72
                if has_elev:
                    header = f"  {'#':<4} {'Start':<22} {'Finish':<22} {'Length, km':>10} {'Gain, m':>10} {'Loss, m':>10}"
                else:
                    header = f"  {'#':<4} {'Start':<25} {'Finish':<25} {'Length, km':>10}"
                lines = ["=" * col_w, header, "-" * col_w]
                for idx, r in enumerate(routes, 1):
                    gain_s = str(r['elev_gain_m']) if r['elev_gain_m'] is not None else '—'
                    loss_s = str(r['elev_loss_m']) if r['elev_loss_m'] is not None else '—'
                    if has_elev:
                        lines.append(f"  {idx:<4} {r['from']:<22} {r['to']:<22} "
                                     f"{r['length_m']/1000:>10.1f} {gain_s:>10} {loss_s:>10}")
                    else:
                        lines.append(f"  {idx:<4} {r['from']:<25} {r['to']:<25} {r['length_m']/1000:>10.1f}")
                lines += ["=" * col_w, f"  Total routes: {len(routes)}"]
                table_text = "\n".join(lines)
                for l in lines: log(l)

                # build segments
                route_color_map = {r['from_id']: ROUTE_COLORS[ri % len(ROUTE_COLORS)]
                                   for ri, r in enumerate(routes)}
                seg_to_routes = defaultdict(list); route_raw_segs = {}
                for ri, route in enumerate(routes):
                    segs = []
                    for k in range(len(route['path'])-1):
                        u, v = route['path'][k], route['path'][k+1]; d = G[u][v]
                        if 'geometry' in d and d['geometry']:
                            coords = d['geometry']
                            for j in range(len(coords)-1):
                                key = (snap_coord(tuple(coords[j])), snap_coord(tuple(coords[j+1])))
                                segs.append(key)
                                if ri not in seg_to_routes[key]: seg_to_routes[key].append(ri)
                        else:
                            p1 = (G.nodes[u]['x'],G.nodes[u]['y']); p2 = (G.nodes[v]['x'],G.nodes[v]['y'])
                            key = (snap_coord(p1), snap_coord(p2)); segs.append(key)
                            if ri not in seg_to_routes[key]: seg_to_routes[key].append(ri)
                    route_raw_segs[ri] = segs

                segs_by_color = defaultdict(list)
                for ri, route in enumerate(routes):
                    color = route_color_map[route['from_id']]
                    for key in route_raw_segs[ri]:
                        p1, p2 = key; ri_list = seg_to_routes[key]; n = len(ri_list)
                        if n == 1:
                            segs_by_color[color].append([p1, p2])
                        else:
                            i = ri_list.index(ri)
                            off_m = (i-(n-1)/2) * ROUTE_OFFSET_M
                            p1_off, p2_off = offset_segment(p1, p2, off_m)
                            segs_by_color[color].append([p1_off, p2_off])

                # build figure
                log("\n[8/8] Rendering map…", 'head')
                fig = mplfig.Figure(figsize=(16, 12))
                ax = fig.add_subplot(111)

                if bbox:
                    pad_lat = 10_000/DEG_TO_M; pad_lon = 10_000/(DEG_TO_M*COS_LAT)
                    ax.set_xlim(BBOX_W-pad_lon, BBOX_E+pad_lon)
                    ax.set_ylim(BBOX_S-pad_lat, BBOX_N+pad_lat)
                    ax.set_aspect('equal')

                for cat in ('other','water','coastline','railway'):
                    draw_bg(ax, bg_classified[cat], cat)

                if road_segs:
                    ax.add_collection(LineCollection(road_segs, colors='#dddddd', linewidths=0.3, alpha=0.4, zorder=2))

                for color, segs in segs_by_color.items():
                    if segs:
                        ax.add_collection(LineCollection(segs, colors=[color], linewidths=2.5, alpha=0.85, zorder=3))

                sel_ids = {s['node_id'] for s in selected}
                for st in stations:
                    is_sel = st['node_id'] in sel_ids
                    if is_sel:
                        ax.plot(st['lon'],st['lat'],'^',color='red',markersize=14,
                                markeredgecolor='#8B0000',markeredgewidth=1.0,zorder=5)
                        ax.annotate(st['name'],(st['lon'],st['lat']),fontsize=10,fontweight='bold',
                                    xytext=(5,6),textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.3',fc='white',alpha=0.85,ec='#ccc'),zorder=6)
                    else:
                        ax.plot(st['lon'],st['lat'],'o',color='#555',markersize=5,zorder=5)
                        ax.annotate(st['name'],(st['lon'],st['lat']),fontsize=8,
                                    xytext=(4,4),textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.2',fc='white',alpha=0.7,ec='none'),zorder=6)

                handles = []
                for route in routes:
                    color = route_color_map.get(route['from_id'], '#E53935')
                    elev_s = (f"  +{route['elev_gain_m']} m / -{route['elev_loss_m']} m"
                              if route['elev_gain_m'] is not None else "")
                    label = f"{route['from']} — {route['to']}  {route['length_m']/1000:.1f} km{elev_s}"
                    handles.append(mlines.Line2D([],[],color=color,linewidth=2.5,label=label))
                if handles:
                    ax.legend(handles=handles, loc='upper right', fontsize=8,
                              framealpha=0.85, edgecolor='#ccc', title='Routes', title_fontsize=10)

                if not bbox: ax.autoscale(); ax.set_aspect('equal')
                ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
                bbox_info = (f" | Lat {BBOX_S}°–{BBOX_N}°, Lon {BBOX_W}°–{BBOX_E}°" if bbox else "")
                iter_label = f" | Iteration {iteration}" if iteration > 1 else ""
                ax.set_title(f"Routes from {n_sel} stations{bbox_info}{iter_label}\n"
                             f"Stations: {len(stations)} | Routes: {len(routes)} | "
                             f"{params['min_km']}–{params['max_km']} km", fontsize=12)
                ax.grid(True, alpha=0.2)
                fig.tight_layout()

                show_fig(fig, f"Routes — Iteration {iteration}")

                # save
                save_dir = params['save_dir']
                if save_dir and ask("Save results?",
                                    f"Save PNG, TXT and GeoJSON to:\n{save_dir}?"):
                    ts  = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    out = os.path.join(save_dir, ts)
                    os.makedirs(out, exist_ok=True)

                    # PNG
                    png = os.path.join(out, f'routes_{ts}.png')
                    fig.savefig(png, dpi=150, bbox_inches='tight')
                    log(f"  PNG: {png}", 'ok')

                    # TXT
                    txt = os.path.join(out, f'routes_{ts}.txt')
                    with open(txt, 'w', encoding='utf-8') as f: f.write(table_text + "\n")
                    log(f"  TXT: {txt}", 'ok')

                    def route_coords(route):
                        path = route['path']; coords = []
                        for k in range(len(path)-1):
                            u, v = path[k], path[k+1]; d = G[u][v]
                            ec = d['geometry'] if ('geometry' in d and d['geometry']) else \
                                 [(G.nodes[u]['x'],G.nodes[u]['y']),(G.nodes[v]['x'],G.nodes[v]['y'])]
                            coords.extend(ec if not coords else ec[1:])
                        return [[c[0],c[1]] for c in coords]

                    features_out = []
                    for cat, feats in bg_classified.items():
                        for feat in feats:
                            fc = dict(feat)
                            if 'properties' not in fc: fc['properties'] = {}
                            fc['properties']['_bg_category'] = cat
                            features_out.append(fc)
                    for st in stations:
                        features_out.append({'type':'Feature',
                            'geometry':{'type':'Point','coordinates':[st['lon'],st['lat']]},
                            'properties':{'name':st['name'],'type':'station',
                                          'snap_dist_m':st['snap_dist_m'],'selected':st['node_id'] in sel_ids}})
                    for route in routes:
                        props = {'type':'route','from':route['from'],'to':route['to'],
                                 'length_m':round(route['length_m'],2)}
                        if route['elev_gain_m'] is not None:
                            props['elev_gain_m'] = route['elev_gain_m']
                            props['elev_loss_m'] = route['elev_loss_m']
                        features_out.append({'type':'Feature',
                            'geometry':{'type':'LineString','coordinates':route_coords(route)},
                            'properties':props})
                    for u, v, d in G.edges(data=True):
                        if d.get('road_type') == 'station_link': continue
                        coords = [[lon,lat] for lon,lat in d['geometry']] if ('geometry' in d and d['geometry']) else \
                                 [[G.nodes[u]['x'],G.nodes[u]['y']],[G.nodes[v]['x'],G.nodes[v]['y']]]
                        features_out.append({'type':'Feature',
                            'geometry':{'type':'LineString','coordinates':coords},
                            'properties':{'type':'road','length_m':round(d.get('length_m',0),2),'road_type':d.get('road_type','')}})
                    bbox_meta = ({'bbox':{'N':BBOX_N,'S':BBOX_S,'W':BBOX_W,'E':BBOX_E}} if bbox else {})
                    gj = {'type':'FeatureCollection',
                          'metadata':{'iteration':iteration,'total_stations':len(stations),
                                      'routes_count':len(routes),'min_route_km':params['min_km'],
                                      'max_route_km':params['max_km'], **bbox_meta},
                          'features':features_out}
                    gj_path = os.path.join(out, f'routes_{ts}.geojson')
                    with open(gj_path,'w',encoding='utf-8') as f: json.dump(gj, f, ensure_ascii=False)
                    sz = os.path.getsize(gj_path)/(1024*1024)
                    log(f"  GeoJSON: {gj_path}  ({sz:.1f} MB)", 'ok')

                    rd = os.path.join(out, 'routes'); os.makedirs(rd, exist_ok=True)
                    for route in routes:
                        st_from = next((s for s in stations if s['node_id']==route['from_id']),None)
                        st_to   = next((s for s in stations if s['node_id']==route['to_id']),None)
                        rp = {'from':route['from'],'to':route['to'],
                              'length_km':round(route['length_m']/1000,2)}
                        if route['elev_gain_m'] is not None:
                            rp['elev_gain_m']=route['elev_gain_m']; rp['elev_loss_m']=route['elev_loss_m']
                        rf = [{'type':'Feature','geometry':{'type':'LineString','coordinates':route_coords(route)},'properties':rp}]
                        if st_from: rf.append({'type':'Feature','geometry':{'type':'Point','coordinates':[st_from['lon'],st_from['lat']]},'properties':{'name':route['from'],'role':'start'}})
                        if st_to:   rf.append({'type':'Feature','geometry':{'type':'Point','coordinates':[st_to['lon'],  st_to['lat']  ]},'properties':{'name':route['to'],  'role':'finish'}})
                        fn = f"{safe_filename(route['from'])}_to_{safe_filename(route['to'])}.geojson"
                        with open(os.path.join(rd,fn),'w',encoding='utf-8') as f: json.dump({'type':'FeatureCollection','features':rf},f,ensure_ascii=False)
                    log(f"\n  Saved to: {out}", 'ok')

                # run again?
                if not ask("Run again?", "Run another search with new randomly selected stations?"):
                    break

            q.put({'type': 'done'})

        except Exception as e:
            import traceback
            q.put({'type': 'error', 'text': traceback.format_exc()})

# ================================================================
# ENTRY POINT
# ================================================================
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == '__main__':
    main()

import json
import math
import os
import re
import glob
import time
import random
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ================================================================
# ПАРАМЕТРЫ
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

node_counter = 0

# ================================================================
# ВВОД: ОДНА ПАПКА TRIPDATA
# ================================================================
TRIPDATA = input("Введите путь к папке tripdata: ").strip().strip('"').strip("'")

if not os.path.isdir(TRIPDATA):
    raise FileNotFoundError(f"Папка не найдена: {TRIPDATA}")

ROADS_DIR = os.path.join(TRIPDATA, 'roads')
STATIONS_DIR = os.path.join(TRIPDATA, 'stations')
BG_DIR = os.path.join(TRIPDATA, 'background')

# --- Проверки подпапок ---
missing = []
for path, label in [(ROADS_DIR, 'roads'), (STATIONS_DIR, 'stations'), (BG_DIR, 'background')]:
    if not os.path.isdir(path):
        missing.append(f"  ! Подпапка не найдена: {label}/ ({path})")
if missing:
    print("\n".join(missing))
    raise FileNotFoundError("Одна или несколько подпапок tripdata отсутствуют.")

# --- Файлы дорог ---
road_files = sorted(
    glob.glob(os.path.join(ROADS_DIR, '*.geojson')) +
    glob.glob(os.path.join(ROADS_DIR, '*.json'))
)
if not road_files:
    raise FileNotFoundError(f"В папке roads/ нет .geojson/.json файлов: {ROADS_DIR}")

# --- Файлы станций ---
station_files = sorted(
    glob.glob(os.path.join(STATIONS_DIR, '*.geojson')) +
    glob.glob(os.path.join(STATIONS_DIR, '*.json'))
)
if not station_files:
    raise FileNotFoundError(f"В папке stations/ нет .geojson/.json файлов: {STATIONS_DIR}")

print("=" * 72)
print(f"  tripdata: {TRIPDATA}")
print(f"  roads/    — {len(road_files)} файл(ов)")
for f in road_files:
    print(f"    - {os.path.basename(f)}")
print(f"  stations/ — {len(station_files)} файл(ов)")
for f in station_files:
    print(f"    - {os.path.basename(f)}")
print(f"  background/ — будет использован для визуализации")
print("=" * 72)

# ================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ================================================================
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
    total = 0.0
    for i in range(len(coords) - 1):
        total += edge_length_m(coords[i], coords[i + 1])
    return total

def in_bbox(lon, lat):
    if BBOX is None:
        return True
    return BBOX_W <= lon <= BBOX_E and BBOX_S <= lat <= BBOX_N

def safe_filename(s):
    s = re.sub(r'[\\/*?:"<>|]', '_', s)
    s = re.sub(r'\s+', '_', s)
    return s.strip('._') or 'маршрут'

def offset_segment(p1, p2, offset_m):
    dx_m = (p2[0] - p1[0]) * DEG_TO_M * COS_LAT
    dy_m = (p2[1] - p1[1]) * DEG_TO_M
    len_m = math.hypot(dx_m, dy_m)
    if len_m < 1e-9:
        return p1, p2
    off_lon = offset_m * (-dy_m / len_m) / (DEG_TO_M * COS_LAT)
    off_lat = offset_m * (dx_m / len_m) / DEG_TO_M
    return (p1[0] + off_lon, p1[1] + off_lat), (p2[0] + off_lon, p2[1] + off_lat)

class UF:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0] * n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
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

# ================================================================
# ФУНКЦИИ ВЫСОТЫ
# ================================================================
def route_elevation_stats(route, G):
    gain = 0.0
    loss = 0.0
    has_data = False
    for k in range(len(route['path']) - 1):
        u, v = route['path'][k], route['path'][k + 1]
        elevs = G[u][v].get('elevations')
        if not elevs:
            continue
        valid = [e for e in elevs if e is not None]
        if len(valid) < 2:
            continue
        has_data = True
        for j in range(len(valid) - 1):
            diff = valid[j + 1] - valid[j]
            if diff > 0:
                gain += diff
            else:
                loss += abs(diff)
    if not has_data:
        return None, None
    return round(gain), round(loss)

def format_elev(value):
    return '—' if value is None else str(value)

# ================================================================
# ЗАГРУЗКА ДОРОГ ИЗ GEOJSON
# ================================================================
def load_geojson_as_edges(filepath):
    global node_counter
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    G_temp = nx.Graph()
    source_file = os.path.basename(filepath)
    if data.get("type") != "FeatureCollection" or "features" not in data:
        print(f"  Пропущен: {source_file}"); return G_temp
    for feature in data["features"]:
        geom = feature.get("geometry", {}); props = feature.get("properties", {})
        geom_type = geom.get("type")
        def process_linestring(coords):
            global node_counter
            if len(coords) < 2: return
            coords_2d = [(round(c[0], COORD_PRECISION), round(c[1], COORD_PRECISION)) for c in coords]
            elevations = [c[2] if len(c) > 2 else None for c in coords]
            has_ele = any(e is not None for e in elevations)
            if coords_2d[0] == coords_2d[-1]: return
            sn = f"node_{node_counter}"; node_counter += 1
            en = f"node_{node_counter}"; node_counter += 1
            G_temp.add_node(sn, x=coords_2d[0][0], y=coords_2d[0][1])
            G_temp.add_node(en, x=coords_2d[-1][0], y=coords_2d[-1][1])
            edge_attrs = dict(
                length_m=round(linestring_length_m(coords_2d), 2),
                road_type=props.get("highway", props.get("type", "")),
                source_file=source_file,
                geometry=coords_2d,
                original_edge=True,
            )
            if has_ele:
                edge_attrs['elevations'] = elevations
            G_temp.add_edge(sn, en, **edge_attrs)
        if geom_type == "LineString": process_linestring(geom["coordinates"])
        elif geom_type == "MultiLineString":
            for lc in geom["coordinates"]: process_linestring(lc)
    return G_temp

# ================================================================
# ЗАГРУЗКА СТАНЦИЙ ИЗ ПАПКИ (все .geojson/.json)
# ================================================================
def load_stations_from_dir(stations_dir):
    """Загружает Point-фичи из всех GeoJSON файлов в папке stations/."""
    files = sorted(
        glob.glob(os.path.join(stations_dir, '*.geojson')) +
        glob.glob(os.path.join(stations_dir, '*.json'))
    )
    raw = []
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        loaded = 0
        for feat in data.get('features', []):
            geom = feat.get('geometry', {})
            props = feat.get('properties', {})
            if geom.get('type') != 'Point':
                continue
            lon, lat = geom['coordinates'][:2]
            name = (props.get('name') or props.get('NAME') or
                    props.get('title') or props.get('TITLE') or
                    f'Станция_{len(raw) + 1}')
            raw.append({'name': str(name), 'lon': lon, 'lat': lat})
            loaded += 1
        print(f"    - {fname}  ({loaded} точек)")
    return raw

# ================================================================
# ОПЕРАЦИИ СШИВКИ
# ================================================================
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
    total = 0
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
                G.add_edge(nl[i], nl[j], length_m=float(d_val), road_type='bridge',
                           source_file=f'auto_bridge{label}', original_edge=False)
                added += 1
        total += added
        if added == 0: break
    return total

def do_edge_intersections(G):
    edges_info = []; edge_lines = []
    for u, v, d in G.edges(data=True):
        coords = d['geometry'] if ('geometry' in d and d['geometry']) else \
                 [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]
        edges_info.append((u, v, dict(d), coords))
        edge_lines.append(LineString(coords))
    strtree_ei = STRtree(edge_lines); split_pts = defaultdict(list)
    eps = 10 ** -(SNAP_PRECISION + 1)
    for i, (u1, v1, d1, coords1) in enumerate(edges_info):
        for j in strtree_ei.query(edge_lines[i]):
            if j <= i: continue
            u2, v2, d2, coords2 = edges_info[j]
            if u1 in (u2, v2) or v1 in (u2, v2): continue
            if not edge_lines[i].intersects(edge_lines[j]): continue
            pt = edge_lines[i].intersection(edge_lines[j])
            if pt.geom_type != 'Point': continue
            endpoints = [coords1[0], coords1[-1], coords2[0], coords2[-1]]
            if any(abs(pt.x-px)<eps and abs(pt.y-py)<eps for px, py in endpoints): continue
            sc = snap_coord((pt.x, pt.y)); split_pts[i].append(sc); split_pts[j].append(sc)
    to_remove = set(); to_add = []
    for ei, pts in split_pts.items():
        u, v, d, coords = edges_info[ei]; to_remove.add((u, v))
        all_pts = [coords[0]] + pts + [coords[-1]]; sx, sy = coords[0]
        all_pts.sort(key=lambda c: (c[0]-sx)**2 + (c[1]-sy)**2)
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
        G.add_edge(sn, en, length_m=edge_length_m(sc, ec), road_type=rt, source_file=sf, original_edge=False)
    G.remove_nodes_from(list(nx.isolates(G)))

def do_tjunction(G):
    comp_map, _ = build_comp_map(G)
    dangling = [n for n in G.nodes() if G.degree(n) == 1]
    tj_edges = []; tj_geoms = []
    for u, v, d in G.edges(data=True):
        coords = d['geometry'] if ('geometry' in d and d['geometry']) else \
                 [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]
        tj_edges.append((u, v, d)); tj_geoms.append(LineString(coords))
    tj_tree = STRtree(tj_geoms); tj_rad_deg = TJUNCTION_RADIUS_M / DEG_TO_M
    edge_projs = defaultdict(list)
    for n in dangling:
        nd = G.nodes[n]; pt = Point(nd['x'], nd['y']); cn = comp_map[n]
        buf = pt.buffer(tj_rad_deg); best_dist = TJUNCTION_RADIUS_M; best_idx = None; best_proj = None
        for idx in tj_tree.query(buf):
            eu, ev, ed = tj_edges[idx]
            if comp_map.get(eu) == cn: continue
            proj = tj_geoms[idx].interpolate(tj_geoms[idx].project(pt))
            dm = edge_length_m((pt.x, pt.y), (proj.x, proj.y))
            if dm < best_dist:
                best_dist = dm; best_idx = idx; best_proj = snap_coord((proj.x, proj.y))
        if best_idx is not None:
            edge_projs[best_idx].append((best_proj, n, best_dist))
    for ei, projs in edge_projs.items():
        u, v, d = tj_edges[ei]
        if not G.has_edge(u, v): continue
        rt = d.get('road_type',''); sf = d.get('source_file','')
        coords = d['geometry'] if ('geometry' in d and d['geometry']) else \
                 [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]
        start, end = snap_coord(coords[0]), snap_coord(coords[-1])
        all_p = [start]; pm = defaultdict(list)
        for proj_c, dn, dm in projs:
            if proj_c != start and proj_c != end: all_p.append(proj_c)
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
            G.add_edge(sn, en, length_m=edge_length_m(sc, ec), road_type=rt, source_file=sf, original_edge=False)
        for proj_c, dang_list in pm.items():
            pid = make_nid(proj_c)
            if pid not in G: G.add_node(pid, x=proj_c[0], y=proj_c[1])
            for dn, dm in dang_list:
                G.add_edge(dn, pid, length_m=dm, road_type='t_junction', source_file='auto_tj', original_edge=False)
    G.remove_nodes_from(list(nx.isolates(G)))

def do_buffer_merge(G):
    comp_map2, comps2 = build_comp_map(G)
    buf_deg = BUFFER_M / DEG_TO_M; comp_edges_geom = defaultdict(list)
    for u, v, d in G.edges(data=True):
        ci = comp_map2[u]
        line = LineString(d['geometry']) if ('geometry' in d and d['geometry']) else \
               LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
        comp_edges_geom[ci].append(line)
    big_comps = [(ci, geoms) for ci, geoms in comp_edges_geom.items() if len(geoms) >= 3]
    comp_buffers = [unary_union(geoms).buffer(buf_deg) for ci, geoms in big_comps]
    comp_ids_buf = [ci for ci, _ in big_comps]
    buf_tree = STRtree(comp_buffers); buf_pairs = set()
    for i, geom in enumerate(comp_buffers):
        for j in buf_tree.query(geom):
            if j <= i: continue
            ci, cj = comp_ids_buf[i], comp_ids_buf[j]
            if ci != cj and comp_buffers[i].intersects(comp_buffers[j]):
                buf_pairs.add((i, j))
    for i, j in buf_pairs:
        ci, cj = comp_ids_buf[i], comp_ids_buf[j]
        nodes_i = list(comps2[ci]); nodes_j = list(comps2[cj])
        if not nodes_i or not nodes_j: continue
        coords_i = np.array([deg_to_m(G.nodes[n]['x'], G.nodes[n]['y']) for n in nodes_i])
        coords_j = np.array([deg_to_m(G.nodes[n]['x'], G.nodes[n]['y']) for n in nodes_j])
        ti = cKDTree(coords_i); dists, idxs = ti.query(coords_j, k=1)
        best = np.argmin(dists); bi = idxs[best]; dm = dists[best]
        if dm > BRIDGE_RADIUS_M: continue
        ni, nj = nodes_i[bi], nodes_j[best]
        if not G.has_edge(ni, nj):
            G.add_edge(ni, nj, length_m=float(dm), road_type='buffer_merge',
                       source_file='auto_buffer', original_edge=False)
    G.remove_nodes_from(list(nx.isolates(G)))

# ================================================================
# КЛАССИФИКАЦИЯ ФОНОВЫХ ОБЪЕКТОВ
# ================================================================
WATER_KEYS = ('water','river','lake','sea','ocean','bay','strait','reservoir','pond',
              'вод','река','озер','море','залив','пруд','водохр','ручей','канал')
COAST_KEYS = ('coastline','coast','берег','побереж')
RAIL_KEYS  = ('railway','rail','жд','жел_дор','железн')

def classify_feature(feat, filename):
    props = feat.get('properties', {})
    vals = ' '.join(str(v).lower() for v in props.values() if v)
    keys_present = set(k.lower() for k in props.keys()); fn = filename.lower()
    if 'railway' in keys_present or any(k in fn for k in RAIL_KEYS) or any(k in vals for k in ('rail','railway')):
        return 'railway'
    nat = str(props.get('natural','')).lower()
    if nat == 'coastline' or any(k in fn for k in COAST_KEYS) or any(k in vals for k in COAST_KEYS):
        return 'coastline'
    if ('waterway' in keys_present or 'water' in keys_present or
            nat in ('water','bay','strait','wetland') or
            any(k in fn for k in WATER_KEYS) or any(k in vals for k in WATER_KEYS)):
        return 'water'
    return 'other'

BG_STYLES = {
    'water':     {'line_color':'#2196F3','fill_color':'#BBDEFB','edge_color':'#1976D2','lw':0.6,'alpha':0.6,'fill_alpha':0.35},
    'coastline': {'line_color':'#2196F3','fill_color':'#BBDEFB','edge_color':'#1976D2','lw':0.8,'alpha':0.7,'fill_alpha':0.1},
    'railway':   {'line_color':'#000000','fill_color':'#000000','edge_color':'#000000','lw':1.2,'alpha':0.8,'fill_alpha':0.3},
    'other':     {'line_color':'#aaaaaa','fill_color':'#e0e0e0','edge_color':'#bbbbbb','lw':0.4,'alpha':0.5,'fill_alpha':0.4},
}

ROUTE_COLORS = ['#E53935','#43A047','#FB8C00','#8E24AA','#00ACC1',
                '#F4511E','#7CB342','#D81B60','#FFB300','#5E35B1']

# ================================================================
# 1. ЗАГРУЗКА ДОРОГ
# ================================================================
print("\n[1/8] Загрузка файлов дорог из roads/...")
graphs = []
for filepath in tqdm(road_files, desc="Загрузка"):
    G_temp = load_geojson_as_edges(filepath)
    if G_temp.number_of_edges() > 0: graphs.append(G_temp)

G = nx.Graph()
for G_temp in graphs:
    for node, attrs in G_temp.nodes(data=True): G.add_node(node, **attrs)
    for u, v, attrs in G_temp.edges(data=True): G.add_edge(u, v, **attrs)
print(f"  Узлов: {G.number_of_nodes():,}  |  Рёбер: {G.number_of_edges():,}  |  Компонент: {nx.number_connected_components(G):,}")

edges_with_elevation = sum(1 for _, _, d in G.edges(data=True) if d.get('elevations'))
if edges_with_elevation:
    print(f"  Рёбер с данными о высоте: {edges_with_elevation:,}")
else:
    print("  Данные о высоте (Z-координаты) не найдены — набор/спуск будет '—'.")

# ================================================================
# 1.5. ФИЛЬТРАЦИЯ ПО КООРДИНАТАМ (опционально)
# ================================================================
print("\n[1.5/8] Ограничения по координатам...")
use_bbox = input("Ограничить область по координатам? (да/нет): ").strip().lower()
BBOX = None; BBOX_N = BBOX_S = BBOX_W = BBOX_E = None
if use_bbox in ('да','yes','y','д'):
    BBOX_N = float(input("  Северная граница (широта): "))
    BBOX_S = float(input("  Южная граница   (широта): "))
    BBOX_W = float(input("  Западная граница (долгота): "))
    BBOX_E = float(input("  Восточная граница (долгота): "))
    BBOX = (BBOX_S, BBOX_N, BBOX_W, BBOX_E)
    to_remove = [n for n in G.nodes()
                 if not (BBOX_W <= G.nodes[n]['x'] <= BBOX_E and BBOX_S <= G.nodes[n]['y'] <= BBOX_N)]
    G.remove_nodes_from(to_remove); G.remove_nodes_from(list(nx.isolates(G)))
    print(f"  Удалено узлов вне области: {len(to_remove):,}")
    print(f"  После фильтрации: {G.number_of_nodes():,} узлов | {G.number_of_edges():,} рёбер | {nx.number_connected_components(G):,} компонент")
    print(f"  Область: Ш {BBOX_S}°–{BBOX_N}°, Д {BBOX_W}°–{BBOX_E}°")
else:
    print("  Ограничения не заданы — используется весь граф.")

# ================================================================
# 2. СШИВКА ГРАФА
# ================================================================
print("\n[2/8] Сшивка графа")
print("=" * 72)

def log_stats(name, G, t0):
    dt = time.time() - t0
    print(f"      Узлов: {G.number_of_nodes():,}  |  Рёбер: {G.number_of_edges():,}  |  Компонент: {nx.number_connected_components(G):,}  |  {dt:.1f} сек")

print("  [SNAP] Привязка координат...")
pbar = tqdm(total=1, desc="  SNAP", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
t0 = time.time(); G = snap_graph(G); pbar.update(1); pbar.close(); log_stats("SNAP", G, t0)

print("  [BRIDGE] Первичная сшивка мостами...")
pbar = tqdm(total=1, desc="  BRIDGE", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
t0 = time.time(); do_bridge_stitching(G); pbar.update(1); pbar.close(); log_stats("BRIDGE", G, t0)

for r in range(1, N_ROUNDS + 1):
    print(f"\n  --- Раунд {r}/{N_ROUNDS} ---")
    for lbl, fn in [("INTERSECT", do_edge_intersections), ("T-JUNCTION", do_tjunction), ("BUFFER", do_buffer_merge)]:
        print(f"  [{lbl} {r}]")
        pbar = tqdm(total=1, desc=f"  {lbl} {r}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        t0 = time.time(); fn(G); pbar.update(1); pbar.close(); log_stats(f"{lbl} {r}", G, t0)
    print(f"  [BRIDGE {r}] Дополнительная сшивка мостами...")
    pbar = tqdm(total=1, desc=f"  BRIDGE {r}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    t0 = time.time(); do_bridge_stitching(G, label='_r'); pbar.update(1); pbar.close(); log_stats(f"BRIDGE {r}", G, t0)

print("=" * 72)
print(f"  ИТОГ:  {G.number_of_nodes():,} узлов | {G.number_of_edges():,} рёбер | {nx.number_connected_components(G):,} компонент")

# ================================================================
# 3. ВИЗУАЛИЗАЦИЯ ТОП-10 КОМПОНЕНТ
# ================================================================
print("\n[3/8] Визуализация топ-10 компонент...")
components = sorted(nx.connected_components(G), key=len, reverse=True)
node_comp = {n: ci for ci, c in enumerate(components) for n in c}
show_n = min(10, len(components))
comp_segs = [[] for _ in range(show_n)]; other_segs = []
for u, v, d in G.edges(data=True):
    ci = node_comp[u]
    if 'geometry' in d and d['geometry']:
        coords = d['geometry']
        for i in range(len(coords)-1):
            (comp_segs[ci] if ci < show_n else other_segs).append([coords[i], coords[i+1]])
    else:
        p1 = (G.nodes[u]['x'], G.nodes[u]['y']); p2 = (G.nodes[v]['x'], G.nodes[v]['y'])
        (comp_segs[ci] if ci < show_n else other_segs).append([p1, p2])

cmap_c = mpl_cm.get_cmap('tab10', max(show_n, 1))
fig_comp, ax_comp = plt.subplots(figsize=(16, 12))
if other_segs:
    ax_comp.add_collection(LineCollection(other_segs, colors='#dddddd', linewidths=0.15, alpha=0.3, label='остальные'))
for i in range(show_n-1, -1, -1):
    if not comp_segs[i]: continue
    ax_comp.add_collection(LineCollection(comp_segs[i], colors=[cmap_c(i)],
                                          linewidths=1.5 if i == 0 else 0.6, alpha=0.85,
                                          label=f'#{i+1} ({len(components[i]):,} узл.)'))
ax_comp.autoscale(); ax_comp.set_aspect('equal')
ax_comp.set_xlabel('Долгота'); ax_comp.set_ylabel('Широта')
bbox_info = (f" | Ш {BBOX_S}°–{BBOX_N}°, Д {BBOX_W}°–{BBOX_E}°" if BBOX else "")
ax_comp.set_title(
    f'Топ-10 компонент{bbox_info}\n'
    f'Компонент: {len(components):,} | Узлов: {G.number_of_nodes():,} | Рёбер: {G.number_of_edges():,}',
    fontsize=12)
ax_comp.legend(loc='upper left', fontsize=8); ax_comp.grid(True, alpha=0.2)
plt.tight_layout(); plt.show()

# ================================================================
# 4. ПРОДОЛЖИТЬ?
# ================================================================
cont = input("\nПродолжить работу? (да/нет): ").strip().lower()
if cont not in ('да','yes','y','д'):
    print("Работа завершена."); raise SystemExit

# ================================================================
# 5. ЗАГРУЗКА СТАНЦИЙ ИЗ stations/
# ================================================================
print("\n[4/8] Загрузка станций из stations/...")
stations_raw_all = load_stations_from_dir(STATIONS_DIR)

# Фильтрация по bbox
stations_raw = []
skipped_bbox = 0
for st in stations_raw_all:
    if in_bbox(st['lon'], st['lat']):
        stations_raw.append(st)
    else:
        skipped_bbox += 1

print(f"  Всего точек: {len(stations_raw_all)}" +
      (f"  |  вне области: {skipped_bbox}" if skipped_bbox else "") +
      f"  |  используется: {len(stations_raw)}")

# ================================================================
# 5.5. ПРОЕЦИРОВАНИЕ СТАНЦИЙ НА ГРАФ
# ================================================================
print("\n[5/8] Проецирование станций на граф...")
edges_data = []; edge_lines_proj = []
for u, v, d in G.edges(data=True):
    coords = d['geometry'] if ('geometry' in d and d['geometry']) else \
             [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]
    edges_data.append((u, v, d, coords)); edge_lines_proj.append(LineString(coords))
strtree_st = STRtree(edge_lines_proj); snap_rad = STATION_SNAP_M / DEG_TO_M
stations = []
for st in tqdm(stations_raw, desc="Проецирование"):
    pt = Point(st['lon'], st['lat']); buf = pt.buffer(snap_rad)
    best_dist = float('inf'); best_proj = None; best_idx = None
    for idx in strtree_st.query(buf):
        line = edge_lines_proj[idx]
        proj = line.interpolate(line.project(pt))
        dm = edge_length_m((pt.x, pt.y), (proj.x, proj.y))
        if dm < best_dist and dm <= STATION_SNAP_M:
            best_dist = dm; best_proj = snap_coord((proj.x, proj.y)); best_idx = idx
    if best_proj is not None:
        node_id = f"st_{len(stations)}"
        u, v, d, coords = edges_data[best_idx]
        G.add_node(node_id, x=best_proj[0], y=best_proj[1], is_station=True, station_name=st['name'])
        G.add_edge(node_id, u, length_m=edge_length_m(best_proj, (G.nodes[u]['x'], G.nodes[u]['y'])), road_type='station_link')
        G.add_edge(node_id, v, length_m=edge_length_m(best_proj, (G.nodes[v]['x'], G.nodes[v]['y'])), road_type='station_link')
        stations.append({'name': st['name'], 'node_id': node_id,
                         'lon': best_proj[0], 'lat': best_proj[1],
                         'orig_lon': st['lon'], 'orig_lat': st['lat'],
                         'snap_dist_m': round(best_dist, 1)})
    else:
        print(f"  ! '{st['name']}' — нет рёбер в радиусе {STATION_SNAP_M} м")
print(f"  Спроецировано: {len(stations)} / {len(stations_raw)}")

# ================================================================
# 6. ДИАПАЗОН МАРШРУТОВ
# ================================================================
MIN_ROUTE_KM = float(input("Мин. длина маршрута (км): "))
MAX_ROUTE_KM = float(input("Макс. длина маршрута (км): "))

# ================================================================
# 7. ЗАГРУЗКА ФОНОВЫХ ФАЙЛОВ из background/
#    (только для визуализации, в обсчёт не входят)
# ================================================================
print(f"\n[7/8] Загрузка фоновых файлов из background/...")
bg_classified = {'water':[],'coastline':[],'railway':[],'other':[]}
bg_files = sorted(
    glob.glob(os.path.join(BG_DIR, '*.geojson')) +
    glob.glob(os.path.join(BG_DIR, '*.json'))
)
if bg_files:
    print(f"  Найдено файлов: {len(bg_files)}")
    counts = defaultdict(int)
    for bf in bg_files:
        fname = os.path.basename(bf); print(f"    - {fname}", end="")
        with open(bf, 'r', encoding='utf-8') as f: bdata = json.load(f)
        fc = defaultdict(int)
        for feat in bdata.get('features', []):
            cat = classify_feature(feat, fname)
            bg_classified[cat].append(feat); counts[cat] += 1; fc[cat] += 1
        print("  (" + ", ".join(f"{c}: {n}" for c, n in sorted(fc.items())) + ")")
    print(f"  Итого: вода={counts['water']}, берег={counts['coastline']}, жд={counts['railway']}, прочее={counts['other']}")
else:
    print(f"  Файлов не найдено в background/ — фон будет пустым.")

road_segs = []
for u, v, d in G.edges(data=True):
    if d.get('road_type') == 'station_link': continue
    if 'geometry' in d and d['geometry']:
        coords = d['geometry']
        for i in range(len(coords)-1): road_segs.append([coords[i], coords[i+1]])
    else:
        road_segs.append([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])

# ================================================================
# ФУНКЦИЯ ОТРИСОВКИ ФОНА
# ================================================================
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
            poly = MplPolygon([(p[0],p[1]) for p in crd[0]],closed=True,
                              fc=s['fill_color'],ec=s['edge_color'],
                              linewidth=s['lw'],alpha=s['fill_alpha'],zorder=1)
            ax.add_patch(poly)
        elif gt == 'MultiPolygon':
            for polygon in crd:
                poly = MplPolygon([(p[0],p[1]) for p in polygon[0]],closed=True,
                                  fc=s['fill_color'],ec=s['edge_color'],
                                  linewidth=s['lw'],alpha=s['fill_alpha'],zorder=1)
                ax.add_patch(poly)
        elif gt == 'Point':
            ax.plot(crd[0],crd[1],'.',color=s['line_color'],markersize=2,alpha=s['alpha'],zorder=1)

# ================================================================
# ОСНОВНОЙ ЦИКЛ: поиск → визуализация → сохранение
# ================================================================
iteration = 0
while True:
    iteration += 1
    print("\n" + "=" * 72)
    iter_tag = f"Итерация {iteration}: " if iteration > 1 else ""
    print(f"[6/8] {iter_tag}Построение маршрутов ({MIN_ROUTE_KM}–{MAX_ROUTE_KM} км)...")

    n_sel = min(N_SELECT, len(stations))
    selected = random.sample(stations, n_sel)
    print(f"  Выбрано {n_sel} станций:")
    for i, s in enumerate(selected, 1): print(f"    {i}. {s['name']}")

    min_m = MIN_ROUTE_KM * 1000; max_m = MAX_ROUTE_KM * 1000
    routes = []
    for src in tqdm(selected, desc="Поиск маршрутов"):
        found = False
        for tgt in stations:
            if tgt['node_id'] == src['node_id']: continue
            try:
                path = nx.shortest_path(G, src['node_id'], tgt['node_id'], weight='length_m')
                length = sum(G[path[k]][path[k+1]]['length_m'] for k in range(len(path)-1))
                if min_m <= length <= max_m:
                    gain, loss = route_elevation_stats({'path': path}, G)
                    routes.append({'from': src['name'], 'to': tgt['name'],
                                   'from_id': src['node_id'], 'to_id': tgt['node_id'],
                                   'length_m': length, 'path': path,
                                   'elev_gain_m': gain, 'elev_loss_m': loss})
                    found = True; break
            except nx.NetworkXNoPath: pass
        if not found: print(f"  ! Маршрут не найден для '{src['name']}'")

    print(f"  Маршрутов: {len(routes)}")
    if routes:
        routes.sort(key=lambda r: r['length_m'])
        print(f"  Длина: {routes[0]['length_m']/1000:.1f} — {routes[-1]['length_m']/1000:.1f} км")

    has_any_elevation = any(r['elev_gain_m'] is not None for r in routes)

    if has_any_elevation:
        col_w = 88
        header = f"  {'№':<4} {'Старт':<22} {'Финиш':<22} {'Длина, км':>10} {'Набор, м':>10} {'Спуск, м':>10}"
    else:
        col_w = 72
        header = f"  {'№':<4} {'Старт':<25} {'Финиш':<25} {'Длина, км':>10}"

    table_lines = ["=" * col_w, header, "-" * col_w]
    for idx, r in enumerate(routes, 1):
        if has_any_elevation:
            table_lines.append(
                f"  {idx:<4} {r['from']:<22} {r['to']:<22} "
                f"{r['length_m']/1000:>10.1f} "
                f"{format_elev(r['elev_gain_m']):>10} "
                f"{format_elev(r['elev_loss_m']):>10}"
            )
        else:
            table_lines.append(
                f"  {idx:<4} {r['from']:<25} {r['to']:<25} {r['length_m']/1000:>10.1f}"
            )
    table_lines += ["=" * col_w, f"  Итого маршрутов: {len(routes)}"]
    if not has_any_elevation:
        table_lines.append("  (Набор/спуск высоты: нет Z-координат в исходных данных)")
    table_text = "\n".join(table_lines)
    print("\n" + table_text)

    route_color_map = {route['from_id']: ROUTE_COLORS[ri % len(ROUTE_COLORS)]
                       for ri, route in enumerate(routes)}

    seg_to_routes = defaultdict(list)
    route_raw_segs = {}
    for ri, route in enumerate(routes):
        path = route['path']; segs = []
        for k in range(len(path)-1):
            u, v = path[k], path[k+1]; d = G[u][v]
            if 'geometry' in d and d['geometry']:
                coords = d['geometry']
                for j in range(len(coords)-1):
                    key = (snap_coord(tuple(coords[j])), snap_coord(tuple(coords[j+1])))
                    segs.append(key)
                    if ri not in seg_to_routes[key]: seg_to_routes[key].append(ri)
            else:
                p1 = (G.nodes[u]['x'], G.nodes[u]['y']); p2 = (G.nodes[v]['x'], G.nodes[v]['y'])
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
                off_m = (i - (n - 1) / 2) * ROUTE_OFFSET_M
                p1_off, p2_off = offset_segment(p1, p2, off_m)
                segs_by_color[color].append([p1_off, p2_off])

    # ================================================================
    # 8. ВИЗУАЛИЗАЦИЯ
    # ================================================================
    iter_label = f" | Итерация {iteration}" if iteration > 1 else ""
    print(f"\n[8/8] Визуализация{iter_label}...")

    fig, ax = plt.subplots(figsize=(144, 108))

    if BBOX is not None:
        pad_lat = 10_000 / DEG_TO_M
        pad_lon = 10_000 / (DEG_TO_M * COS_LAT)
        ax.set_xlim(BBOX_W - pad_lon, BBOX_E + pad_lon)
        ax.set_ylim(BBOX_S - pad_lat, BBOX_N + pad_lat)
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
            ax.plot(st['lon'], st['lat'], '^', color='red', markersize=18,
                    markeredgecolor='#8B0000', markeredgewidth=1.0, zorder=5)
            ax.annotate(st['name'], (st['lon'], st['lat']), fontsize=14, fontweight='bold',
                        xytext=(6,8), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='#cccccc'), zorder=6)
        else:
            ax.plot(st['lon'], st['lat'], 'o', color='#555555', markersize=6, zorder=5)
            ax.annotate(st['name'], (st['lon'], st['lat']), fontsize=10,
                        xytext=(5,5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'), zorder=6)

    legend_handles = []
    for route in routes:
        color = route_color_map.get(route['from_id'], '#E53935')
        elev_str = ''
        if route['elev_gain_m'] is not None:
            elev_str = f"  +{route['elev_gain_m']} м / -{route['elev_loss_m']} м"
        label = f"{route['from']} \u2014 {route['to']}  {route['length_m']/1000:.1f} км{elev_str}"
        legend_handles.append(mlines.Line2D([], [], color=color, linewidth=2.5, label=label))
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=10,
                  framealpha=0.85, edgecolor='#cccccc', title='Маршруты', title_fontsize=12)

    if BBOX is None:
        ax.autoscale(); ax.set_aspect('equal')

    ax.set_xlabel('Долгота', fontsize=12); ax.set_ylabel('Широта', fontsize=12)
    title_bbox = (f" | Ш {BBOX_S}°–{BBOX_N}°, Д {BBOX_W}°–{BBOX_E}°" if BBOX else "")
    ax.set_title(
        f'Маршруты от {n_sel} станций{title_bbox}{iter_label}\n'
        f'Станций: {len(stations)} | Маршрутов: {len(routes)} | {MIN_ROUTE_KM}–{MAX_ROUTE_KM} км',
        fontsize=14)
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.show()

    # ================================================================
    # 9. СОХРАНЕНИЕ
    # ================================================================
    save = input("\nСохранить результаты? (да/нет): ").strip().lower()
    if save in ('да','yes','y','д'):
        SAVE_DIR = input("Путь к папке для сохранения: ").strip().strip('"').strip("'")
        if not os.path.isdir(SAVE_DIR):
            raise FileNotFoundError("Папка не найдена: " + SAVE_DIR)
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_dir = os.path.join(SAVE_DIR, ts); os.makedirs(out_dir, exist_ok=True)

        # PNG
        png_path = os.path.join(out_dir, f'routes_{ts}.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"  PNG: {png_path}")

        # TXT
        txt_path = os.path.join(out_dir, f'routes_{ts}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"tripdata: {TRIPDATA}\n")
            if BBOX:
                f.write(f"Область: Ш {BBOX_S}°–{BBOX_N}°, Д {BBOX_W}°–{BBOX_E}°\n")
            f.write(f"Маршруты: {MIN_ROUTE_KM}–{MAX_ROUTE_KM} км\n\n")
            f.write(table_text + "\n")
        print(f"  TXT: {txt_path}")

        def route_coords(route):
            path = route['path']; coords = []
            for k in range(len(path)-1):
                u, v = path[k], path[k+1]; d = G[u][v]
                ec = d['geometry'] if ('geometry' in d and d['geometry']) else \
                     [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])]
                coords.extend(ec if not coords else ec[1:])
            return [[c[0], c[1]] for c in coords]

        # GeoJSON сводный
        features_out = []
        for cat, feats in bg_classified.items():
            for feat in feats:
                fc_copy = dict(feat)
                if 'properties' not in fc_copy: fc_copy['properties'] = {}
                fc_copy['properties']['_bg_category'] = cat
                features_out.append(fc_copy)
        for st in stations:
            features_out.append({'type':'Feature',
                                 'geometry':{'type':'Point','coordinates':[st['lon'],st['lat']]},
                                 'properties':{'name':st['name'],'type':'station',
                                               'snap_dist_m':st['snap_dist_m'],
                                               'selected':st['node_id'] in sel_ids}})
        for route in routes:
            props = {'type':'route','from':route['from'],'to':route['to'],
                     'length_m':round(route['length_m'],2)}
            if route['elev_gain_m'] is not None:
                props['elev_gain_m'] = route['elev_gain_m']
                props['elev_loss_m'] = route['elev_loss_m']
            features_out.append({'type':'Feature',
                                 'geometry':{'type':'LineString','coordinates':route_coords(route)},
                                 'properties': props})
        for u, v, d in G.edges(data=True):
            if d.get('road_type') == 'station_link': continue
            coords = [[lon,lat] for lon,lat in d['geometry']] if ('geometry' in d and d['geometry']) else \
                     [[G.nodes[u]['x'],G.nodes[u]['y']],[G.nodes[v]['x'],G.nodes[v]['y']]]
            features_out.append({'type':'Feature',
                                 'geometry':{'type':'LineString','coordinates':coords},
                                 'properties':{'type':'road','length_m':round(d.get('length_m',0),2),
                                               'road_type':d.get('road_type','')}})
        bbox_meta = {'bbox':{'N':BBOX_N,'S':BBOX_S,'W':BBOX_W,'E':BBOX_E}} if BBOX else {}
        geojson = {'type':'FeatureCollection',
                   'metadata':{'tripdata':TRIPDATA,'iteration':iteration,
                                'total_stations':len(stations),'selected_stations':n_sel,
                                'routes_count':len(routes),'min_route_km':MIN_ROUTE_KM,
                                'max_route_km':MAX_ROUTE_KM,'graph_nodes':G.number_of_nodes(),
                                'graph_edges':G.number_of_edges(),
                                'graph_components':nx.number_connected_components(G),**bbox_meta},
                   'features':features_out}
        gj_path = os.path.join(out_dir, f'routes_{ts}.geojson')
        with open(gj_path,'w',encoding='utf-8') as f: json.dump(geojson, f, ensure_ascii=False)
        sz = os.path.getsize(gj_path)/(1024*1024)
        print(f"  GeoJSON сводный: {gj_path}  ({sz:.1f} МБ)")

        # GeoJSON по маршруту
        routes_dir = os.path.join(out_dir, 'routes')
        os.makedirs(routes_dir, exist_ok=True)
        for route in routes:
            st_from = next((s for s in stations if s['node_id'] == route['from_id']), None)
            st_to   = next((s for s in stations if s['node_id'] == route['to_id']),   None)
            r_props = {'from':route['from'],'to':route['to'],
                       'length_m':round(route['length_m'],2),
                       'length_km':round(route['length_m']/1000,2)}
            if route['elev_gain_m'] is not None:
                r_props['elev_gain_m'] = route['elev_gain_m']
                r_props['elev_loss_m'] = route['elev_loss_m']
            r_features = [
                {'type':'Feature',
                 'geometry':{'type':'LineString','coordinates':route_coords(route)},
                 'properties': r_props}
            ]
            if st_from:
                r_features.append({'type':'Feature',
                                   'geometry':{'type':'Point','coordinates':[st_from['lon'],st_from['lat']]},
                                   'properties':{'name':route['from'],'role':'start'}})
            if st_to:
                r_features.append({'type':'Feature',
                                   'geometry':{'type':'Point','coordinates':[st_to['lon'],st_to['lat']]},
                                   'properties':{'name':route['to'],'role':'finish'}})
            r_gj = {'type':'FeatureCollection','features':r_features}
            fn = f"{safe_filename(route['from'])}_до_{safe_filename(route['to'])}.geojson"
            with open(os.path.join(routes_dir, fn),'w',encoding='utf-8') as f:
                json.dump(r_gj, f, ensure_ascii=False)
            print(f"    Маршрут: routes/{fn}")

        print(f"\n  Сохранено в: {out_dir}")
    else:
        print("  Сохранение пропущено.")

    again = input("\nПровести ещё один поиск с новыми станциями? (да/нет): ").strip().lower()
    if again not in ('да','yes','y','д'):
        print("Работа завершена."); break

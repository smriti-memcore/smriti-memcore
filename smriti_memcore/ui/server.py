"""
Smriti Memory Browser — Built-in HTTP Server.

Zero external dependencies: only uses Python stdlib http.server + json.
Reads directly from palace.json on disk — no LLM / Ollama needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

# ── The entire UI is embedded here — no template files needed ──────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Smriti — Memory Browser</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    :root {
      --bg:#050810;--surface:#0d1117;--surface2:#161b27;--border:#1e2a3a;
      --accent:#6366f1;--accent2:#8b5cf6;--accent3:#06b6d4;
      --gold:#f59e0b;--green:#10b981;--red:#ef4444;
      --text:#e2e8f0;--muted:#64748b;
      --font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;
    }
    *{margin:0;padding:0;box-sizing:border-box;}
    body{background:var(--bg);color:var(--text);font-family:var(--font);height:100vh;overflow:hidden;display:flex;flex-direction:column;}

    /* Header */
    header{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;border-bottom:1px solid var(--border);background:rgba(5,8,16,0.97);flex-shrink:0;gap:16px;}
    .logo{display:flex;align-items:center;gap:10px;}
    .logo-icon{width:34px;height:34px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;}
    .logo-title{font-size:17px;font-weight:800;background:linear-gradient(90deg,#e2e8f0,var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .logo-sub{font-size:10px;color:var(--muted);font-family:var(--mono);}
    .hstats{display:flex;gap:20px;}
    .hstat{text-align:center;}
    .hstat-val{font-size:19px;font-weight:800;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .hstat-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
    .hcontrols{display:flex;gap:8px;align-items:center;}
    .cbtn{background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:6px 13px;border-radius:7px;font-size:11px;cursor:pointer;font-family:var(--font);transition:all .2s;}
    .cbtn:hover{border-color:var(--accent);color:var(--accent);}
    .cbtn.active{background:var(--accent);border-color:var(--accent);color:#fff;}

    /* Tabs */
    .tabs{display:flex;border-bottom:1px solid var(--border);background:var(--surface);flex-shrink:0;}
    .tab{padding:11px 22px;font-size:12px;font-weight:600;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;user-select:none;}
    .tab:hover{color:var(--text);}
    .tab.active{color:var(--accent);border-bottom-color:var(--accent);}

    /* Main layout */
    .main{display:flex;flex:1;overflow:hidden;}

    /* Graph view */
    #view-graph{flex:1;display:flex;overflow:hidden;}
    #graph-container{flex:1;position:relative;overflow:hidden;}
    svg{width:100%;height:100%;}
    .tooltip{position:absolute;background:rgba(13,17,23,0.92);backdrop-filter:blur(8px);border:1px solid var(--border);border-radius:9px;padding:10px 13px;font-size:12px;pointer-events:none;max-width:260px;line-height:1.5;z-index:100;opacity:0;transition:opacity .15s;box-shadow:0 12px 36px rgba(0,0,0,.6);}
    .tooltip.vis{opacity:1;}
    .t-room{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}

    /* Node and Link Aesthetics matching Smriti Webapp */
    .node {
      transition: r 0.2s, stroke-width 0.2s, opacity 0.2s;
      cursor: pointer;
    }
    .node:hover circle {
      stroke: #ffffff !important;
      stroke-width: 2px !important;
    }
    .node.room-node circle {
      filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.55));
    }
    .node.active circle {
      stroke: #ffffff !important;
      stroke-width: 2.5px !important;
      filter: drop-shadow(0 0 12px rgba(255, 255, 255, 0.7)) !important;
    }
    .node-label {
      font-size: 11px;
      font-weight: 600;
      fill: #cbd5e1;
      pointer-events: none;
      user-select: none;
      text-shadow: 0 1px 4px rgba(0,0,0,0.85);
    }

    /* Curved Links */
    .link {
      stroke-width: 1.4px;
      stroke-opacity: 0.6;
      opacity: 0.45;
      transition: stroke-width 0.2s, opacity 0.2s, stroke 0.2s;
      pointer-events: none;
    }
    .link.active {
      stroke-width: 2.5px !important;
      stroke-opacity: 1 !important;
      opacity: 0.95 !important;
      filter: drop-shadow(0 0 6px rgba(255, 255, 255, 0.4));
    }

    /* Neural Impulse Particles */
    .particle-dot {
      pointer-events: none;
      transition: r 0.25s, filter 0.25s, fill 0.25s, opacity 0.25s;
    }
    .particle-dot.active {
      r: 4.5px !important;
      fill: #ffffff !important;
      filter: drop-shadow(0 0 5px #ffffff) drop-shadow(0 0 10px rgba(255, 255, 255, 0.7)) !important;
    }

    /* Focus dimming when a node is hovered or active */
    svg.has-active .node:not(.active) {
      opacity: 0.32;
    }
    svg.has-active .link:not(.active) {
      opacity: 0.10;
    }
    svg.has-active .particle-dot:not(.active) {
      opacity: 0.12;
    }

    /* Sidebar shared */
    .sidebar{width:300px;border-left:1px solid var(--border);background:var(--surface);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0;}
    .sh{padding:12px 16px;border-bottom:1px solid var(--border);font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;flex-shrink:0;}

    /* Node detail */
    .node-detail{padding:16px;border-bottom:1px solid var(--border);min-height:120px;max-height:280px;overflow-y:auto;display:flex;flex-direction:column;gap:9px;flex-shrink:0;}
    .node-detail::-webkit-scrollbar{width:3px;}
    .node-detail::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
    .node-detail.empty{align-items:center;justify-content:center;color:var(--muted);font-size:13px;text-align:center;}
    .detail-room{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;width:fit-content;}
    .detail-content{font-size:12px;line-height:1.6;}
    .dmetrics{display:grid;grid-template-columns:1fr 1fr;gap:7px;}
    .dmetric{background:var(--surface2);border:1px solid var(--border);border-radius:7px;padding:7px 11px;}
    .dmetric-val{font-size:17px;font-weight:700;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .dmetric-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;}
    .bar-wrap{} .bar-label{font-size:9px;color:var(--muted);margin-bottom:3px;}
    .bar-track{background:var(--surface2);border-radius:3px;height:6px;overflow:hidden;}
    .bar-fill{height:100%;border-radius:3px;transition:width .5s ease;background:linear-gradient(90deg,var(--accent),var(--accent2));}

    /* Legend */
    .legend{padding:12px 16px;border-bottom:1px solid var(--border);max-height:160px;overflow-y:auto;flex-shrink:0;}
    .legend::-webkit-scrollbar{width:3px;}
    .legend::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
    .legend-title{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;font-weight:600;}
    .legend-item{display:flex;align-items:center;gap:7px;font-size:11px;margin-bottom:5px;}
    .ldot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}

    /* Memory list */
    .mem-list{flex:1;overflow-y:auto;padding:6px 0;}
    .mem-list::-webkit-scrollbar{width:3px;}
    .mem-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
    .mem-item{padding:9px 16px;cursor:pointer;transition:background .13s;border-left:3px solid transparent;}
    .mem-item:hover{background:var(--surface2);}
    .mem-item.sel{background:var(--surface2);border-left-color:var(--accent);}
    .mic{font-size:11px;color:var(--text);line-height:1.5;}
    .mim{font-size:10px;color:var(--muted);margin-top:2px;font-family:var(--mono);}

    /* Table view */
    #view-table{flex:1;overflow:hidden;display:none;flex-direction:column;}
    .tbl-toolbar{padding:12px 20px;border-bottom:1px solid var(--border);display:flex;gap:10px;align-items:center;flex-shrink:0;}
    .search-input{flex:1;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:7px 12px;border-radius:8px;font-size:12px;font-family:var(--font);outline:none;}
    .search-input:focus{border-color:var(--accent);}
    .search-input::placeholder{color:var(--muted);}
    .tbl-wrap{flex:1;overflow:auto;}
    table{width:100%;border-collapse:collapse;}
    th{padding:10px 16px;text-align:left;font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;border-bottom:1px solid var(--border);background:var(--surface);position:sticky;top:0;cursor:pointer;user-select:none;}
    th:hover{color:var(--text);}
    td{padding:10px 16px;font-size:12px;border-bottom:1px solid var(--border);vertical-align:top;}
    tr:hover td{background:var(--surface2);}
    .room-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;}
    .status-ok{color:var(--green);} .status-pin{color:var(--gold);}
    .status-pending{background:rgba(245,158,11,.15);color:var(--gold);padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600;}

    /* Stats view */
    #view-stats{flex:1;overflow:auto;display:none;padding:24px;}
    .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px;}
    .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;}
    .sc-val{font-size:28px;font-weight:800;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .sc-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:4px;}
    .sc-sub{font-size:12px;color:var(--muted);margin-top:6px;}
    .room-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;}
    .room-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;}
    .rc-title{font-size:13px;font-weight:700;margin-bottom:10px;}
    .rc-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;}
    .rc-label{font-size:10px;color:var(--muted);width:55px;}
    .rc-track{flex:1;background:var(--surface2);height:5px;border-radius:3px;overflow:hidden;}
    .rc-fill{height:100%;border-radius:3px;}
    .section-title{font-size:13px;font-weight:700;margin-bottom:12px;color:var(--text);}
  </style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🧠</div>
    <div>
      <div class="logo-title">Smriti Memory Browser</div>
      <div class="logo-sub" id="storage-path">Loading...</div>
    </div>
  </div>
  <div class="hstats">
    <div class="hstat"><div class="hstat-val" id="hs-mems">–</div><div class="hstat-label">Memories</div></div>
    <div class="hstat"><div class="hstat-val" id="hs-conns">–</div><div class="hstat-label">Connections</div></div>
    <div class="hstat"><div class="hstat-val" id="hs-rooms">–</div><div class="hstat-label">Rooms</div></div>
    <div class="hstat"><div class="hstat-val" id="hs-str">–</div><div class="hstat-label">Avg Strength</div></div>
    <div class="hstat"><div class="hstat-val" id="hs-pending">0</div><div class="hstat-label">In Queue</div></div>
  </div>
  <div class="hcontrols">
    <button class="cbtn active" id="btn-force" onclick="setLayout('force')">Force</button>
    <button class="cbtn" id="btn-cluster" onclick="setLayout('cluster')">Cluster</button>
    <button class="cbtn" onclick="resetZoom()">Reset</button>
    <button class="cbtn" onclick="refreshData()" title="Reload from disk">↻ Refresh</button>
  </div>
</header>

<div class="tabs">
  <div class="tab active" id="tab-graph" onclick="showTab('graph')">🏛️ Semantic Palace</div>
  <div class="tab" id="tab-table" onclick="showTab('table')">📋 Memory Table</div>
  <div class="tab" id="tab-stats" onclick="showTab('stats')">📊 Statistics</div>
  <div class="tab" id="tab-episodes" onclick="showTab('episodes')">📼 Episode Feed <span id="episodes-badge" style="display:none;background:var(--gold);color:#000;border-radius:10px;padding:1px 6px;font-size:9px;margin-left:4px;font-weight:bold;">N</span></div>
</div>

<div id="consolidation-banner" style="display:none; background: rgba(245,158,11,0.08); border-bottom: 1px solid rgba(245,158,11,0.25); padding: 10px 24px; font-size: 11.5px; align-items: center; justify-content: space-between; color: var(--gold); transition: all 0.3s ease; flex-shrink: 0;">
  <div style="display: flex; align-items: center; gap: 8px;">
    <span>⚡</span>
    <span><strong><span id="banner-count">0</span> new memories</strong> are in the queue for consolidation. Run <code>smriti_consolidate</code> in your terminal or wait for background processing.</span>
  </div>
  <button onclick="dismissBanner()" style="background: none; border: none; color: var(--muted); cursor: pointer; font-size: 14px; font-weight: bold; line-height: 1; padding: 0 4px;">&times;</button>
</div>

<div class="main">
  <!-- GRAPH VIEW -->
  <div id="view-graph">
    <div id="graph-container">
      <div class="tooltip" id="tooltip"></div>
      <svg id="svg"></svg>
    </div>
    <div class="sidebar">
      <div class="sh">Memory Detail</div>
      <div class="node-detail empty" id="node-detail">
        <div>👆 Click a node to inspect</div>
      </div>
      <div class="legend">
        <div class="legend-title">Node Types</div>
        <div style="display:flex;flex-direction:column;gap:6px;margin-bottom:12px;font-size:11px;">
          <div style="display:flex;align-items:center;gap:7px;">
            <div style="width:11px;height:11px;border-radius:50%;background:#6366f1;box-shadow:0 0 8px rgba(99,102,241,0.6);flex-shrink:0;"></div>
            <span style="font-weight:600;color:#cbd5e1;">🏛️ Category Room</span>
          </div>
          <div style="display:flex;align-items:center;gap:7px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#06b6d4;box-shadow:0 0 6px rgba(6,182,212,0.6);margin-left:1.5px;flex-shrink:0;"></div>
            <span style="color:#cbd5e1;">🧠 Shared Memory</span>
          </div>
          <div style="display:flex;align-items:center;gap:7px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#f59e0b;box-shadow:0 0 6px rgba(245,158,11,0.6);margin-left:1.5px;flex-shrink:0;"></div>
            <span style="color:#cbd5e1;">🔒 Private Memory</span>
          </div>
        </div>
        <div class="legend-title">Topics</div>
        <div id="legend-items"></div>
      </div>
      <div class="sh" style="border-top:1px solid var(--border)">All Memories</div>
      <div class="mem-list" id="mem-list"></div>
    </div>
  </div>

  <!-- TABLE VIEW -->
  <div id="view-table">
    <div class="tbl-toolbar">
      <input class="search-input" id="search-input" type="text" placeholder="Search memories..." oninput="filterTable()"/>
      <select class="search-input" id="room-filter" onchange="filterTable()" style="max-width:140px">
        <option value="">All Rooms</option>
      </select>
    </div>
    <div class="tbl-wrap">
      <table id="mem-table">
        <thead>
          <tr>
            <th onclick="sortTable('content')">Content ↕</th>
            <th onclick="sortTable('room')">Room ↕</th>
            <th onclick="sortTable('strength')">Strength ↕</th>
            <th onclick="sortTable('salience')">Salience ↕</th>
            <th onclick="sortTable('status')">Status ↕</th>
            <th onclick="sortTable('access_count')">Accessed ↕</th>
          </tr>
        </thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
  </div>

  <!-- STATS VIEW -->
  <div id="view-stats">
    <div id="stats-content"></div>
  </div>

  <!-- EPISODE FEED VIEW -->
  <div id="view-episodes" style="display:none;flex:1;overflow:hidden;flex-direction:column;">
    <div class="tbl-wrap">
      <table id="episodes-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Content</th>
            <th>Source</th>
            <th>Salience</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="episodes-body">
          <tr><td colspan="5" style="color:var(--muted);text-align:center">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
const COLORS = {
  identity:     '#6366f1', architecture: '#06b6d4',
  release:      '#10b981', medical:      '#ef4444',
  tech:         '#f59e0b', work:         '#f59e0b',
  general:      '#8b5cf6',
};
const TEXT_COLORS = {
  identity:'#818cf8', architecture:'#22d3ee', release:'#34d399',
  medical:'#f87171', tech:'#fbbf24', work:'#fbbf24', general:'#a78bfa',
};
function rc(room){return COLORS[room]||'#8b5cf6';}
function rtc(room){return TEXT_COLORS[room]||'#a78bfa';}

let graphData = {nodes:[],edges:[]};
let simulation, svg, g, linkSel, nodeSel, particleSel;
let sortKey = 'strength', sortAsc = false;

function dismissBanner() {
  document.getElementById('consolidation-banner').style.display = 'none';
}

async function checkPendingMemories() {
  try {
    const res = await fetch('/api/episodes');
    if (!res.ok) return;
    const episodes = await res.json();
    const pendingCount = episodes.filter(ep => !ep.consolidated).length;
    
    // Update Pending Stat in Header
    const pendingValEl = document.getElementById('hs-pending');
    if (pendingValEl) {
      pendingValEl.textContent = pendingCount;
      if (pendingCount > 0) {
        pendingValEl.style.color = 'var(--gold)';
        pendingValEl.style.textShadow = '0 0 8px rgba(245,158,11,0.5)';
      } else {
        pendingValEl.style.color = 'var(--text)';
        pendingValEl.style.textShadow = 'none';
      }
    }
    
    // Update tab badge
    const badgeEl = document.getElementById('episodes-badge');
    if (badgeEl) {
      if (pendingCount > 0) {
        badgeEl.textContent = pendingCount;
        badgeEl.style.display = 'inline-block';
      } else {
        badgeEl.style.display = 'none';
      }
    }
    
    // Update Banner
    const bannerEl = document.getElementById('consolidation-banner');
    if (bannerEl) {
      if (pendingCount > 0) {
        document.getElementById('banner-count').textContent = pendingCount;
        bannerEl.style.display = 'flex';
      } else {
        bannerEl.style.display = 'none';
      }
    }
  } catch (err) {
    console.error('Failed to check pending memories:', err);
  }
}

async function refreshData(){
  episodesLoaded = false;
  const res = await fetch('/api/graph');
  const data = await res.json();
  graphData = data;
  const memNodes = data.nodes.filter(n => n.type === 'memory' || !n.type);
  updateHeader(data, memNodes);
  buildGraph(data);
  buildTable(memNodes);
  buildStats(data, memNodes);
  buildLegend(data, memNodes);
  buildMemList(memNodes);
  
  await checkPendingMemories();
}

function updateHeader(data, memNodes){
  document.getElementById('storage-path').textContent = data.storage_path || '';
  const list = memNodes || (data.nodes ? data.nodes.filter(n => n.type === 'memory' || !n.type) : []);
  document.getElementById('hs-mems').textContent = data.stats?.total_memories ?? list.length;
  document.getElementById('hs-conns').textContent = data.edges.length;
  document.getElementById('hs-rooms').textContent = data.room_count ?? (data.stats?.total_rooms ?? [...new Set(list.map(n=>n.room))].length);
  const avg = list.length ? (list.reduce((s,n)=>s+(n.strength||0),0)/list.length).toFixed(2) : '0';
  document.getElementById('hs-str').textContent = avg;
}

// ── Graph ───────────────────────────────────────────────────────
function buildGraph(data){
  const container = document.getElementById('graph-container');
  const W = container.clientWidth || 800, H = container.clientHeight || 600;

  svg = d3.select('#svg');
  svg.selectAll('*').remove();

  const defs = svg.append('defs');
  const p = defs.append('pattern').attr('id','grid').attr('width',40).attr('height',40).attr('patternUnits','userSpaceOnUse');
  p.append('path').attr('d','M 40 0 L 0 0 0 40').attr('fill','none').attr('stroke','rgba(30,42,58,0.4)').attr('stroke-width',.5);
  svg.append('rect').attr('width','100%').attr('height','100%').attr('fill','url(#grid)');

  const flt = defs.append('filter').attr('id','glow');
  flt.append('feGaussianBlur').attr('stdDeviation','3.5').attr('result','cb');
  const fm = flt.append('feMerge');
  fm.append('feMergeNode').attr('in','cb');
  fm.append('feMergeNode').attr('in','SourceGraphic');

  const zoom = d3.zoom().scaleExtent([.15,5]).on('zoom',e=>g.attr('transform',e.transform));
  svg.call(zoom);
  window._zoom=zoom; window._svg=svg;

  g = svg.append('g');

  const nodes = data.nodes.map(n=>({
    ...n,
    x: W/2 + (Math.random() - .5) * 360,
    y: H/2 + (Math.random() - .5) * 280
  }));
  const nodeById = Object.fromEntries(nodes.map(n=>[n.id,n]));
  const links = data.edges.map(e=>({
    source: nodeById[e.from],
    target: nodeById[e.to],
    weight: e.weight || .6,
    type: e.type || 'link',
    t: Math.random(),
    speed: 0.003 + Math.random() * 0.004,
    is_active: false
  })).filter(e=>e.source&&e.target);

  // Fallback: connect memories in same room if no edges exist
  if(!links.length){
    const roomMap = {};
    nodes.filter(n=>n.type!=='room').forEach(n=>{(roomMap[n.room]=roomMap[n.room]||[]).push(n);});
    Object.values(roomMap).forEach(mems=>{
      for(let i=0;i<mems.length;i++)for(let j=i+1;j<mems.length;j++)
        links.push({
          source: mems[i],
          target: mems[j],
          weight: .5,
          t: Math.random(),
          speed: 0.003 + Math.random() * 0.004,
          is_active: false
        });
    });
  }

  // Draw curved links (<path>)
  linkSel = g.append('g').attr('class','links').selectAll('path')
    .data(links)
    .join('path')
    .attr('class','link')
    .attr('stroke', d => d.source.color || (d.source.type==='room' ? '#6366f1' : 'var(--accent3)'))
    .attr('fill','none');

  // Draw neural impulse particles (synapses transmitting impulses)
  particleSel = g.append('g').attr('class','particles').selectAll('circle')
    .data(links)
    .join('circle')
    .attr('class','particle-dot')
    .attr('r', 2.5)
    .attr('fill', d => d.source.color || (d.source.type==='room' ? '#6366f1' : 'var(--accent3)'));

  // Draw Nodes
  nodeSel = g.append('g').attr('class','nodes').selectAll('.node')
    .data(nodes)
    .join('g')
    .attr('class', d => d.type === 'room' ? 'node room-node' : 'node memory-node')
    .call(d3.drag()
      .on('start',(e,d)=>{if(!e.active)simulation.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;})
      .on('drag', (e,d)=>{d.fx=e.x;d.fy=e.y;})
      .on('end',  (e,d)=>{if(!e.active)simulation.alphaTarget(0);d.fx=null;d.fy=null;}))
    .on('click',(e,d)=>{e.stopPropagation();selectNode(d);})
    .on('mouseover',(e,d)=>showTip(e,d))
    .on('mousemove',e=>moveTip(e))
    .on('mouseout',hideTip);

  svg.on('click',()=>clearSel());

  // Room nodes: indigo #6366f1 hub with glow
  const roomNodes = nodeSel.filter(d => d.type === 'room');
  roomNodes.append('circle')
    .attr('r', 16)
    .attr('fill', '#6366f1')
    .attr('stroke', '#090d16')
    .attr('stroke-width', 1.5)
    .attr('filter', 'url(#glow)');

  roomNodes.append('text')
    .attr('class', 'node-label')
    .attr('dx', 22)
    .attr('dy', 5)
    .text(d => '🏛️ ' + d.label);

  // Memory satellite nodes: cyan #06b6d4 (shared) or gold #f59e0b (private)
  const memNodesSel = nodeSel.filter(d => d.type !== 'room');
  memNodesSel.append('circle')
    .attr('r', d => 6 + Math.min((d.strength || 1) * 2.5, 6))
    .attr('fill', d => d.color || '#06b6d4')
    .attr('stroke', '#090d16')
    .attr('stroke-width', 1.5);

  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d=>d.id).distance(d => d.target.type === 'room' || d.source.type === 'room' ? 110 : 55).strength(0.5))
    .force('charge', d3.forceManyBody().strength(-320))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collision', d3.forceCollide(d => d.type === 'room' ? 44 : 18))
    .on('tick', () => {
      // Curved quadratic Bézier lines
      linkSel.attr('d', d => {
        const x1 = d.source.x, y1 = d.source.y;
        const x2 = d.target.x, y2 = d.target.y;
        const dx = x2 - x1, dy = y2 - y1;
        const dr = Math.sqrt(dx * dx + dy * dy);
        if (dr === 0) return `M ${x1} ${y1} L ${x2} ${y2}`;
        const curveFactor = 0.18;
        const cx = (x1 + x2) / 2 - (dy / dr) * (dr * curveFactor);
        const cy = (y1 + y2) / 2 + (dx / dr) * (dr * curveFactor);
        return `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
      });
      nodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
    });

  // Continuous particle animation along curved synapses
  if (window._particleTimer) window._particleTimer.stop();
  window._particleTimer = d3.timer(() => {
    if (!particleSel) return;
    particleSel.each(function(d) {
      const x1 = d.source.x, y1 = d.source.y;
      const x2 = d.target.x, y2 = d.target.y;
      if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) return;
      const dx = x2 - x1, dy = y2 - y1;
      const dr = Math.sqrt(dx * dx + dy * dy);
      if (dr === 0) {
        d3.select(this).attr('cx', x1).attr('cy', y1);
        return;
      }
      const curveFactor = 0.18;
      const cx = (x1 + x2) / 2 - (dy / dr) * (dr * curveFactor);
      const cy = (y1 + y2) / 2 + (dx / dr) * (dr * curveFactor);

      d.t = (d.t + (d.is_active ? 0.015 : d.speed)) % 1;
      const t = d.t;
      const px = (1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * cx + t * t * x2;
      const py = (1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * cy + t * t * y2;
      d3.select(this).attr('cx', px).attr('cy', py);
    });
  });

  window._nodes=nodes; window._links=links; window._sim=simulation;
  setTimeout(() => autoFit(true), 500);
}

function setLayout(mode){
  ['force','cluster'].forEach(m=>document.getElementById('btn-'+m).classList.toggle('active',m===mode));
  if(!window._sim||!window._nodes)return;
  const W=document.getElementById('graph-container').clientWidth || 800;
  const H=document.getElementById('graph-container').clientHeight || 600;
  if(mode==='cluster'){
    const roomNodes = window._nodes.filter(n=>n.type==='room');
    const rooms = roomNodes.map(r=>r.room_id || r.id.replace('room:',''));
    const radius = Math.min(W, H) * 0.32;
    const centers = Object.fromEntries(rooms.map((r,i)=>[r,{
      x: W/2 + Math.cos(i*2*Math.PI/rooms.length) * radius,
      y: H/2 + Math.sin(i*2*Math.PI/rooms.length) * radius
    }]));
    window._sim
      .force('x', d3.forceX(d => centers[d.room_id]?.x || W/2).strength(0.35))
      .force('y', d3.forceY(d => centers[d.room_id]?.y || H/2).strength(0.35))
      .force('charge', d3.forceManyBody().strength(-120))
      .alphaTarget(0.3).restart();
    setTimeout(()=>window._sim.alphaTarget(0), 1800);
  } else {
    window._sim
      .force('x', null)
      .force('y', null)
      .force('charge', d3.forceManyBody().strength(-320))
      .force('center', d3.forceCenter(W/2, H/2))
      .alphaTarget(0.3).restart();
    setTimeout(()=>window._sim.alphaTarget(0), 1800);
  }
}

function autoFit(animate = true){
  if(!window._nodes || !window._nodes.length || !window._svg || !window._zoom) return;
  const container = document.getElementById('graph-container');
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;
  const xs = window._nodes.map(n => n.x || 0);
  const ys = window._nodes.map(n => n.y || 0);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 60;
  const graphWidth = (maxX - minX) + pad * 2;
  const graphHeight = (maxY - minY) + pad * 2;
  if (graphWidth <= 0 || graphHeight <= 0) return;
  let scale = Math.min(width / graphWidth, height / graphHeight);
  scale = Math.min(Math.max(scale, 0.3), 1.4);
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  const dx = (width / 2) - scale * midX;
  const dy = (height / 2) - scale * midY;
  const transform = d3.zoomIdentity.translate(dx, dy).scale(scale);
  if (animate) {
    window._svg.transition().duration(600).call(window._zoom.transform, transform);
  } else {
    window._svg.call(window._zoom.transform, transform);
  }
}

function resetZoom(){ autoFit(true); }

function selectNode(d){
  window._selectedId = d.id;
  document.querySelectorAll('.mem-item').forEach(el=>el.classList.remove('sel'));
  const rawId = d.raw_id || d.id;
  const li = document.getElementById('li-' + rawId) || document.getElementById('li-' + d.id);
  if(li){ li.classList.add('sel'); li.scrollIntoView({behavior:'smooth',block:'nearest'}); }

  const det = document.getElementById('node-detail');
  det.className = 'node-detail';

  if (d.type === 'room') {
    det.innerHTML = `
      <div class="detail-room" style="background:rgba(99,102,241,0.15);color:#818cf8;border:1px solid rgba(99,102,241,0.3)">🏛️ CATEGORY ROOM</div>
      <div style="font-size:15px;font-weight:700;color:var(--text);margin-top:2px">${d.label}</div>
      <div class="dmetrics" style="margin-top:6px">
        <div class="dmetric"><div class="dmetric-val">${d.memory_count || 0}</div><div class="dmetric-label">Memories</div></div>
        <div class="dmetric"><div class="dmetric-val">${d.visit_count || 0}</div><div class="dmetric-label">Visits</div></div>
      </div>
      <div style="font-size:10.5px;color:var(--muted);font-family:var(--mono);margin-top:4px">
        Room ID: ${d.room_id || d.id}<br>Visibility: <span style="text-transform:uppercase;color:#818cf8">${d.visibility || 'shared'}</span>
      </div>
    `;
  } else {
    const c = rc(d.room), tc = rtc(d.room);
    const visBadgeColor = d.visibility === 'private' ? 'var(--gold)' : 'var(--accent3)';
    const level = d.reflection_level || 0;
    const levelNames = ["Direct", "Observation", "Insight", "Principle"];
    const stateName = levelNames[level] || "Direct";
    const stateColor = level === 0 ? "#22d3ee" : level === 1 ? "#c084fc" : level === 2 ? "#f472b6" : "#f59e0b";

    det.innerHTML = `
      <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
        <div class="detail-room" style="background:${c}22;color:${tc};border:1px solid ${c}44">🏛️ ${d.room}</div>
        <span style="font-size:9.5px;font-weight:700;padding:2px 8px;border-radius:12px;background:rgba(255,255,255,0.05);color:${visBadgeColor};border:1px solid ${visBadgeColor}44;text-transform:uppercase;">${d.visibility || 'shared'}</span>
        <span style="font-size:9.5px;font-weight:700;padding:2px 8px;border-radius:12px;background:rgba(255,255,255,0.05);color:${stateColor};border:1px solid ${stateColor}44;text-transform:uppercase;">${stateName} (L${level})</span>
      </div>
      <div class="detail-content" style="margin-top:4px">${d.content}</div>
      <div class="dmetrics">
        <div class="dmetric"><div class="dmetric-val">${((d.strength||0)*100).toFixed(0)}%</div><div class="dmetric-label">Strength</div></div>
        <div class="dmetric"><div class="dmetric-val">${(d.salience||0).toFixed(3)}</div><div class="dmetric-label">Salience</div></div>
      </div>
      <div class="bar-wrap"><div class="bar-label">Memory Retention</div><div class="bar-track"><div class="bar-fill" style="width:${Math.min((d.strength||0)*100, 100)}%"></div></div></div>
      <div style="font-size:10px;color:var(--muted);font-family:var(--mono)">
        ID: ${d.raw_id || d.id}<br>Status: <span style="color:${d.status==='active'?'var(--green)':'var(--gold)'}">${d.status}</span> · Accessed: ${d.access_count||0}×
      </div>`;
  }

  svg.classed('has-active', true);
  nodeSel?.classed('active', n => n.id === d.id);
  linkSel?.classed('active', l => {
    const isActive = (l.source.id === d.id || l.target.id === d.id);
    l.is_active = isActive;
    return isActive;
  });
  particleSel?.classed('active', l => l.is_active);
}

function selectById(id){
  const n = window._nodes?.find(n => n.id === id || n.raw_id === id || n.id === 'mem:' + id || n.id === 'room:' + id);
  if(n) selectNode(n);
}

function clearSel(){
  window._selectedId = null;
  svg?.classed('has-active', false);
  nodeSel?.classed('active', false);
  linkSel?.classed('active', false).each(l => { l.is_active = false; });
  particleSel?.classed('active', false);
  document.querySelectorAll('.mem-item').forEach(el=>el.classList.remove('sel'));
  const det = document.getElementById('node-detail');
  det.className = 'node-detail empty';
  det.innerHTML = '<div>👆 Click a node to inspect</div>';
}

function showTip(e,d){
  const t = document.getElementById('tooltip');
  if (d.type === 'room') {
    t.innerHTML = `
      <div class="t-room" style="color:#818cf8">🏛️ CATEGORY ROOM</div>
      <div style="font-size:13px;font-weight:700;margin-bottom:4px">${d.label}</div>
      <div style="font-size:10px;color:var(--muted)">${d.memory_count || 0} clustered memories · ${d.visit_count || 0} visits</div>
    `;
  } else {
    const strength = ((d.strength || 0) * 100).toFixed(0);
    const level = d.reflection_level || 0;
    const levelNames = ["Direct", "Observation", "Insight", "Principle"];
    const stateName = levelNames[level] || "Direct";
    const visColor = d.visibility === 'private' ? 'var(--gold)' : 'var(--accent3)';
    t.innerHTML = `
      <div class="t-room" style="color:${rtc(d.room)}">🧠 MEMORY NODE · <span style="color:${visColor}">${(d.visibility||'shared').toUpperCase()}</span></div>
      <div style="margin-bottom:5px;font-size:12px;line-height:1.4">${d.content.length > 110 ? d.content.slice(0, 110) + '…' : d.content}</div>
      <div style="font-size:10px;color:var(--muted)">Strength: <b style="color:var(--text)">${strength}%</b> · State: <b style="color:var(--text)">${stateName} (L${level})</b></div>
    `;
  }
  t.classList.add('vis');
  moveTip(e);

  svg.classed('has-active', true);
  nodeSel?.classed('active', n => n.id === d.id);
  linkSel?.classed('active', l => {
    const isActive = (l.source.id === d.id || l.target.id === d.id);
    l.is_active = isActive;
    return isActive;
  });
  particleSel?.classed('active', l => l.is_active);
}

function moveTip(e){
  const t = document.getElementById('tooltip'), r = document.getElementById('graph-container').getBoundingClientRect();
  let x = e.clientX - r.left + 14, y = e.clientY - r.top - 8;
  if(x + 260 > r.width) x = e.clientX - r.left - 270;
  t.style.left = x + 'px'; t.style.top = y + 'px';
}

function hideTip(){
  document.getElementById('tooltip').classList.remove('vis');
  if (!window._selectedId) {
    svg?.classed('has-active', false);
    nodeSel?.classed('active', false);
    linkSel?.classed('active', false).each(l => { l.is_active = false; });
    particleSel?.classed('active', false);
  } else {
    const selNode = window._nodes?.find(n => n.id === window._selectedId);
    if (selNode) {
      nodeSel?.classed('active', n => n.id === selNode.id);
      linkSel?.classed('active', l => {
        const isActive = (l.source.id === selNode.id || l.target.id === selNode.id);
        l.is_active = isActive;
        return isActive;
      });
      particleSel?.classed('active', l => l.is_active);
    }
  }
}

// ── Legend + mem list ───────────────────────────────────────────
function buildLegend(data, memNodes){
  const list = memNodes || (data.nodes ? data.nodes.filter(n => n.type === 'memory' || !n.type) : []);
  const rooms = [...new Set(list.map(n=>n.room))];
  document.getElementById('legend-items').innerHTML = rooms.map(r=>`
    <div class="legend-item"><div class="ldot" style="background:${rc(r)}"></div><span>${r.charAt(0).toUpperCase()+r.slice(1)}</span><span style="color:var(--muted);font-size:10px;margin-left:auto">${list.filter(n=>n.room===r).length} mems</span></div>`).join('');
}

function buildMemList(memNodes){
  document.getElementById('mem-list').innerHTML = memNodes.map(n=>`
    <div class="mem-item" id="li-${n.raw_id || n.id}" onclick="selectById('${n.id}')">
      <div class="mic">${n.content.length > 80 ? n.content.slice(0, 80) + '…' : n.content}</div>
      <div class="mim">
        <span style="color:${n.visibility==='private'?'var(--gold)':'var(--accent3)'}">● ${n.visibility.toUpperCase()}</span> · 
        <span style="color:${rtc(n.room)}">${n.room}</span> · str ${(n.strength||0).toFixed(2)}
      </div>
    </div>`).join('');
}

// ── Table view ──────────────────────────────────────────────────
let _allMemories = [];
function buildTable(nodes){
  _allMemories = nodes;
  const rf = document.getElementById('room-filter');
  const rooms = [...new Set(nodes.map(n=>n.room))];
  rf.innerHTML = '<option value="">All Rooms</option>' + rooms.map(r=>`<option value="${r}">${r}</option>`).join('');
  renderTable(nodes);
}

function filterTable(){
  const q = document.getElementById('search-input').value.toLowerCase();
  const r = document.getElementById('room-filter').value;
  let filtered = _allMemories.filter(n=>{
    const matchQ = !q || (n.content && n.content.toLowerCase().includes(q)) || (n.room && n.room.toLowerCase().includes(q));
    const matchR = !r || n.room === r;
    return matchQ && matchR;
  });
  renderTable(filtered);
}

function sortTable(key){
  if(sortKey===key) sortAsc = !sortAsc;
  else { sortKey = key; sortAsc = false; }
  filterTable();
}

function renderTable(nodes){
  const sorted = [...nodes].sort((a,b)=>{
    let av = a[sortKey], bv = b[sortKey];
    if(typeof av === 'string') av = av.toLowerCase(), bv = bv?.toLowerCase();
    return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  });
  document.getElementById('table-body').innerHTML = sorted.map(n=>`
    <tr onclick="showTab('graph');setTimeout(()=>selectById('${n.id}'),200)">
      <td style="max-width:340px">${n.content}</td>
      <td><span class="room-badge" style="background:${rc(n.room)}22;color:${rtc(n.room)}">${n.room}</span></td>
      <td>${(n.strength||0).toFixed(3)}</td>
      <td>${(n.salience||0).toFixed(3)}</td>
      <td><span class="${n.status==='active'?'status-ok':'status-pin'}">${n.status}</span></td>
      <td>${n.access_count||0}</td>
    </tr>`).join('');
}

// ── Stats view ──────────────────────────────────────────────────
function buildStats(data, memNodes){
  const nodes = memNodes || (data.nodes ? data.nodes.filter(n => n.type === 'memory' || !n.type) : []);
  if(!nodes.length){document.getElementById('stats-content').innerHTML='<p style="color:var(--muted)">No memories yet.</p>';return;}
  const rooms = [...new Set(nodes.map(n=>n.room))];
  const avgStr = (nodes.reduce((s,n)=>s+(n.strength||0),0)/nodes.length).toFixed(3);
  const avgSal = (nodes.reduce((s,n)=>s+(n.salience||0),0)/nodes.length).toFixed(3);
  const pinned = nodes.filter(n=>n.status==='pinned').length;
  const archived = nodes.filter(n=>n.status==='archived').length;
  const privCount = nodes.filter(n=>n.visibility==='private').length;
  const sharedCount = nodes.filter(n=>n.visibility!=='private').length;
  const totalAccess = nodes.reduce((s,n)=>s+(n.access_count||0),0);

  const roomStats = rooms.map(r=>{
    const mems = nodes.filter(n=>n.room===r);
    const avg = (mems.reduce((s,n)=>s+(n.strength||0),0)/mems.length).toFixed(2);
    return {room:r, count:mems.length, avg, color:rc(r)};
  });

  document.getElementById('stats-content').innerHTML = `
    <div class="stats-grid">
      <div class="stat-card"><div class="sc-val">${nodes.length}</div><div class="sc-label">Total Memories</div></div>
      <div class="stat-card"><div class="sc-val">${data.edges.length}</div><div class="sc-label">Synaptic Links</div></div>
      <div class="stat-card"><div class="sc-val">${data.room_count || rooms.length}</div><div class="sc-label">Category Rooms</div></div>
      <div class="stat-card"><div class="sc-val">${avgStr}</div><div class="sc-label">Avg Memory Strength</div></div>
      <div class="stat-card"><div class="sc-val">${avgSal}</div><div class="sc-label">Avg Salience Score</div></div>
      <div class="stat-card"><div class="sc-val">${pinned}</div><div class="sc-label">Pinned Memories</div><div class="sc-sub">Never forgotten</div></div>
      <div class="stat-card"><div class="sc-val">${privCount} / ${sharedCount}</div><div class="sc-label">Private / Shared</div><div class="sc-sub">Vault segregation</div></div>
      <div class="stat-card"><div class="sc-val">${totalAccess}</div><div class="sc-label">Total Recalls</div></div>
    </div>
    <div class="section-title">Rooms Breakdown</div>
    <div class="room-cards">
      ${roomStats.map(rs=>`
        <div class="room-card" style="border-color:${rs.color}33">
          <div class="rc-title" style="color:${rs.color}">${rs.room.charAt(0).toUpperCase()+rs.room.slice(1)}</div>
          <div class="rc-bar-row"><span class="rc-label">Memories</span><div class="rc-track"><div class="rc-fill" style="width:${(rs.count/nodes.length*100).toFixed(0)}%;background:${rs.color}"></div></div><span style="font-size:10px;color:var(--muted)">${rs.count}</span></div>
          <div class="rc-bar-row"><span class="rc-label">Avg Str</span><div class="rc-track"><div class="rc-fill" style="width:${(rs.avg*100).toFixed(0)}%;background:${rs.color}88"></div></div><span style="font-size:10px;color:var(--muted)">${rs.avg}</span></div>
        </div>`).join('')}
    </div>
  `;
}

// ── Tabs ────────────────────────────────────────────────────────
function showTab(tab){
  ['graph','table','stats','episodes'].forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active',t===tab);
    document.getElementById('view-'+t).style.display=t===tab?'flex':'none';
  });
  if(tab==='graph'&&window._nodes){
    setTimeout(()=>autoFit(false), 50);
  }
  if(tab==='episodes'&&!episodesLoaded){buildEpisodes();}
}

// ── Episode Feed ────────────────────────────────────────────────
let episodesLoaded = false;

async function buildEpisodes(){
  try {
    const res = await fetch('/api/episodes');
    if(!res.ok) throw new Error('HTTP ' + res.status);
    const episodes = await res.json();
    episodesLoaded = true;
    const tbody = document.getElementById('episodes-body');
    if(!episodes.length){
      tbody.innerHTML='<tr><td colspan="5" style="color:var(--muted);text-align:center">No episodes found.</td></tr>';
      return;
    }
    tbody.innerHTML = episodes.map(ep=>`
      <tr>
        <td style="white-space:nowrap;font-family:var(--mono);font-size:11px">${ep.timestamp.replace('T',' ').slice(0,16)}</td>
        <td>${ep.content}</td>
        <td>${ep.source}</td>
        <td>${ep.salience.toFixed(3)}</td>
        <td><span class="${ep.consolidated?'status-ok':'status-pending'}">${ep.consolidated?'✓ consolidated':'⏳ pending'}</span></td>
      </tr>`).join('');
  } catch(err) {
    console.error('buildEpisodes failed:', err);
    document.getElementById('episodes-body').innerHTML=
      '<tr><td colspan="5" style="color:var(--red)">Error loading episodes.</td></tr>';
  }
}

// ── Boot ────────────────────────────────────────────────────────
refreshData().catch(console.error);
</script>
</body>
</html>
"""


# ── Data reader — reads palace.json directly, no embeddings loaded ─────────────

def _read_palace(storage_path: str) -> dict:
    """Parse palace.json and return graph-ready dict with category room hubs and memory satellites."""
    palace_file = Path(storage_path).expanduser() / "palace" / "palace.json"
    if not palace_file.exists():
        return {
            "nodes": [],
            "edges": [],
            "memories": [],
            "rooms": [],
            "storage_path": str(palace_file),
            "room_count": 0,
            "stats": {"total_memories": 0, "total_rooms": 0},
        }

    with open(palace_file) as f:
        raw = json.load(f)

    raw_memories = raw.get("memories", {})
    raw_rooms = raw.get("rooms", {})

    room_by_id = {}
    room_list = []
    for rid, r in raw_rooms.items():
        r_id = r.get("id", rid) if isinstance(r, dict) else rid
        topic = r.get("topic", rid) if isinstance(r, dict) else str(r)
        room_by_id[r_id] = topic
        room_list.append({
            "id": r_id,
            "topic": topic,
            "visibility": r.get("visibility", "shared") if isinstance(r, dict) else "shared",
            "memory_ids": r.get("memory_ids", []) if isinstance(r, dict) else [],
            "visit_count": r.get("visit_count", 0) if isinstance(r, dict) else 0,
        })

    # If no rooms exist but memories do, synthesize a General room
    if not room_list and raw_memories:
        room_list.append({
            "id": "general",
            "topic": "General",
            "visibility": "shared",
            "memory_ids": list(raw_memories.keys()),
            "visit_count": 1,
        })
        room_by_id["general"] = "General"

    nodes = []
    # 1. Add Room Nodes (Hubs) — indigo #6366f1
    for r in room_list:
        nodes.append({
            "id": f"room:{r['id']}",
            "room_id": r["id"],
            "label": r["topic"],
            "topic": r["topic"],
            "type": "room",
            "val": 16,
            "color": "#6366f1",
            "visibility": r.get("visibility", "shared"),
            "memory_count": len(r.get("memory_ids", [])),
            "visit_count": r.get("visit_count", 0),
        })

    # 2. Add Memory Nodes (Satellites) — cyan #06b6d4 (shared) or gold #f59e0b (private)
    mem_list = []
    for mem_id, mem in raw_memories.items():
        mid = mem.get("id", mem_id)
        sal = mem.get("salience", {})
        composite = sal.get("composite", 0.0) if isinstance(sal, dict) else 0.0
        room_id = mem.get("room_id", "")
        room_topic = room_by_id.get(room_id, "General")
        visibility = mem.get("visibility", "shared")
        reflection_level = mem.get("reflection_level", 0)
        strength = round(mem.get("strength", 1.0), 4)

        # In Smriti Webapp: private = #f59e0b (amber/gold), shared = #06b6d4 (cyan)
        node_color = "#f59e0b" if visibility == "private" else "#06b6d4"

        mem_dict = {
            "id": f"mem:{mid}",
            "raw_id": mid,
            "content": mem.get("content", ""),
            "label": (mem.get("content", "")[:35] + "...") if len(mem.get("content", "")) > 35 else mem.get("content", ""),
            "room": room_topic,
            "room_id": room_id or (room_list[0]["id"] if room_list else "general"),
            "type": "memory",
            "strength": strength,
            "salience": round(composite, 4),
            "status": mem.get("status", "active"),
            "visibility": visibility,
            "reflection_level": reflection_level,
            "color": node_color,
            "val": 6 + round(strength * 2.5, 1),
            "access_count": mem.get("access_count", 0),
            "created": mem.get("creation_time", ""),
            "last_accessed": mem.get("last_accessed", ""),
            "associations": mem.get("associations", []),
        }
        nodes.append(mem_dict)
        mem_list.append(mem_dict)

    # 3. Add Edges (curved synapses)
    edges = []
    room_node_ids = {n["id"] for n in nodes if n["type"] == "room"}
    mem_node_ids = {n["id"] for n in nodes if n["type"] == "memory"}

    # Connect each memory to its parent category room
    for m in mem_list:
        target_room = f"room:{m['room_id']}"
        if target_room in room_node_ids:
            edges.append({
                "from": m["id"],
                "to": target_room,
                "weight": 0.85,
                "type": "room_member",
            })
        elif room_node_ids:
            fallback = next(iter(room_node_ids))
            edges.append({
                "from": m["id"],
                "to": fallback,
                "weight": 0.5,
                "type": "room_member",
            })

        # Connect memory associations
        for assoc_id in m.get("associations", []):
            target_mem = f"mem:{assoc_id}"
            if target_mem in mem_node_ids:
                edges.append({
                    "from": m["id"],
                    "to": target_mem,
                    "weight": 0.65,
                    "type": "association",
                })

    # Add explicit palace edges (e.g. room-to-room bridges or cross edges)
    for edge in raw.get("edges", []):
        src = edge.get("source") or edge.get("from")
        tgt = edge.get("target") or edge.get("to")
        if not src or not tgt:
            continue
        src_id = f"room:{src}" if f"room:{src}" in room_node_ids else (f"mem:{src}" if f"mem:{src}" in mem_node_ids else src)
        tgt_id = f"room:{tgt}" if f"room:{tgt}" in room_node_ids else (f"mem:{tgt}" if f"mem:{tgt}" in mem_node_ids else tgt)
        edges.append({
            "from": src_id,
            "to": tgt_id,
            "weight": edge.get("weight", edge.get("strength", 0.7)),
            "relationship": edge.get("relationship", "cross"),
            "type": "cross",
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "memories": mem_list,
        "rooms": room_list,
        "storage_path": str(palace_file),
        "room_count": len(room_list),
        "stats": {
            "total_memories": len(mem_list),
            "total_rooms": len(room_list),
        },
    }


def _read_episodes(storage_path: str) -> list:
    """Read episodes.db and return a list of episode dicts, newest first.

    fetchall() materialises all rows into a Python list before conn.close(),
    so the for-loop runs safely after the finally block.
    """
    db_file = Path(storage_path).expanduser() / "episodes" / "episodes.db"
    if not db_file.exists():
        return []

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row  # enables named column access: row["salience_json"] not row[N]
    try:
        rows = conn.execute(
            "SELECT id, content, timestamp, source, salience_json, consolidated "
            "FROM episodes ORDER BY timestamp DESC"
        ).fetchall()  # fetchall() returns a plain list — safe to use after conn.close()
    finally:
        conn.close()

    episodes = []
    for row in rows:
        try:
            salience = json.loads(row["salience_json"] or "{}").get("composite", 0.0)
        except (json.JSONDecodeError, TypeError):
            salience = 0.0
        episodes.append({
            "id":           row["id"],
            "content":      row["content"],
            "timestamp":    row["timestamp"],
            "source":       row["source"],
            "salience":     salience,
            "consolidated": bool(row["consolidated"]),  # SQLite stores 0/1 integers
        })
    return episodes


# ── HTTP Handler ────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    storage_path: str = ""

    def log_message(self, format, *args):
        pass  # Silence default access logs

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._respond(200, "text/html", _HTML.encode())

        elif path == "/api/graph":
            data = _read_palace(self.storage_path)
            body = json.dumps(data).encode()
            self._respond(200, "application/json", body)

        elif path == "/api/episodes":
            data = _read_episodes(self.storage_path)
            body = json.dumps(data).encode()
            self._respond(200, "application/json", body)

        elif path == "/api/health":
            body = json.dumps({"status": "ok", "storage": self.storage_path}).encode()
            self._respond(200, "application/json", body)

        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ── Public API ──────────────────────────────────────────────────────────────────

def launch(
    storage_path: str = "~/.smriti/global",
    port: int = 7799,
    open_browser: bool = True,
    blocking: bool = True,
) -> Optional[HTTPServer]:
    """
    Launch the Smriti Memory Browser UI.

    Args:
        storage_path: Path to the Smriti storage directory.
        port:         Local port to serve on (default: 7799).
        open_browser: Automatically open in the default browser.
        blocking:     If True, blocks until Ctrl+C. If False, runs in background thread.

    Returns:
        The HTTPServer instance (only when blocking=False).

    Example::

        from smriti_memcore.ui import launch
        launch(storage_path="~/.smriti/global")
    """
    resolved = str(Path(storage_path).expanduser())

    # Patch storage path into handler class
    class Handler(_Handler):
        pass
    Handler.storage_path = resolved

    server = HTTPServer(("127.0.0.1", port), Handler)

    url = f"http://127.0.0.1:{port}"
    print(f"\n🏛️  Smriti Memory Browser")
    print(f"   Storage : {resolved}")
    print(f"   URL     : {url}")
    print(f"   Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    if blocking:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑  Smriti Browser stopped.")
        finally:
            server.server_close()
        return None
    else:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server

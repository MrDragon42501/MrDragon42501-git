/* =========================================================
 * SAM3 水体分割监控平台 - 前端逻辑
 *
 * 与云服务器对接的接口约定（按需修改 CONFIG 中的路径）：
 *
 *   实时画面（WebSocket，服务器主动推送）：
 *     WS  {server}/ws/live?site=xx
 *         <- 每条二进制消息为一帧 JPEG 图片，25 帧/秒
 *
 *   站点参数：
 *     所有请求均携带 ?site=xx（边坡1 / 边坡2 / 边坡3），用于指导服务器分割
 *
 *   检测控制（HTTP + JSON）：
 *     GET  {server}/api/status?site=xx  -> { "task_running": bool }
 *     POST {server}/api/detect?site=xx
 *          body: { "timestamp": 1723627383000,   // 毫秒时间戳
 *                  "mode": "continuous" | "single" }  // 工作模式
 *     GET  {server}/api/result?site=xx  -> 未就绪: { "ready": false }
 *                                          就绪: {
 *                                            "ready": true,
 *                                            "avg_elevation": 12.35,        // 数字：平均水位高程
 *                                            "sample_start_time": "..."或毫秒时间戳,
 *                                            "result_video_url": "/xx.mp4"  // MP4：分割结果
 *                                          }
 * ========================================================= */

const CONFIG = {
  wsLivePath: '/ws/live',          // WebSocket：实时画面帧推送
  apiStatus: '/api/status',
  apiDetect: '/api/detect',
  apiStop: '/api/stop',            // 立即中断当前检测任务
  apiResult: '/api/result',
  apiSites: '/api/sites',
  apiSitesUpload: '/api/sites/upload',
  apiSitesRename: '/api/sites/rename',
  apiSitesDelete: '/api/sites/delete',
  apiSettings: '/api/settings',
  apiContinuousSettings: '/api/continuous_settings',
  apiDownloadAll: '/api/download_all',

  pollIntervalMs: 2000,            // 等待结果时的轮询间隔
  liveReconnectMs: 5000,           // 实时画面断线重连间隔

  /* 站点列表及其服务器地址：选择站点后自动填入地址；留空则需手动填写 */
  sites: {
    '边坡1': '',
    '边坡2': '',
    '边坡3': '',
  },

  historyMaxPoints: 10000,                 // 历史数据最多保留的点数
  chartDefaultSpanMs: 10 * 60 * 1000,      // 折线图默认显示最近 10 分钟
  chartMinSpanMs: 30 * 1000,               // 最小 30 秒
  chartMaxSpanMs: 24 * 3600 * 1000,        // 最大 24 小时
};

/* ---------- DOM ---------- */
const $ = (id) => document.getElementById(id);
const els = {
  siteSelect: $('siteSelect'),
  serverAddr: $('serverAddr'), btnConnect: $('btnConnect'),
  connDot: $('connDot'), connText: $('connText'),
  liveCanvas: $('liveCanvas'), livePlaceholder: $('livePlaceholder'),
  resultVideo: $('resultVideo'), resultPlaceholder: $('resultPlaceholder'),
  avgElevation: $('avgElevation'), sampleStartTime: $('sampleStartTime'),
  demoBadge: $('demoBadge'),
  btnContinuous: $('btnContinuous'), btnSingle: $('btnSingle'), btnStop: $('btnStop'),
  btnSettings: $('btnSettings'),
  taskDot: $('taskDot'), taskText: $('taskText'),
  levelChart: $('levelChart'),
  gaugeScale: $('gaugeScale'), gaugeWaterTint: $('gaugeWaterTint'),
  gaugeWaterLine: $('gaugeWaterLine'), gaugeRange: $('gaugeRange'),
  gaugeInput: $('gaugeInput'),
  settingsPage: $('settingsPage'), btnSettingsClose: $('btnSettingsClose'),
  csvFile: $('csvFile'), btnUploadCsv: $('btnUploadCsv'), siteList: $('siteList'),
  tabSites: $('tabSites'), tabOutput: $('tabOutput'), tabServer: $('tabServer'), tabContinuous: $('tabContinuous'),
  panelSites: $('panelSites'), panelOutput: $('panelOutput'), panelServer: $('panelServer'), panelContinuous: $('panelContinuous'),
  btnToggleDownload: $('btnToggleDownload'), btnPickDir: $('btnPickDir'),
  downloadDirName: $('downloadDirName'), btnSaveOutput: $('btnSaveOutput'),
  serverAddrSetting: $('serverAddrSetting'), btnSaveServer: $('btnSaveServer'),
  intervalSeconds: $('intervalSeconds'), detectCount: $('detectCount'), btnSaveContinuous: $('btnSaveContinuous'),
};

/* ---------- 状态 ---------- */
const state = {
  server: '',            // 云服务器地址
  site: '',              // 当前站点
  connected: false,
  mode: 'idle',          // idle | continuous | single
  waiting: false,        // 是否正在等待服务器返回检测结果
  resultUrl: '',         // 当前已加载的分割结果视频地址（避免重复加载）
  demo: true,            // 未连接服务器时进入演示模式，便于预览页面效果
  pollTimer: null,
  demoTimer: null,
  hls: null,             // 结果视频 HLS 实例（仅当地址为 .m3u8 时使用）
  liveWs: null,          // 实时画面 WebSocket
  liveWsRetryTimer: null,
  liveDrawing: false,    // 正在绘制一帧（用于丢帧防堆积）
  outputSettings: { download_enabled: false, download_path: '' },  // 输出文件设置
  downloadDirHandle: null,     // 用户选择的本地目录句柄（File System Access API）
  downloadDirName: '',         // 所选目录名
};

/* =========================================================
 * 历史水位折线图（Canvas，拖动平移 / 滚轮缩放）
 * ========================================================= */
const chart = {
  data: [],                        // [{t: 毫秒时间戳, v: 高程}]
  spanMs: CONFIG.chartDefaultSpanMs,
  endMs: null,                     // 视图右端；null = 跟随最新数据
  canvas: null, ctx: null,
  W: 0, H: 0,                      // CSS 像素尺寸
  M: { l: 48, r: 14, t: 12, b: 24 },   // 边距

  init(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();

    /* ---- 拖动平移 ---- */
    let drag = null;
    canvas.addEventListener('pointerdown', (e) => {
      drag = { x0: e.clientX, end0: this.viewEnd() };
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = 'grabbing';
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const plotW = this.W - this.M.l - this.M.r;
      if (plotW <= 0) return;
      const dt = (e.clientX - drag.x0) / plotW * this.spanMs;
      const latest = this.latestT();
      let end = drag.end0 - dt;            // 向右拖 -> 查看更早的数据
      if (end >= latest - this.spanMs * 0.02) {
        this.endMs = null;                 // 拖回右端 -> 恢复跟随最新
      } else {
        const first = this.data.length ? this.data[0].t : end;
        this.endMs = Math.max(end, first + 1000);
      }
      this.draw();
    });
    const endDrag = () => { drag = null; canvas.style.cursor = 'grab'; };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    /* ---- 滚轮缩放（以光标位置为锚点） ---- */
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const plotW = this.W - this.M.l - this.M.r;
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left - this.M.l) / plotW));
      const end = this.viewEnd(), start = end - this.spanMs;
      const anchor = start + frac * this.spanMs;

      const factor = e.deltaY > 0 ? 1.25 : 0.8;
      this.spanMs = Math.max(CONFIG.chartMinSpanMs,
                     Math.min(CONFIG.chartMaxSpanMs, this.spanMs * factor));
      const newEnd = anchor + (1 - frac) * this.spanMs;
      this.endMs = newEnd >= this.latestT() - this.spanMs * 0.02 ? null : newEnd;
      this.draw();
    }, { passive: false });
  },

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = w; this.H = h;
    this.draw();
  },

  latestT() {
    return this.data.length ? this.data[this.data.length - 1].t : Date.now();
  },
  viewEnd() {
    return this.endMs == null ? this.latestT() : this.endMs;
  },

  push(t, v) {
    if (v == null || isNaN(v)) return;
    this.data.push({ t, v });
    if (this.data.length > CONFIG.historyMaxPoints) this.data.shift();
    this.draw();
  },

  fmtAxisTime(t, stepMs) {
    const d = new Date(t);
    const p = (n) => String(n).padStart(2, '0');
    if (stepMs < 60 * 1000) return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
    if (stepMs < 24 * 3600 * 1000) return `${p(d.getHours())}:${p(d.getMinutes())}`;
    return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  },

  draw() {
    const { ctx, W, H, M } = this;
    if (!ctx || W <= 0) return;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px "PingFang SC", "Microsoft YaHei", sans-serif';

    const plotW = W - M.l - M.r;
    const plotH = H - M.t - M.b;
    const end = this.viewEnd();
    const start = end - this.spanMs;

    /* ---- Y 轴范围：取可见数据的最值并留边 ---- */
    let lo = Infinity, hi = -Infinity;
    for (const p of this.data) {
      if (p.t < start - this.spanMs || p.t > end) continue;
      if (p.v < lo) lo = p.v;
      if (p.v > hi) hi = p.v;
    }
    if (lo > hi) { lo = 0; hi = 10; }
    const pad = Math.max((hi - lo) * 0.15, 0.05);
    lo -= pad; hi += pad;

    const x = (t) => M.l + (t - start) / this.spanMs * plotW;
    const y = (v) => M.t + (1 - (v - lo) / (hi - lo)) * plotH;

    /* ---- 横向网格线 + Y 轴标签（取整步长） ---- */
    const rawStep = (hi - lo) / 4;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    const step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
    const decimals = step < 0.1 ? 2 : step < 1 ? 1 : 0;

    ctx.strokeStyle = '#eef2f7';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      const yy = y(v);
      ctx.beginPath(); ctx.moveTo(M.l, yy); ctx.lineTo(W - M.r, yy); ctx.stroke();
      ctx.fillText(v.toFixed(decimals), M.l - 8, yy);
    }

    /* ---- X 轴时间标签：按整步长取刻度，避免标签重复 ---- */
    const TIME_STEPS = [
      10e3, 30e3, 60e3, 2 * 60e3, 5 * 60e3, 10 * 60e3, 30 * 60e3,
      3600e3, 2 * 3600e3, 6 * 3600e3, 12 * 3600e3, 24 * 3600e3,
    ];
    const maxTicks = Math.max(2, Math.floor(plotW / 110));
    let tStep = TIME_STEPS[TIME_STEPS.length - 1];
    for (const s of TIME_STEPS) {
      if (this.spanMs / s <= maxTicks) { tStep = s; break; }
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let t = Math.ceil(start / tStep) * tStep; t <= end; t += tStep) {
      ctx.fillText(this.fmtAxisTime(t, tStep), x(t), H - M.b + 6);
    }

    if (!this.data.length) {
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('暂无历史数据', M.l + plotW / 2, M.t + plotH / 2);
      return;
    }

    /* ---- 折线 + 区域填充（裁剪到绘图区） ---- */
    ctx.save();
    ctx.beginPath();
    ctx.rect(M.l, M.t, plotW, plotH);
    ctx.clip();

    ctx.beginPath();
    let started = false;
    for (const p of this.data) {
      const px = x(p.t), py = y(p.v);
      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.lineTo(x(this.data[this.data.length - 1].t), M.t + plotH);
    ctx.lineTo(x(this.data[0].t), M.t + plotH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(37, 99, 235, .08)';
    ctx.fill();
    ctx.restore();

    /* ---- 最新数据点标记 ---- */
    const last = this.data[this.data.length - 1];
    if (last.t >= start && last.t <= end) {
      const px = x(last.t), py = y(last.v);
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#2563eb';
      ctx.fill();
      ctx.fillStyle = '#1f2933';
      ctx.textAlign = px > M.l + plotW - 60 ? 'right' : 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`${last.v.toFixed(2)} m`, px + (px > M.l + plotW - 60 ? -6 : 6), py - 4);
    }
  },
};

/* =========================================================
 * 检测结果展示
 * ========================================================= */

/** 进入 / 退出"等待服务器返回结果"状态 */
function setWaiting(on) {
  state.waiting = on;
  els.avgElevation.parentElement.classList.toggle('pending', on);
  els.sampleStartTime.classList.toggle('pending', on);
  if (on) {
    els.avgElevation.textContent = '等待结果…';
    els.sampleStartTime.textContent = '等待结果…';
    /* 清空旧的分割结果视频 */
    state.resultUrl = '';
    els.resultVideo.removeAttribute('src');
    els.resultVideo.load();
    els.resultPlaceholder.textContent = '等待分割结果…';
    els.resultPlaceholder.classList.remove('hidden');
  }
}

/** 展示服务器返回的一项完整检测结果 */
function showResult(d) {
  state.waiting = false;
  els.avgElevation.parentElement.classList.remove('pending');
  els.sampleStartTime.classList.remove('pending');

  /* 平均水位高程（数字）；无边坡时 avg_elevation 为 null，显示 -- */
  const elev = (d.avg_elevation == null) ? NaN : Number(d.avg_elevation);
  els.avgElevation.textContent = isNaN(elev) ? '--' : elev.toFixed(2);
  if (!isNaN(elev)) chart.push(Date.now(), elev);

  /* 虚拟水尺：上下限 + 水位 */
  if (d.ruler_min != null && d.ruler_max != null) {
    gauge.setRange(d.ruler_min, d.ruler_max);
  } else {
    gauge.reset();  // 无边坡时清空水尺
  }
  if (!isNaN(elev)) gauge.update(elev);

  /* 采样开始时间（时间戳，兼容毫秒数字与字符串） */
  const t = d.sample_start_time;
  els.sampleStartTime.textContent =
    typeof t === 'number' ? fmtTime(new Date(t)) : (t || '--');

  /* 分割结果（MP4 视频地址，仅在变化时重新加载） */
  if (d.result_video_url && d.result_video_url !== state.resultUrl) {
    state.resultUrl = d.result_video_url;
    attachResultVideo(resolveUrl(d.result_video_url));
    maybeDownloadAll(d.result_video_url);
  }
}

/* =========================================================
 * 实时画面：WebSocket 接收 JPEG 帧并绘制到 Canvas
 * ========================================================= */
const live = {
  ctx: null,

  init() {
    this.ctx = els.liveCanvas.getContext('2d');
    new ResizeObserver(() => this.resize()).observe(els.liveCanvas.parentElement);
    this.resize();
  },

  resize() {
    const dpr = window.devicePixelRatio || 1;
    els.liveCanvas.width = els.liveCanvas.clientWidth * dpr;
    els.liveCanvas.height = els.liveCanvas.clientHeight * dpr;
  },

  /** 等比绘制一帧（contain 适配） */
  draw(bitmap) {
    const cw = els.liveCanvas.width, ch = els.liveCanvas.height;
    if (!cw || !ch) { bitmap.close(); return; }
    const scale = Math.min(cw / bitmap.width, ch / bitmap.height);
    const dw = bitmap.width * scale, dh = bitmap.height * scale;
    this.ctx.clearRect(0, 0, cw, ch);
    this.ctx.drawImage(bitmap, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
    bitmap.close();
  },

  connect() {
    this.close();
    const wsUrl = state.server.replace(/^http/, 'ws') + CONFIG.wsLivePath +
                  '?site=' + encodeURIComponent(state.site);
    let ws;
    try { ws = new WebSocket(wsUrl); }
    catch (err) { this.scheduleReconnect(); return; }
    state.liveWs = ws;

    ws.onopen = () => {
      els.livePlaceholder.textContent = '等待实时画面…';
    };
    ws.onmessage = async (e) => {
      /* 上一帧还未绘制完则丢弃本帧，防止 25fps 下解码堆积 */
      if (state.liveDrawing) return;
      state.liveDrawing = true;
      try {
        const blob = e.data instanceof Blob ? e.data : new Blob([e.data]);
        const bitmap = await createImageBitmap(blob);
        this.draw(bitmap);
        els.livePlaceholder.classList.add('hidden');
      } catch (err) {
        /* 无法解码的消息直接忽略 */
      } finally {
        state.liveDrawing = false;
      }
    };
    ws.onclose = () => {
      els.livePlaceholder.textContent = '实时画面连接断开，等待重连…';
      els.livePlaceholder.classList.remove('hidden');
      this.scheduleReconnect();
    };
    ws.onerror = () => ws.close();
  },

  scheduleReconnect() {
    clearTimeout(state.liveWsRetryTimer);
    if (!state.connected || state.demo) return;
    state.liveWsRetryTimer = setTimeout(() => this.connect(), CONFIG.liveReconnectMs);
  },

  close() {
    clearTimeout(state.liveWsRetryTimer);
    if (state.liveWs) {
      state.liveWs.onclose = null;   // 主动关闭不触发重连
      state.liveWs.close();
      state.liveWs = null;
    }
  },
};

/* =========================================================
 * 演示模式（未配置服务器时自动开启，仅用于页面预览）
 * ========================================================= */

/** 演示水位：12m 附近平滑波动，历史与实时共用同一函数保证连续 */
function demoValue(tMs) {
  const s = tMs / 1000;
  return 12 + Math.sin(s / 30) * 0.25 + Math.sin(s / 90 + 1) * 0.15;
}

function startDemo() {
  state.demo = true;
  els.demoBadge.classList.remove('hidden');
  const startAt = new Date();

  /* 预生成过去 2 小时的演示历史数据，便于预览折线图 */
  const now = Date.now();
  for (let t = now - 2 * 3600 * 1000; t <= now; t += 30 * 1000) {
    chart.push(t, demoValue(t));
  }

  showDemo(demoValue(now), fmtTime(startAt));
  state.demoTimer = setInterval(() => {
    showDemo(demoValue(Date.now()), fmtTime(startAt));
  }, 1000);
}

function showDemo(elev, startTime) {
  els.avgElevation.textContent = elev.toFixed(2);
  els.sampleStartTime.textContent = startTime;
  chart.push(Date.now(), elev);
}

function stopDemo() {
  state.demo = false;
  els.demoBadge.classList.add('hidden');
  clearInterval(state.demoTimer);
}

function fmtTime(d) {
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
         `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/* =========================================================
 * 云服务器连接
 * ========================================================= */

function setConn(online, text) {
  state.connected = online;
  els.connDot.className = 'dot ' + (online ? 'online' : 'offline');
  els.connText.textContent = text;
  els.btnSettings.disabled = !online;
}

function setMode(mode) {
  state.mode = mode;
  const busy = mode !== 'idle';
  els.btnContinuous.disabled = busy || !state.connected;
  els.btnSingle.disabled = busy || !state.connected;
  els.btnStop.disabled = !busy;
  els.taskDot.className = 'dot ' + (busy ? 'running' : 'idle');
  els.taskText.textContent =
    mode === 'continuous' ? '连续检测运行中' :
    mode === 'single' ? '单次检测执行中' : '任务空闲';
}

/** 挂载分割结果视频（MP4 原生播放；若为 .m3u8 则走 HLS） */
function attachResultVideo(url) {
  if (state.hls) { state.hls.destroy(); state.hls = null; }
  if (url.split('?')[0].endsWith('.m3u8') && window.Hls && Hls.isSupported()) {
    const hls = new Hls();
    hls.loadSource(url);
    hls.attachMedia(els.resultVideo);
    state.hls = hls;
  } else {
    els.resultVideo.src = url;
  }
  els.resultVideo.play().catch(() => {});
}

/** 服务器相对路径 -> 完整 URL */
function resolveUrl(u) {
  return /^https?:\/\//.test(u) ? u : state.server + u;
}

function bindResultPlaceholder() {
  els.resultVideo.addEventListener('playing', () => els.resultPlaceholder.classList.add('hidden'));
  els.resultVideo.addEventListener('error', () => {
    if (state.waiting) return;   // 等待结果期间不覆盖"等待分割结果"提示
    els.resultPlaceholder.classList.remove('hidden');
  });
}

function apiUrl(path) {
  return state.server + path + '?site=' + encodeURIComponent(state.site);
}

async function connect() {
  const server = els.serverAddr.value.trim().replace(/\/+$/, '');
  if (!server) { alert('请先填写云服务器地址'); return; }
  state.server = server;
  setConn(false, '连接中…');

  try {
    const res = await fetch(apiUrl(CONFIG.apiStatus), { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const info = await res.json();

    stopDemo();
    setConn(true, '已连接');
    setMode(info.task_running ? 'continuous' : 'idle');

    loadSites();              // 拉取边坡列表，同步站点下拉框
    loadOutputSettings();     // 拉取输出文件设置（下载开关、路径）
    live.connect();      // 实时画面：WebSocket 帧推送
    startPolling();      // 分割结果：轮询获取
  } catch (err) {
    console.error(err);
    setConn(false, '连接失败');
  }
}

/* ---------- 轮询检测结果 ---------- */
function startPolling() {
  clearInterval(state.pollTimer);
  pollOnce();
  state.pollTimer = setInterval(pollOnce, CONFIG.pollIntervalMs);
}

function stopPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = null;
}

async function pollOnce() {
  try {
    const res = await fetch(apiUrl(CONFIG.apiResult), { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();

    if (d && d.ready === true) {
      showResult(d);
    }

    if (state.mode === 'single') {
      // 单次检测：拿到结果即完成
      if (d && d.ready === true) {
        setMode('idle');
        stopPolling();
      }
    } else if (state.mode === 'continuous') {
      // 连续检测：检查任务是否已结束（次数完成或中断）
      const sres = await fetch(apiUrl(CONFIG.apiStatus), { cache: 'no-store' });
      if (sres.ok) {
        const s = await sres.json();
        if (s.task_running === false) {
          setMode('idle');
          stopPolling();
        }
      }
    }
  } catch (err) {
    console.warn('结果数据获取失败：', err);
  }
}

/* ---------- 开始连续 / 单次检测 ----------
 * 向服务器发送当前毫秒时间戳 + 工作模式 + 站点参数，
 * 随后平均水位高程 / 分割结果 / 采样开始时间进入"等待结果"状态，
 * 直到轮询到服务器返回的结果为止。 */
async function startDetect(kind) {
  setMode(kind);
  try {
    const res = await fetch(apiUrl(CONFIG.apiDetect), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ timestamp: Date.now(), mode: kind }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    setWaiting(true);
    startPolling();
  } catch (err) {
    alert('启动检测任务失败：' + err.message);
    setMode('idle');
  }
}

async function stopDetect() {
  try {
    await fetch(apiUrl(CONFIG.apiStop), { method: 'POST' });
  } catch (err) {
    console.warn('中断请求失败：', err);
  }
  stopPolling();
  setWaiting(false);
  setMode('idle');
}

/* =========================================================
 * 虚拟水尺（右下角，可调节）
 * ========================================================= */
const gauge = {
  min: null, max: null, level: null,
  svgH: 400,

  elevToY(elev) {
    if (this.min == null || this.max == null || this.max === this.min) return 0;
    const t = (this.max - elev) / (this.max - this.min);
    return Math.max(0, Math.min(1, t)) * this.svgH;
  },

  drawScale() {
    if (this.min == null || this.max == null) return;
    const NS = 'http://www.w3.org/2000/svg';
    const g = els.gaugeScale;
    g.innerHTML = '';
    const add = (tag, attrs) => {
      const n = document.createElementNS(NS, tag);
      for (const k in attrs) n.setAttribute(k, attrs[k]);
      g.appendChild(n);
      return n;
    };
    add('rect', { class: 'strip', x: 0, y: 0, width: 60, height: this.svgH });
    const step = 0.1;
    for (let e = this.min; e <= this.max + 1e-9; e += step) {
      const y = this.elevToY(e);
      const isMajor = Math.abs(e - Math.round(e)) < 1e-6;     // 1m 主刻度
      const isMid = !isMajor && Math.abs(e * 10 % 5) < 1e-6;  // 0.5m 中刻度
      const x1 = isMajor ? 26 : isMid ? 40 : 48;
      add('line', {
        class: isMajor ? 'tick tick-major' : isMid ? 'tick tick-mid' : 'tick',
        x1, y1: y, x2: 60, y2: y,
      });
      if (isMajor) {
        const ty = Math.max(8, Math.min(this.svgH - 8, y));
        const t = add('text', { x: 20, y: ty, 'text-anchor': 'end', 'dominant-baseline': 'central' });
        t.textContent = e.toFixed(1);
      }
    }
  },

  update(level) {
    if (level == null || isNaN(level)) return;
    this.level = level;
    const y = this.elevToY(level);
    els.gaugeWaterLine.setAttribute('y1', y);
    els.gaugeWaterLine.setAttribute('y2', y);
    els.gaugeWaterTint.setAttribute('y', y);
    els.gaugeWaterTint.setAttribute('height', this.svgH - y);
  },

  setRange(min, max) {
    this.min = min;
    this.max = max;
    els.gaugeRange.textContent = `${min.toFixed(2)} ~ ${max.toFixed(2)} m`;
    this.drawScale();
    if (this.level != null) this.update(this.level);
  },

  reset() {
    this.min = null;
    this.max = null;
    this.level = null;
    els.gaugeRange.textContent = '-- ~ -- m';
    els.gaugeScale.innerHTML = '';
    els.gaugeWaterTint.setAttribute('y', '0');
    els.gaugeWaterTint.setAttribute('height', '0');
    els.gaugeWaterLine.setAttribute('y1', '0');
    els.gaugeWaterLine.setAttribute('y2', '0');
  },

  init() {
    els.gaugeInput.addEventListener('change', () => {
      let v = parseFloat(els.gaugeInput.value);
      if (isNaN(v)) return;
      if (this.min != null && this.max != null) {
        v = Math.max(this.min, Math.min(this.max, v));
        els.gaugeInput.value = v;
      }
      this.update(v);
    });
  },
};

/* =========================================================
 * 边坡设置页面
 * ========================================================= */
async function openSettings() {
  els.settingsPage.classList.remove('hidden');
  await Promise.all([loadSites(), loadOutputSettings(), loadContinuousSettings()]);
}

function closeSettings() {
  els.settingsPage.classList.add('hidden');
}

/* 设置页 tab 切换 */
function switchSettingsTab(which) {
  const tabs = {
    sites:      [els.tabSites,      els.panelSites],
    output:     [els.tabOutput,     els.panelOutput],
    server:     [els.tabServer,     els.panelServer],
    continuous: [els.tabContinuous, els.panelContinuous],
  };
  for (const key in tabs) {
    const [tab, panel] = tabs[key];
    const active = key === which;
    tab.classList.toggle('active', active);
    panel.classList.toggle('hidden', !active);
  }
}

/* 连续检测设置 */
async function loadContinuousSettings() {
  try {
    const res = await fetch(apiUrl(CONFIG.apiContinuousSettings), { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();
    els.intervalSeconds.value = d.interval_seconds || 0;
    els.detectCount.value = d.count || 0;
  } catch (err) {
    console.warn('获取连续检测设置失败：', err);
  }
}

async function saveContinuousSettings() {
  const settings = {
    interval_seconds: Math.max(0, parseInt(els.intervalSeconds.value, 10) || 0),
    count: Math.max(0, parseInt(els.detectCount.value, 10) || 0),
  };
  try {
    const res = await fetch(apiUrl(CONFIG.apiContinuousSettings), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    alert('保存成功');
  } catch (err) {
    alert('保存失败：' + err.message);
  }
}

/* 服务器设置：保存地址到 localStorage，下次打开自动连接 */
function saveServerAddr() {
  const addr = els.serverAddrSetting.value.trim().replace(/\/+$/, '');
  if (!addr) { alert('请输入服务器地址'); return; }
  localStorage.setItem('serverAddr', addr);
  alert('已保存，下次打开页面会自动连接该服务器');
}

/* 输出文件设置 */
async function loadOutputSettings() {
  try {
    const res = await fetch(apiUrl(CONFIG.apiSettings), { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();
    state.outputSettings = {
      download_enabled: !!d.download_enabled,
      download_path: d.download_path || '',
    };
    renderOutputSettings();
  } catch (err) {
    console.warn('获取输出设置失败：', err);
  }
}

function renderOutputSettings() {
  els.btnToggleDownload.textContent = state.outputSettings.download_enabled ? '已启用' : '已禁用';
  els.btnToggleDownload.classList.toggle('primary', state.outputSettings.download_enabled);
  els.downloadDirName.textContent = state.downloadDirName || '未选择';
}

async function pickDownloadDir() {
  if (!('showDirectoryPicker' in window)) {
    alert('当前浏览器不支持目录选择，请使用 Chrome 或 Edge');
    return;
  }
  try {
    const dirHandle = await window.showDirectoryPicker();
    // 在用户手势内立即请求读写权限，这样后续自动写入时就不需要再次用户激活
    if (typeof dirHandle.requestPermission === 'function') {
      const perm = await dirHandle.requestPermission({ mode: 'readwrite' });
      if (perm !== 'granted') {
        alert('未获得该目录的写入权限，自动保存可能失败');
      }
    }
    state.downloadDirHandle = dirHandle;
    state.downloadDirName = dirHandle.name;
    els.downloadDirName.textContent = dirHandle.name;
    alert('已选择目录：' + dirHandle.name + '（已授权写入）');
  } catch (e) {
    console.warn('选择目录失败：', e);
  }
}

async function saveOutputSettings() {
  const settings = {
    download_enabled: state.outputSettings.download_enabled,
    download_path: state.downloadDirName || '',
  };
  try {
    const res = await fetch(apiUrl(CONFIG.apiSettings), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    state.outputSettings = settings;
    renderOutputSettings();
    alert('保存成功');
  } catch (err) {
    alert('保存失败：' + err.message);
  }
}

/* 检测完成后，若启用下载则把 all/{视频名}.zip 写入所选目录（未选目录则回退默认下载） */
async function maybeDownloadAll(resultUrl) {
  if (!state.outputSettings.download_enabled) return;
  const m = String(resultUrl).match(/([^/]+)\.mp4$/);
  if (!m) return;
  const video = m[1];
  if (localStorage.getItem('downloadedVideo') === video) return;  // 同一视频已下载过，避免重复下载
  localStorage.setItem('downloadedVideo', video);
  const url = state.server + CONFIG.apiDownloadAll + '?video=' + encodeURIComponent(video);

  if (state.downloadDirHandle) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error('下载接口 HTTP ' + res.status);
      const blob = await res.blob();
      const fileHandle = await state.downloadDirHandle.getFileHandle(video + '.zip', { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(blob);
      await writable.close();
      console.log('已保存 ' + video + '.zip 到所选目录');
      return;
    } catch (e) {
      alert('写入所选目录失败，已回退默认下载：' + (e && e.message ? e.message : e));
    }
  }
  fallbackDownload(url, video);
}

function fallbackDownload(url, video) {
  const a = document.createElement('a');
  a.href = url;
  a.download = video + '.zip';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function loadSites() {
  try {
    const res = await fetch(apiUrl(CONFIG.apiSites), { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();
    renderSites(d.sites || []);
    syncSiteSelect(d.sites || []);
  } catch (err) {
    els.siteList.innerHTML = '<div class="site-empty">获取边坡列表失败</div>';
  }
}

function syncSiteSelect(sites) {
  const names = sites.map((s) => s.name);
  const options = ['<option value="无边坡">无边坡</option>'].concat(
    names.map((n) => `<option value="${n}">${n}</option>`)
  );
  els.siteSelect.innerHTML = options.join('');
  state.site = names.length ? names[0] : '无边坡';
}

function renderSites(sites) {
  if (!sites.length) {
    els.siteList.innerHTML = '<div class="site-empty">暂无边坡，请上传 CSV 标定文件</div>';
    return;
  }
  els.siteList.innerHTML = sites.map((s) => `
    <div class="site-item" data-name="${s.name}">
      <span class="site-name">${s.name}</span>
      <span class="site-csv">${s.csv}</span>
      <input type="text" value="${s.name}" placeholder="重命名">
      <button class="btn btn-rename">保存</button>
      <button class="btn danger btn-delete">删除</button>
    </div>
  `).join('');

  els.siteList.querySelectorAll('.site-item').forEach((item) => {
    const name = item.dataset.name;
    const input = item.querySelector('input');
    const btnRename = item.querySelector('.btn-rename');
    const btnDelete = item.querySelector('.btn-delete');
    btnRename.addEventListener('click', async () => {
      const newName = input.value.trim();
      if (!newName || newName === name) return;
      await renameSite(name, newName);
    });
    btnDelete.addEventListener('click', async () => {
      if (!confirm(`确定删除边坡「${name}」及其 CSV 文件吗？`)) return;
      await deleteSite(name);
    });
  });
}

async function deleteSite(name) {
  try {
    const res = await fetch(apiUrl(CONFIG.apiSitesDelete), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const d = await res.json();
    if (!res.ok) { alert(d.error || '删除失败'); return; }
    await loadSites();
  } catch (err) {
    alert('删除失败：' + err.message);
  }
}

async function renameSite(oldName, newName) {
  try {
    const res = await fetch(apiUrl(CONFIG.apiSitesRename), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_name: oldName, new_name: newName }),
    });
    const d = await res.json();
    if (!res.ok) { alert(d.error || '重命名失败'); return; }
    await loadSites();
  } catch (err) {
    alert('重命名失败：' + err.message);
  }
}

async function uploadCsv(file) {
  if (!file) return;
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
  const base64Content = String(dataUrl).split(',')[1];
  try {
    const res = await fetch(apiUrl(CONFIG.apiSitesUpload), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: base64Content }),
    });
    const d = await res.json();
    if (!res.ok) { alert(d.error || '上传失败'); return; }
    await loadSites();
  } catch (err) {
    alert('上传失败：' + err.message);
  }
}

/* =========================================================
 * 初始化
 * ========================================================= */
function init() {
  chart.init(els.levelChart);
  live.init();
  gauge.init();

  /* 站点下拉框 */
  const names = Object.keys(CONFIG.sites);
  els.siteSelect.innerHTML = names.map((n) => `<option value="${n}">${n}</option>`).join('');
  state.site = names[0];
  if (CONFIG.sites[state.site]) els.serverAddr.value = CONFIG.sites[state.site];
  els.siteSelect.addEventListener('change', () => {
    state.site = els.siteSelect.value;
    const url = CONFIG.sites[state.site];
    if (url) els.serverAddr.value = url;
    /* 已连接时切换站点：实时画面与结果轮询随之切换 */
    if (state.connected) {
      live.connect();
      state.resultUrl = '';
      startPolling();
    }
  });

  bindResultPlaceholder();

  els.btnConnect.addEventListener('click', connect);
  els.btnContinuous.addEventListener('click', () => startDetect('continuous'));
  els.btnSingle.addEventListener('click', () => startDetect('single'));
  els.btnStop.addEventListener('click', stopDetect);
  els.btnSettings.addEventListener('click', openSettings);
  els.btnSettingsClose.addEventListener('click', closeSettings);
  els.tabSites.addEventListener('click', () => switchSettingsTab('sites'));
  els.tabOutput.addEventListener('click', () => switchSettingsTab('output'));
  els.tabServer.addEventListener('click', () => switchSettingsTab('server'));
  els.tabContinuous.addEventListener('click', () => switchSettingsTab('continuous'));
  els.btnUploadCsv.addEventListener('click', () => els.csvFile.click());
  els.csvFile.addEventListener('change', () => {
    if (els.csvFile.files.length) {
      uploadCsv(els.csvFile.files[0]);
      els.csvFile.value = '';
    }
  });
  els.btnToggleDownload.addEventListener('click', () => {
    state.outputSettings.download_enabled = !state.outputSettings.download_enabled;
    renderOutputSettings();
  });
  els.btnPickDir.addEventListener('click', pickDownloadDir);
  els.btnSaveOutput.addEventListener('click', saveOutputSettings);
  els.btnSaveServer.addEventListener('click', saveServerAddr);
  els.btnSaveContinuous.addEventListener('click', saveContinuousSettings);

  setConn(false, '未连接');
  setMode('idle');
  startDemo();   // 未连接服务器时展示演示数据，连接后自动切换为真实数据

  /* 若已保存服务器地址，自动填入并连接 */
  const savedAddr = localStorage.getItem('serverAddr');
  if (savedAddr) {
    els.serverAddr.value = savedAddr;
    els.serverAddrSetting.value = savedAddr;
    connect();
  }
}

init();

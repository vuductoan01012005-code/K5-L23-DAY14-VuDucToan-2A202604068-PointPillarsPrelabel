// ==UserScript==
// @name         CVAT labeling telemetry (Day 14 bonus)
// @description  Ghi click trên canvas + phím tắt trong job CVAT, gửi vào /api/events (ClickHouse) để tính metric chỉnh sửa nhãn.
// @match        http://localhost:8080/tasks/*/jobs/*
// @match        *://*/tasks/*/jobs/*
// @grant        none
// ==/UserScript==
//
// Vì sao cần: log gốc của CVAT chỉ có click:element cho nút (ant-btn), change:frame, save:job...
// Không có toạ độ click trên canvas, không biết phím nào được bấm. Script này bổ sung hai thứ đó
// mà không sửa mã CVAT: dùng đúng endpoint /api/events và các scope server đã cho phép.
//   click:element  obj_name="canvas-click"  payload {x, y, button, dt_ms, dist_px, view}
//   debug:info     obj_name="keydown"       obj_val=<tổ hợp phím>  payload {dt_ms}
(() => {
    'use strict';
    const FLUSH_MS = 5000;
    const jobMatch = window.location.pathname.match(/\/tasks\/(\d+)\/jobs\/(\d+)/);
    if (!jobMatch) return;
    const taskId = Number(jobMatch[1]);
    const jobId = Number(jobMatch[2]);
    let buffer = [];
    const history = []; // bản sao cục bộ để tải về file, không phụ thuộc quyền đọc log server
    let lastClick = null; // {t, x, y}
    let lastKeyT = null;

    const csrf = () => (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || '';
    const push = (scope, objName, objVal, payload) => {
        const ev = {
        scope, obj_name: objName, obj_val: objVal, source: 'client',
        timestamp: new Date().toISOString(), task_id: taskId, job_id: jobId,
        payload: JSON.stringify(payload),
    };
        buffer.push(ev); history.push(ev);
    };

    // Chỉ tính click trên vùng vẽ (canvas 2D/3D và các view phụ Top/Side/Front)
    document.addEventListener('pointerdown', (e) => {
        const canvas = e.target.closest && e.target.closest('canvas, svg#cvat_canvas_content');
        if (!canvas) return;
        const now = performance.now();
        const view = (canvas.closest('[class*="cvat-canvas3d"], [class*="cvat-canvas"]') || canvas).className;
        const rec = {
            x: Math.round(e.clientX), y: Math.round(e.clientY), button: e.button,
            dt_ms: lastClick ? Math.round(now - lastClick.t) : null,
            dist_px: lastClick ? Math.round(Math.hypot(e.clientX - lastClick.x, e.clientY - lastClick.y)) : null,
            view: String(view).split(/\s+/).find((c) => c.startsWith('cvat')) || 'canvas',
        };
        lastClick = { t: now, x: e.clientX, y: e.clientY };
        push('click:element', 'canvas-click', null, rec);
    }, true);

    // Phím tắt: bỏ qua khi đang gõ vào ô nhập liệu
    document.addEventListener('keydown', (e) => {
        if (e.repeat || ['INPUT', 'TEXTAREA'].includes(e.target.tagName) || e.target.isContentEditable) return;
        if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return;
        const combo = [e.ctrlKey && 'Ctrl', e.metaKey && 'Meta', e.altKey && 'Alt', e.shiftKey && 'Shift', e.key]
            .filter(Boolean).join('+');
        const now = performance.now();
        push('debug:info', 'keydown', combo, { dt_ms: lastKeyT ? Math.round(now - lastKeyT) : null });
        lastKeyT = now;
    }, true);

    const flush = (keepalive = false) => {
        if (!buffer.length) return Promise.resolve('empty');
        const events = buffer; buffer = [];
        return fetch('/api/events', {
            method: 'POST', credentials: 'include', keepalive,
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf() },
            body: JSON.stringify({ events, timestamp: new Date().toISOString() }),
        }).then((r) => { if (!r.ok) buffer = events.concat(buffer); return r.status; })
            .catch((err) => { buffer = events.concat(buffer); return String(err); });
    };
    setInterval(flush, FLUSH_MS);
    window.addEventListener('pagehide', () => flush(true));
    // Tải log của phiên này: gõ __d14Telemetry.download() trong Console (F12)
    const download = () => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([JSON.stringify(history, null, 1)], { type: 'application/json' }));
        a.download = `d14-telemetry-job${jobId}-${Date.now()}.json`;
        a.click();
        return history.length;
    };
    window.__d14Telemetry = { flush, download, pending: () => buffer.length, count: () => history.length };
})();

// Image zoom and the touch-only constellation fullscreen.
// - zoomable(img): touch opens the swipe/pinch lightbox below; mouse opens a modal
//   <dialog> (fit to screen, click for 1:1 pixels and drag to pan, Esc / × / backdrop
//   closes), styled and behaving like the expanded chart view in charts.js.
// - fullscreenStage(): touch only; a no-op on desktop.
//
// Both overlays push a history entry, so the phone's back gesture closes them
// instead of leaving the page.

export const isTouch = () => matchMedia('(pointer: coarse)').matches;

function historyOverlay(onClose) {
  history.pushState({ overlay: true }, '');
  const pop = () => { window.removeEventListener('popstate', pop); onClose(); };
  window.addEventListener('popstate', pop);
  return () => history.back();               // close via back, so history stays balanced
}

function closeButton(onClick) {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'overlay-close'; b.setAttribute('aria-label', 'Close'); b.textContent = '×';
  b.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
  return b;
}

/** Tap an image to open its detail view: the image fills 50% of the screen
 *  height, vertically centred, and scrolls horizontally with native momentum.
 *  Pinch changes the height (50% .. full screen) anchored on the fingers;
 *  double-tap toggles 50% <-> full height. Closes on ×, back, a tap on the
 *  black backdrop, or a vertical swipe (> 90 px) in either direction. */
export function zoomable(img) {
  if (img.dataset.zoomable) return img;
  img.dataset.zoomable = '1';
  img.classList.add('zoomable');
  img.title = img.title || 'click to enlarge';
  // decide at click time, so a window that changes pointer type still gets the right view
  img.addEventListener('click', () => (isTouch() ? openLightbox : openImageModal)(img.currentSrc || img.src, img.alt));
  return img;
}

/** Mouse: the image in a modal dialog, fitted to the screen. A click on the image
 *  toggles 1:1 pixels; at 1:1 a drag pans. Esc, × or a backdrop click closes. */
function openImageModal(src, alt) {
  const dialog = document.createElement('dialog');
  dialog.className = 'img-modal';
  dialog.setAttribute('aria-label', alt || 'image');
  const head = document.createElement('div'); head.className = 'img-modal-head';
  const title = document.createElement('h2'); title.textContent = alt || '';
  const hint = document.createElement('span'); hint.className = 'img-modal-hint';
  hint.textContent = 'click the image for full size · Esc to close';
  const close = document.createElement('button');
  close.type = 'button'; close.className = 'img-modal-close'; close.textContent = '×';
  close.setAttribute('aria-label', 'close'); close.title = 'close (Esc)';
  head.append(title, hint, close);
  const stage = document.createElement('div'); stage.className = 'img-modal-stage';
  const im = document.createElement('img'); im.src = src; im.alt = alt || ''; im.draggable = false;
  stage.append(im);
  dialog.append(head, stage);
  document.body.append(dialog);

  // fit <-> 1:1 on click; drag pans at 1:1 (a drag never toggles)
  let drag = null;
  const setActual = (on) => {
    im.classList.toggle('actual', on);
    hint.textContent = on ? 'drag to pan · click to fit · Esc to close' : 'click the image for full size · Esc to close';
  };
  im.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, sl: stage.scrollLeft, st: stage.scrollTop, moved: false };
    im.setPointerCapture(e.pointerId);
  });
  im.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
    if (drag.moved && im.classList.contains('actual')) { stage.scrollLeft = drag.sl - dx; stage.scrollTop = drag.st - dy; }
  });
  im.addEventListener('pointerup', (e) => {
    if (!drag) return;
    const d = drag; drag = null;
    if (d.moved) return;
    // keep the clicked point under the cursor when switching to 1:1
    const r = im.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    const toActual = !im.classList.contains('actual');
    setActual(toActual);
    if (toActual) {
      const s = stage.getBoundingClientRect();
      stage.scrollLeft = fx * im.naturalWidth - (e.clientX - s.left);
      stage.scrollTop = fy * im.naturalHeight - (e.clientY - s.top);
    }
  });
  im.addEventListener('pointercancel', () => { drag = null; });

  close.addEventListener('click', () => dialog.close());
  // backdrop clicks land on the dialog itself, outside its box; ignore the opening click
  const openedAt = performance.now();
  dialog.addEventListener('click', (e) => {
    if (e.target !== dialog || performance.now() - openedAt < 300) return;
    const r = dialog.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => { dialog.remove(); document.documentElement.classList.remove('img-modal-open'); });
  document.documentElement.classList.add('img-modal-open');
  dialog.showModal();
  close.focus();
}

function openLightbox(src, alt) {
  const box = document.createElement('div');
  box.className = 'lightbox';
  const im = document.createElement('img');
  im.src = src; im.alt = alt || ''; im.draggable = false;
  box.append(im);
  let close;
  const btn = closeButton(() => close());
  btn.classList.add('lightbox-close');
  document.body.append(box, btn);
  document.body.classList.add('overlay-open');

  // Cover exactly what is on screen. `position: fixed; inset: 0` follows the layout
  // viewport, which a mobile browser makes larger than the screen when anything on
  // the page is too wide (or the page is pinch-zoomed); the box was then too large
  // and off centre. The visual viewport is what the user actually sees.
  const vv = window.visualViewport;
  const fitToScreen = () => {
    if (!vv) return;
    Object.assign(box.style, { inset: 'auto', left: `${vv.offsetLeft}px`, top: `${vv.offsetTop}px`, width: `${vv.width}px`, height: `${vv.height}px` });
    Object.assign(btn.style, { left: `${vv.offsetLeft + vv.width - 54}px`, top: `${vv.offsetTop + 10}px`, right: 'auto' });
  };
  fitToScreen();

  // Height is the only zoom state; horizontal position is the box's native
  // scrollLeft, so one-finger flings keep the browser's momentum scrolling.
  const minH = () => box.clientHeight * 0.5, maxH = () => box.clientHeight;
  let h = minH();
  const setHeight = (next, anchorX) => {       // keep the content under anchorX fixed on screen
    const prev = im.getBoundingClientRect().height || h;
    next = Math.min(maxH(), Math.max(minH(), next));
    const contentX = (box.scrollLeft + anchorX) / prev;
    h = next; im.style.height = `${h}px`;
    box.scrollLeft = contentX * h - anchorX;
  };
  im.style.height = `${h}px`;

  const pts = new Map();
  let pinch = null, lastTap = 0, swipe = null;   // swipe: {x0, y0, dy, on} for a one-finger vertical drag
  const mid = () => { const [a, b] = [...pts.values()]; return { x: (a.x + b.x) / 2, d: Math.hypot(a.x - b.x, a.y - b.y) }; };
  const setSwipe = (dy) => {                    // image follows the finger, backdrop fades
    im.style.transform = dy ? `translateY(${dy}px)` : '';
    box.style.backgroundColor = `rgba(0,0,0,${0.94 * Math.max(0.25, 1 - Math.abs(dy) / 400)})`;
  };
  box.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.overlay-close')) return;
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    im.classList.remove('settle');
    swipe = pts.size === 1 ? { x0: e.clientX, y0: e.clientY, dy: 0, on: false } : null;
    if (pts.size === 2) pinch = { d: mid().d, h };
    if (pts.size === 1) {
      const now = performance.now();
      if (now - lastTap < 300 && e.target === im) { setHeight(h > minH() * 1.05 ? minH() : maxH(), e.clientX); lastTap = 0; swipe = null; return; }
      lastTap = now;
    }
  });
  box.addEventListener('pointermove', (e) => {
    const p = pts.get(e.pointerId); if (!p) return;
    p.x = e.clientX; p.y = e.clientY;
    if (pts.size >= 2 && pinch) { const m = mid(); setHeight(pinch.h * m.d / Math.max(pinch.d, 1), m.x); return; }
    if (swipe && pts.size === 1) {              // horizontal drags belong to native scrolling (touch-action: pan-x)
      const dx = e.clientX - swipe.x0, dy = e.clientY - swipe.y0;
      if (!swipe.on && Math.abs(dy) > 10 && Math.abs(dy) > 2 * Math.abs(dx)) swipe.on = true;
      if (swipe.on) { swipe.dy = dy; setSwipe(dy); }
    }
  });
  const end = (e) => {
    pts.delete(e.pointerId); if (pts.size < 2) pinch = null;
    if (swipe?.on && e.type === 'pointerup' && Math.abs(swipe.dy) > 90) { swipe = null; close(); return; }
    if (swipe?.on) { im.classList.add('settle'); setSwipe(0); }   // not far enough: spring back
    swipe = null;
  };
  box.addEventListener('pointerup', end); box.addEventListener('pointercancel', end);
  // A tap on the black backdrop (not the image) closes.
  box.addEventListener('click', (e) => { if (e.target === box) close(); });
  const onResize = () => { fitToScreen(); setHeight(h, box.clientWidth / 2); };   // rotation: stay within 50%..100%
  window.addEventListener('resize', onResize);
  vv?.addEventListener('resize', onResize);
  vv?.addEventListener('scroll', fitToScreen);

  close = historyOverlay(() => {
    window.removeEventListener('resize', onResize);
    vv?.removeEventListener('resize', onResize); vv?.removeEventListener('scroll', fitToScreen);
    box.remove(); btn.remove(); document.body.classList.remove('overlay-open');
  });
}

/** Tap the constellation stage to take it fullscreen with touch orbit/pinch
 *  enabled; × or back returns it to the page. `view` is a LatentView. */
export function fullscreenStage(stage, view, hint = 'drag to orbit · pinch to zoom · double-tap to reset') {
  if (!isTouch() || stage.dataset.fs) return;
  stage.dataset.fs = '1';
  stage.classList.add('tap-to-expand');
  stage.addEventListener('click', () => {
    if (stage.classList.contains('fs')) return;
    const home = { parent: stage.parentNode, next: stage.nextSibling };
    // .panel's backdrop-filter makes it the containing block for position:fixed,
    // so the stage moves to <body> while fullscreen and is put back after.
    document.body.append(stage);
    stage.classList.add('fs');
    document.body.classList.add('overlay-open');
    view.setTouch(true);
    const bar = document.createElement('div');
    bar.className = 'fs-hint'; bar.textContent = hint;
    let close;
    const btn = closeButton(() => close());
    stage.append(bar, btn);
    close = historyOverlay(() => {
      view.setTouch(false);
      bar.remove(); btn.remove();
      stage.classList.remove('fs');
      document.body.classList.remove('overlay-open');
      home.parent.insertBefore(stage, home.next);
    });
  });
}

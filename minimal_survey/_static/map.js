// Spatial arrangement canvas.
// Expects window.PM_CONFIG = {dots: [...], box: 550, width: 760} and
// <canvas id="dragCanvas">, <input type="hidden" id="positions">.

window.addEventListener("load", function () {
  const cfg = window.PM_CONFIG;
  const canvas = document.getElementById("dragCanvas");
  const ctx = canvas.getContext("2d");
  const hidden = document.getElementById("positions");

  const BOX = cfg.box;     // side length of the positioning square
  const W = cfg.width;     // full canvas width (square + "parking lot")
  const H = cfg.box;

  const points = cfg.dots.map(function (d) {
    return {
      varname: d.varname,
      name_disp: d.name_disp,
      dottype: d.dottype,
      x: d.x,
      y: d.y,
      radius: 14,
      color: COLORS[d.varname] || COLORS[d.dottype] || "#cccccc",
      t_first_moved: -1,
      t_last_moved: -1,
    };
  });

  // --- sizing (fit to screen, sharp on retina) ---
  const ratio = window.devicePixelRatio || 1;
  const scale = Math.min(1, canvas.parentElement.clientWidth / W);
  canvas.width = W * scale * ratio;
  canvas.height = H * scale * ratio;
  canvas.style.width = W * scale + "px";
  canvas.style.height = H * scale + "px";
  ctx.setTransform(ratio * scale, 0, 0, ratio * scale, 0, 0);

  let dragging = null;
  let hovered = null;
  let offset = { dx: 0, dy: 0 };

  function draw() {
    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = "#fafafa";
    ctx.fillRect(0, 0, BOX, BOX);
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, BOX - 1, BOX - 1);

    ctx.fillStyle = "#999";
    ctx.font = "12px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("\u2190 drag dots into the square", (BOX + W) / 2, H - 8);

    const order = points.filter(function (p) { return p !== dragging; });
    if (dragging) order.push(dragging);

    order.forEach(function (p) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
      ctx.strokeStyle = p === hovered ? "#fff" : "#333";
      ctx.lineWidth = p === hovered ? 3 : 2;
      ctx.stroke();

      ctx.fillStyle = "#333";
      ctx.font = "12px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(p.name_disp, p.x, p.y - p.radius - 2);
    });
  }

  function getPos(evt) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((evt.clientX - rect.left) * W) / rect.width,
      y: ((evt.clientY - rect.top) * H) / rect.height,
    };
  }

  function findPoint(pos) {
    for (let i = points.length - 1; i >= 0; i--) {
      const p = points[i];
      if (Math.hypot(p.x - pos.x, p.y - pos.y) < p.radius) return p;
    }
    return null;
  }

  function allInside() {
    return points.every(function (p) {
      return p.x >= 0 && p.x <= BOX && p.y >= 0 && p.y <= BOX;
    });
  }

  function updateHidden() {
    hidden.value = JSON.stringify(
      points.map(function (p) {
        return {
          varname: p.varname,
          name_disp: p.name_disp,
          dottype: p.dottype,
          x: Math.round(p.x * 10) / 10,
          y: Math.round(p.y * 10) / 10,
          t_first_moved: p.t_first_moved,
          t_last_moved: p.t_last_moved,
        };
      })
    );
  }

  // --- pointer events (mouse + touch) ---
  canvas.addEventListener("pointerdown", function (evt) {
    const pos = getPos(evt);
    dragging = findPoint(pos);
    if (dragging) {
      canvas.setPointerCapture(evt.pointerId);
      offset = { dx: dragging.x - pos.x, dy: dragging.y - pos.y };
    }
    draw();
    evt.preventDefault();
  });

  canvas.addEventListener("pointermove", function (evt) {
    const pos = getPos(evt);
    if (dragging) {
      const r = dragging.radius;
      dragging.x = Math.max(r, Math.min(BOX - r, pos.x + offset.dx));
      dragging.y = Math.max(r, Math.min(BOX - r, pos.y + offset.dy));
      draw();
    } else {
      const h = findPoint(pos);
      canvas.style.cursor = h ? "grab" : "default";
      if (h !== hovered) {
        hovered = h;
        draw();
      }
    }
  });

  function endDrag() {
    if (dragging) {
      const now = Math.floor(Date.now() / 1000);
      if (dragging.t_first_moved === -1) dragging.t_first_moved = now;
      dragging.t_last_moved = now;
      dragging = null;
      updateHidden();
      draw();
    }
  }
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  // --- submit guard ---
  const nextButton = document.querySelector(".otree-btn-next");
  if (nextButton) {
    nextButton.addEventListener(
      "click",
      function (e) {
        if (!allInside()) {
          e.preventDefault();
          alert("Please place all dots inside the square before continuing.");
          return;
        }
        updateHidden();
      },
      true
    );
  }

  updateHidden();
  draw();
});
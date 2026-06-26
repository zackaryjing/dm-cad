const pptxgen = require("/root/projects/presentation/middle/node_modules/pptxgenjs");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "dm-cad2";
pptx.subject = "DeepCAD command operation table";
pptx.title = "DeepCAD Command Operations";
pptx.company = "XJTU";
pptx.lang = "zh-CN";
pptx.theme = {
  lang: "zh-CN",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

const C = {
  bg: "F7F8FA",
  ink: "172033",
  muted: "596273",
  line: "D7DDE8",
  header: "203A5F",
  header2: "EAF0F8",
  accent: "C96C3B",
  pale: "FFF4EC",
  white: "FFFFFF",
};

const slide = pptx.addSlide();
slide.background = { color: C.bg };

function addText(text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    margin: 0,
    breakLine: false,
    fit: "shrink",
    fontSize: opts.fontSize || 16,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    valign: opts.valign || "mid",
    align: opts.align || "left",
    ...opts,
  });
}

slide.addShape(pptx.ShapeType.rect, {
  x: 0,
  y: 0,
  w: 13.333,
  h: 0.16,
  fill: { color: C.accent },
  line: { color: C.accent },
});

addText("DeepCAD 命令类型与参数槽位", 0.58, 0.38, 7.2, 0.42, {
  fontSize: 24,
  bold: true,
});
addText("Command operations and active argument fields in the 17-D CAD token", 0.6, 0.82, 8.2, 0.28, {
  fontSize: 10.8,
  color: C.muted,
});

slide.addShape(pptx.ShapeType.roundRect, {
  x: 9.05,
  y: 0.34,
  w: 3.72,
  h: 0.78,
  rectRadius: 0.06,
  fill: { color: C.pale },
  line: { color: "F0B48C", width: 1 },
});
addText("[cmd, x, y, α, f, r, θ, φ, γ, pₓ, pᵧ, p_z, s, e₁, e₂, b, u]", 9.25, 0.48, 3.32, 0.23, {
  fontSize: 9.4,
  bold: true,
  color: "6E351B",
  align: "center",
});
addText("无效槽位以 -1 padding；所有离散参数按 8-bit bin 表示", 9.2, 0.78, 3.42, 0.18, {
  fontSize: 7.4,
  color: "7B4B35",
  align: "center",
});

const rows = [
  [
    { text: "cmd id", options: { bold: true } },
    { text: "操作", options: { bold: true } },
    { text: "建模含义", options: { bold: true } },
    { text: "启用参数槽位", options: { bold: true } },
  ],
  ["0", "Line", "草图直线段；起点由上一条曲线终点隐式给出", "x, y"],
  ["1", "Arc", "草图圆弧；用终点、扫掠角和方向标志恢复圆弧", "x, y, α, f"],
  ["2", "Circle", "草图圆；圆心和半径定义闭合圆轮廓", "x, y, r"],
  ["3", "EOS", "序列结束标记；也用于 padding", "无"],
  ["4", "SOL", "Sketch loop 起始标记，用于分隔轮廓环", "无"],
  ["5", "Ext", "拉伸操作；确定草图平面、位置、尺度和布尔/拉伸方式", "θ, φ, γ, pₓ, pᵧ, p_z, s, e₁, e₂, b, u"],
];

slide.addTable(rows, {
  x: 0.58,
  y: 1.35,
  w: 12.18,
  h: 4.6,
  border: { type: "solid", color: C.line, pt: 0.65 },
  color: C.ink,
  fontSize: 10.2,
  valign: "mid",
  margin: 0.07,
  autoFit: false,
  colW: [0.82, 1.15, 4.6, 5.61],
  rowH: [0.46, 0.58, 0.58, 0.58, 0.52, 0.52, 0.72],
  fill: { color: C.white },
  autoPage: false,
  marginH: 0.08,
  marginV: 0.06,
  fit: "shrink",
  breakLine: false,
  bold: false,
  align: "left",
  valign: "mid",
  fill: "FFFFFF",
  border: { color: C.line, pt: 0.7 },
  fontSize: 10.1,
  color: C.ink,
  headerRow: true,
  headerFill: C.header,
  headerColor: C.white,
});

// Overlay header background for reliable styling across Office/LibreOffice.
slide.addShape(pptx.ShapeType.rect, {
  x: 0.58,
  y: 1.35,
  w: 12.18,
  h: 0.46,
  fill: { color: C.header },
  line: { color: C.header, transparency: 100 },
});
const headerX = [0.58, 1.4, 2.55, 7.15];
const headerW = [0.82, 1.15, 4.6, 5.61];
["cmd id", "操作", "建模含义", "启用参数槽位"].forEach((t, i) => {
  addText(t, headerX[i] + 0.08, 1.46, headerW[i] - 0.16, 0.18, {
    fontSize: 10.3,
    bold: true,
    color: C.white,
    align: i === 0 ? "center" : "left",
  });
});

slide.addShape(pptx.ShapeType.roundRect, {
  x: 0.6,
  y: 6.22,
  w: 12.12,
  h: 0.62,
  rectRadius: 0.04,
  fill: { color: C.header2 },
  line: { color: C.line, width: 0.6 },
});
addText("补充：b 表示 Boolean operation（NewBody / Join / Cut / Intersect），u 表示 Extent type（OneSide / Symmetric / TwoSides）。", 0.82, 6.37, 11.68, 0.18, {
  fontSize: 9.5,
  color: C.ink,
});

addText("Source: DeepCAD cadlib.macro / curves / extrude", 0.6, 7.12, 5.2, 0.16, {
  fontSize: 7.5,
  color: "8A93A3",
});

pptx.writeFile({ fileName: "/root/projects/presentation/final/out/table.pptx" });

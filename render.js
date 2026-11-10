const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType, BorderStyle,
        ShadingType, ImageRun, Footer, Header, PageNumber, PageBreak, TabStopType } = require('docx');
const C = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const FONT = 'Times New Roman';
function runs(text, base = {}) {              // minimal inline markup: <i>..</i>, <b>..</b>
  const out = []; const re = /<(i|b)>(.*?)<\/\1>/gs; let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    out.push(new TextRun({ text: m[2], font: FONT, ...base, italics: m[1] === 'i' ? true : base.italics, bold: m[1] === 'b' ? true : base.bold }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const P = (text, o = {}) => new Paragraph({ children: runs(text, { size: o.size || 22, italics: o.italics, bold: o.bold }),
  alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 100, before: o.before ?? 0, line: o.line || 264 },
  indent: o.indent, keepNext: o.keepNext, keepLines: o.keepLines });
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const rule = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: '000000' });
function table(b) {
  const widths = b.cols; const total = widths.reduce((a, c) => a + c, 0);
  const mk = (txt, i, o) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 70, right: 70 },
    borders: { top: o.top || none, bottom: o.bottom || none, left: none, right: none },
    shading: o.shade ? { type: ShadingType.CLEAR, color: 'auto', fill: 'F2F2F2' } : undefined,
    children: [new Paragraph({ keepNext: ((b.keep!==false) && !o.last) || !!o.kn, keepLines: true, alignment: (b.align && b.align[i] === 'l') ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { after: 0, line: 240 },
      children: runs(String(txt), { size: 17, bold: o.bold, italics: o.italics }) })] });
  const rows = [];
  rows.push(new TableRow({ tableHeader: true, children: b.header.map((h, i) => mk(h, i, { top: rule(10), bottom: rule(6), bold: true })) }));
  b.rows.forEach((r, ri) => {
    const isSub = r.length === 1;                       // sub-header row spanning the table
    if (isSub) { rows.push(new TableRow({ children: [new TableCell({ columnSpan: widths.length, width: { size: total, type: WidthType.DXA }, margins: { top: 40, bottom: 30, left: 70, right: 70 }, borders: { top: none, bottom: none, left: none, right: none },
      children: [new Paragraph({ keepNext: true, spacing: { after: 0 }, children: runs(r[0], { size: 17, italics: true, bold: true }) })] })] })); return; }
    const last = ri === b.rows.length - 1;
    rows.push(new TableRow({ cantSplit: true, children: r.map((c, i) => mk(c, i, { bottom: last ? rule(10) : undefined, last, kn: (b.kn||[]).includes(ri) || (b.keep===false && last) })) }));
  });
  return [P(`<b>${b.title}</b>`, { align: AlignmentType.LEFT, before: 120, after: 50, size: 20, keepNext: true }),
          new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows }),
          P(`<i>Notes:</i> ${b.note}`, { size: 16, after: 140, before: 40, line: 230 })];
}
function figure(b) {
  const w = Math.round(b.width_in * 96), h = Math.round(w * b.h / b.w);
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 }, keepNext: true,
            children: [new ImageRun({ type: 'png', data: fs.readFileSync(b.path), transformation: { width: w, height: h },
                       altText: { title: b.caption.slice(0, 40), description: b.caption, name: b.path.split('/').pop() } })] }),
          P(`<b>${b.caption}</b>`, { align: AlignmentType.LEFT, size: 19, after: 20, keepNext: true }),
          P(`<i>Notes:</i> ${b.note}`, { size: 16, after: 140, line: 230 })];
}
const children = [];
for (const b of C.blocks) {
  if (b.t === 'title') children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 }, children: [new TextRun({ text: b.text, font: FONT, size: 30, bold: true })] }));
  else if (b.t === 'center') children.push(P(b.text, { align: AlignmentType.CENTER, after: 40, size: b.size || 22 }));
  else if (b.t === 'h1') children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 200, after: 80 }, children: [new TextRun({ text: b.text, font: FONT, size: 24, bold: true })] }));
  else if (b.t === 'h2') children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 140, after: 60 }, children: [new TextRun({ text: b.text, font: FONT, size: 22, bold: true, italics: true })] }));
  else if (b.t === 'p') children.push(P(b.text));
  else if (b.t === 'eq') children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 80 }, children: runs(b.text, { size: 21, italics: true }) }));
  else if (b.t === 'hyp') children.push(P(b.text, { indent: { left: 360 } }));
  else if (b.t === 'abstract') children.push(P(b.text, { size: 21, after: 80 }));
  else if (b.t === 'ref') children.push(new Paragraph({ spacing: { after: 50, line: 240 }, indent: { left: 540, hanging: 540 }, alignment: AlignmentType.LEFT, children: runs(b.text, { size: 19 }) }));
  else if (b.t === 'table') children.push(...table(b));
  else if (b.t === 'figure') children.push(...figure(b));
  else if (b.t === 'pagebreak') children.push(new Paragraph({ children: [new PageBreak()] }));
}
const doc = new Document({
  creator: 'Author', title: C.title,
  styles: { default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 22, bold: true, italics: true }, paragraph: { spacing: { before: 140, after: 60 }, outlineLevel: 1 } } ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, bottom: 1200, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: C.running, font: FONT, size: 17, italics: true })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children }] });
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[3], buf); console.log('written', process.argv[3]); });

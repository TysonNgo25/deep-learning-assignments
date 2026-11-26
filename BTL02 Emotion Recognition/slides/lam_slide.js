// Tạo slide BTL02 từ một bản mô tả duy nhất: ra slides/BTL02_Nhom05.pptx (pptxgenjs) và
// slides/BTL02_Nhom05.pdf (cùng tọa độ, dựng bằng HTML rồi in bằng Edge/Chrome chạy ẩn).
//
// Phụ trách: Huỳnh Ngọc Trúc
//
// Cần Node và pptxgenjs (`npm install pptxgenjs`). Đứng ở thư mục BTL02:
//     node slides/lam_slide.js
// Mọi số liệu lấy từ báo cáo (report/*.md), nguồn gốc ở results/; sửa số thì sửa cả hai nơi.

const fs = require("fs");
const path = require("path");
const os = require("os");
const { execFileSync } = require("child_process");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const R = (p) => path.join(ROOT, "results", p);
const OUT = path.join(__dirname, "BTL02_Nhom05");
const W = 13.333, H = 7.5;

// Bảng màu: xanh đêm làm nền chính, đỏ san hô làm điểm nhấn (cảm xúc), xanh mòng két cho số liệu.
const COL = { ink: "22223B", coral: "E5484D", teal: "1B7F8C", amber: "F2A541", card: "F1F3F6",
              muted: "5C6370", white: "FFFFFF", soft: "C9CEDA" };
const FONT = "Calibri";
const MEMBERS = { truc: "Huỳnh Ngọc Trúc", khang: "Phạm Trần Minh Khang",
                  trung: "Tạ Quang Trung", nhan: "Ngô Nguyễn Thành Nhân" };

// Tỉ lệ rộng/cao các hình (đo từ file PNG) để đặt vào khung mà không méo.
const RATIO = {
  "eda/class_distribution.png": 2.0, "eda/mean_face_per_class.png": 5.25,
  "BASE/figures/curves.png": 2.75, "E5/figures/curves.png": 2.75,
  "figures/tong_hop_val_acc.png": 2.381, "final/figures/curves.png": 2.75,
  "final/figures/last_conv_filters.png": 0.897, "final/figures/last_conv_top_images.png": 0.554,
  "final/figures/gradcam.png": 1.556, "final/figures/confusion_matrix.png": 2.323,
};

// ------------------------------------------------------------------ phần tử
const text = (x, y, w, h, s, o = {}) => ({ t: "text", x, y, w, h, s, size: 16, color: COL.ink, ...o });
const bullets = (x, y, w, h, items, o = {}) => ({ t: "bullets", x, y, w, h, items, size: 16, color: COL.ink, ...o });
const box = (x, y, w, h, fill, o = {}) => ({ t: "rect", x, y, w, h, fill, ...o });
function img(file, x, y, w, h) {             // đặt hình vừa khung (contain), canh giữa
  const r = RATIO[file];
  let iw = w, ih = w / r;
  if (ih > h) { ih = h; iw = h * r; }
  return { t: "img", file: R(file), x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih };
}
const bar = (x, y, w, h, o) => ({ t: "bar", x, y, w, h, ...o });
const table = (x, y, w, colW, rows, o = {}) => ({ t: "table", x, y, w, colW, rows, size: 13, ...o });
function stat(x, y, w, h, big, label, color = COL.teal) {
  return [box(x, y, w, h, COL.card, { radius: true }),
          text(x + 0.2, y + 0.12, w - 0.4, h * 0.52, big, { size: 32, bold: true, color, valign: "bottom" }),
          text(x + 0.2, y + h * 0.62, w - 0.4, h * 0.36, label, { size: 13, color: COL.muted })];
}
const title = (s) => text(0.6, 0.38, 12.1, 0.8, s, { size: 30, bold: true, valign: "middle" });

// ------------------------------------------------------------------ nội dung 15 slide
const SLIDES = [
  { section: "Mở đầu", dark: true, by: "truc", notes:
    "Giới thiệu nhóm và đề tài: nhận dạng 7 cảm xúc trên ảnh khuôn mặt xám 48x48 của FER2013, viết bằng PyTorch. Kết quả chính: 66.09% accuracy trên PrivateTest, đánh giá một lần.",
    els: [
      text(0.8, 0.85, 11.7, 0.45, "Bài tập lớn 02 · Học phần Học sâu · Nhóm 05", { size: 17, color: COL.amber, bold: true }),
      text(0.8, 1.35, 11.7, 1.75, "Nhận dạng cảm xúc khuôn mặt trên FER2013\nbằng mạng nơ-ron tích chập", { size: 40, bold: true, color: COL.white }),
      text(0.8, 3.15, 11.7, 0.45, "PyTorch · 18 thực nghiệm mỗi lần đổi một biến · mô hình cuối 66.09% trên PrivateTest", { size: 18, color: COL.soft }),
      text(0.8, 3.85, 11.7, 1.45, "Huỳnh Ngọc Trúc (3123411312), nhóm trưởng\nPhạm Trần Minh Khang (3123411143)\nTạ Quang Trung (3123411318)\nNgô Nguyễn Thành Nhân (3123411205)", { size: 15, color: COL.white, lineSpacing: 1.25 }),
      img("eda/mean_face_per_class.png", 0.8, 5.25, 11.7, 1.5),
    ] },

  { section: "Mở đầu", by: "truc", notes:
    "Đề yêu cầu CNN nhận ảnh 48x48, ra vector 7 chiều; báo cáo có 4 phần bắt buộc. Mốc so sánh: tốt nhất thế giới 71% PrivateTest, báo cáo mẫu của thầy 0.64 đến 0.67.",
    els: [
      title("Đề bài: phân loại 7 cảm xúc từ ảnh xám 48×48"),
      text(0.6, 1.3, 7.6, 0.45, "Báo cáo phải có bốn phần", { size: 20, bold: true, color: COL.teal }),
      ...[["1", "Tóm tắt mã nguồn PyTorch"], ["2", "Nhiều thực nghiệm, mỗi lần đổi một tham số"],
          ["3", "Hình các filter của lớp tích chập cuối"], ["4", "Phân tích, giải thích theo cách hiểu của nhóm"]]
        .flatMap(([n, s], i) => {
          const x = 0.6 + (i % 2) * 3.9, y = 1.9 + Math.floor(i / 2) * 2.05;
          return [box(x, y, 3.6, 1.75, COL.card, { radius: true }),
                  { t: "circle", x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fill: COL.coral },
                  text(x + 0.25, y + 0.25, 0.6, 0.6, n, { size: 20, bold: true, color: COL.white, align: "center", valign: "middle" }),
                  text(x + 0.25, y + 0.95, 3.15, 0.75, s, { size: 16, bold: true })];
        }),
      ...stat(8.75, 1.3, 3.95, 1.6, "71%", "Tốt nhất thế giới trên PrivateTest (69% PublicTest)", COL.coral),
      ...stat(8.75, 3.1, 3.95, 1.6, "0.64 – 0.67", "val_acc của 4 CNN trong báo cáo mẫu (Keras)"),
      ...stat(8.75, 4.9, 3.95, 1.6, "7 lớp", "Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral", COL.ink),
    ] },

  { section: "Dữ liệu và mã nguồn", by: "truc", notes:
    "FER2013 có 35887 ảnh, chia sẵn theo cột Usage. Nhóm dùng PublicTest làm validation để chọn mọi thứ, PrivateTest là test và chỉ đánh giá một lần. Dữ liệu lệch lớp: Happy 25%, Disgust 1.5%, nên ngoài accuracy nhóm báo thêm F1 từng lớp.",
    els: [
      title("FER2013: 35887 ảnh, lệch lớp mạnh"),
      img("eda/class_distribution.png", 0.6, 1.3, 7.7, 3.85),
      bullets(0.6, 5.35, 7.7, 1.5, [
        "Training 28709 ảnh: huấn luyện",
        "PublicTest 3589 ảnh: validation, dùng để chọn cấu hình, epoch, checkpoint",
        "PrivateTest 3589 ảnh: test, chỉ đánh giá một lần ở mô hình cuối",
      ], { size: 15 }),
      ...stat(8.75, 1.3, 3.95, 1.5, "25.13%", "Happy, lớp đông nhất của tập train", COL.teal),
      ...stat(8.75, 2.95, 3.95, 1.5, "1.52%", "Disgust, chỉ 436 ảnh train", COL.coral),
      ...stat(8.75, 4.6, 3.95, 1.5, "24.94%", "Đoán toàn Happy đã đạt mức này trên validation: mốc sàn", COL.ink),
      text(8.75, 6.25, 3.95, 0.55, "→ báo thêm precision, recall, F1 từng lớp", { size: 14, color: COL.muted, italic: true }),
    ] },

  { section: "Dữ liệu và mã nguồn", by: "truc", notes:
    "Ảnh trung bình cho thấy Happy và Surprise khác rõ ở miệng, bốn lớp còn lại gần giống nhau. Nhóm còn đếm ảnh trùng: khoảng 8% ảnh test có y hệt trong train, 57 nhóm ảnh trùng mà khác nhãn, tức nhãn có nhiễu.",
    els: [
      title("Dữ liệu nhiễu: ảnh trùng, nhãn mâu thuẫn"),
      img("eda/mean_face_per_class.png", 0.6, 1.3, 12.1, 2.3),
      text(0.6, 3.65, 12.1, 0.5, "Ảnh trung bình mỗi lớp: Happy (miệng cười) và Surprise (miệng há) khác rõ; Angry, Fear, Sad, Neutral gần giống nhau.", { size: 15, color: COL.muted }),
      ...stat(0.6, 4.4, 3.85, 1.75, "34034", "ảnh khác nhau trên 35887 ảnh của cả bộ", COL.ink),
      ...stat(4.72, 4.4, 3.85, 1.75, "8.02%", "ảnh test (288 ảnh) trùng y hệt một ảnh train", COL.coral),
      ...stat(8.85, 4.4, 3.85, 1.75, "57", "nhóm ảnh giống hệt nhau nhưng mang nhãn khác nhau", COL.teal),
    ] },

  { section: "Dữ liệu và mã nguồn", by: "truc", notes:
    "Mã nguồn chia 5 file theo luồng xử lý. Ba nguyên tắc: mỗi run chỉ đổi một biến so với baseline.yaml; đặt seed để kết quả lặp lại; PrivateTest chỉ đọc trong evaluate_final và gọi lần hai sẽ báo lỗi.",
    els: [
      title("Mã nguồn PyTorch: một luồng, năm bước"),
      ...[["data.py", "Đọc CSV từng dòng, chuẩn hóa bằng μ, σ của train, augmentation cho train"],
          ["models.py", "4 kiến trúc Small → Large, tùy chọn BN, kích hoạt, dropout; khởi tạo He"],
          ["train.py", "Adam, early stopping và checkpoint theo val_loss, log CSV mỗi epoch"],
          ["evaluate.py", "Precision, recall, F1, confusion matrix; test chỉ một lần"],
          ["visualize.py", "Filter lớp conv cuối, feature map, Grad‑CAM"]]
        .flatMap(([f, s], i) => {
          const x = 0.6 + i * 2.5;
          const els = [box(x, 1.45, 2.15, 2.0, COL.card, { radius: true }),
                       text(x + 0.15, 1.6, 1.85, 0.45, f, { size: 17, bold: true, color: COL.teal }),
                       text(x + 0.15, 2.1, 1.85, 1.3, s, { size: 14 })];
          if (i < 4) els.push(text(x + 2.15, 2.15, 0.35, 0.6, "→", { size: 24, bold: true, color: COL.coral, align: "center" }));
          return els;
        }),
      text(0.6, 3.95, 12.1, 0.45, "Ba nguyên tắc để kết quả so sánh được và tin được", { size: 20, bold: true, color: COL.ink }),
      ...[["Một biến", "Mỗi run chỉ đổi một biến so với configs/baseline.yaml; khóa sai tên báo lỗi ngay"],
          ["Tái lập", "Cùng seed cho log giống hệt từng epoch (E1-2 trùng BASE, F-0 trùng E1-1)"],
          ["Test một lần", "PrivateTest chỉ đọc trong evaluate_final; gọi lần hai báo lỗi"]]
        .flatMap(([h, s], i) => {
          const x = 0.6 + i * 4.1;
          return [box(x, 4.55, 3.85, 1.6, COL.card, { radius: true }),
                  text(x + 0.2, 4.7, 3.45, 0.4, h, { size: 16, bold: true, color: COL.coral }),
                  text(x + 0.2, 5.12, 3.45, 0.95, s, { size: 14 })];
        }),
    ] },

  { section: "Dữ liệu và mã nguồn", by: "truc", notes:
    "Lúc đầu baseline đứng yên ở 24.94%, tức đoán toàn Happy. Nguyên nhân là khởi tạo mặc định của PyTorch làm tín hiệu teo qua 9 lớp ReLU. Đổi sang khởi tạo He, gradient ở lớp đầu lớn hơn khoảng 5500 lần và baseline đạt 66.43%.",
    els: [
      title("Baseline 66.43% sau khi đổi sang khởi tạo He"),
      img("BASE/figures/curves.png", 0.6, 1.3, 8.1, 2.95),
      text(0.6, 4.3, 8.1, 0.45, "Đường học của BASE: quá khớp nhẹ từ khoảng epoch 21–30, tốt nhất ở epoch 47", { size: 13, color: COL.muted }),
      box(0.6, 4.95, 8.1, 1.55, COL.card, { radius: true }),
      text(0.8, 5.05, 7.7, 0.4, "Cấu hình baseline", { size: 16, bold: true, color: COL.teal }),
      text(0.8, 5.47, 7.7, 1.2, "Medium (8 conv) · ReLU · không BatchNorm · dropout 0.2 / 0.5\nAdam lr 1e-4 · batch 64 · tối đa 60 epoch, patience 10\nLật ngang, xoay ±10°, dịch 10% · seed 0", { size: 14, lineSpacing: 1.2 }),
      ...stat(9.0, 1.3, 3.7, 1.65, "66.43%", "val_acc của BASE tại epoch có val_loss nhỏ nhất", COL.teal),
      ...stat(9.0, 3.15, 3.7, 1.65, "≈5500×", "gradient ở conv đầu: khởi tạo He so với mặc định", COL.coral),
      text(9.0, 5.0, 3.7, 1.75, "Khởi tạo mặc định của PyTorch: tín hiệu teo từ 0.34 xuống 0.005 qua 9 lớp ReLU, mạng đứng yên ở mức đoán toàn Happy.", { size: 14, color: COL.ink }),
    ] },

  { section: "Thực nghiệm", by: "khang", notes:
    "E1 so 4 kiến trúc: chênh nhau chỉ 1.98 điểm. Small có val_acc cao nhất nhưng nhiều tham số nhất vì chỉ pool 2 lần. Mạng sâu hơn quá khớp hơn. E2 thêm BatchNorm vào Small: thấp hơn 4.65 điểm và chưa hội tụ sau 60 epoch.",
    els: [
      title("E1–E2: kiến trúc ít ảnh hưởng, BatchNorm chưa hội tụ"),
      bar(0.6, 1.3, 7.3, 5.4, { title: "val_acc (%) tại epoch val_loss nhỏ nhất",
          cats: ["Small", "Medium (BASE)", "Medium_01", "Large", "Small + BN"],
          vals: [67.01, 66.43, 65.03, 65.87, 62.36], min: 55, max: 70, major: 5,
          colors: [COL.teal, COL.ink, COL.teal, COL.teal, COL.coral] }),
      bullets(8.3, 1.4, 4.4, 5.3, [
        "Bốn kiến trúc chỉ chênh 1.98 điểm (65.03% – 67.01%)",
        "Small nhiều tham số nhất (38.9 triệu): chỉ pool 2 lần, lớp FC đầu nhận 36864 chiều",
        "Mạng sâu hơn quá khớp hơn: train − val 6.00 (Medium_01), 5.14 (Large) so với 4.11",
        "BatchNorm: −4.65 điểm so với Small, chạm trần 60 epoch, train_acc vẫn thấp hơn val_acc",
      ], { size: 16 }),
    ] },

  { section: "Thực nghiệm", by: "trung", notes:
    "E3: Adam lr 1e-4 tốt nhất; lr 1e-3 kém 5.27 điểm; SGD học chậm; ReduceLROnPlateau giảm lr quá sớm. E4: khoảng cách train trừ val giảm dần khi tăng augmentation, từ 8.62 xuống 1.42 điểm; bỏ augmentation mất 6.55 điểm.",
    els: [
      title("E3–E4: Adam 1e-4 tốt nhất, augmentation chống quá khớp"),
      bar(0.6, 1.3, 6.0, 3.9, { title: "E3: val_acc (%)", cats: ["Adam 1e-4 (BASE)", "SGD 1e-3", "Adam 1e-3", "Plateau"],
          vals: [66.43, 61.83, 61.16, 65.20], min: 55, max: 70, major: 5, colors: [COL.ink, COL.coral, COL.coral, COL.teal] }),
      bar(6.73, 1.3, 6.0, 3.9, { title: "E4: train − val (điểm), càng thấp càng ít quá khớp",
          cats: ["Không", "Lật", "Lật + xoay + dịch (BASE)", "+ zoom 10%"],
          vals: [8.62, 7.14, 4.11, 1.42], min: 0, max: 10, major: 2, colors: [COL.coral, COL.coral, COL.ink, COL.teal] }),
      bullets(0.6, 5.35, 6.0, 1.45, [
        "Adam lr 1e-3: dừng sớm ở epoch 38, kém 5.27 điểm",
        "Plateau giảm lr còn 1e-5 ở epoch 26, rồi 1e-7: mạng ngừng học",
      ], { size: 15 }),
      bullets(6.73, 5.35, 6.0, 1.45, [
        "Bỏ augmentation: mất 6.55 điểm, quá khớp từ epoch 11",
        "Thêm zoom: quá khớp giảm mạnh, val_acc ngang BASE (−0.17)",
      ], { size: 15 }),
    ] },

  { section: "Thực nghiệm", by: "nhan", notes:
    "E5: sigmoid kém nhất mọi run, 56.59%; 7 epoch đầu mạng đoán toàn Happy vì gradient biến mất; LeakyReLU gần như bằng ReLU. E6: bỏ dropout làm quá khớp sớm; dropout 0.4 giảm quá khớp nhưng học chậm; trọng số lớp làm giảm accuracy, cần F1 mới đánh giá đủ.",
    els: [
      title("E5–E6: sigmoid kém nhất, dropout cần thiết"),
      img("E5/figures/curves.png", 0.6, 1.3, 6.4, 2.35),
      text(0.6, 3.7, 6.4, 0.4, "E5: val_loss và val_acc theo epoch; đường xanh lá là sigmoid", { size: 13, color: COL.muted }),
      bullets(0.6, 4.25, 6.4, 2.5, [
        "Sigmoid 56.59%, kém ReLU 9.84 điểm: 7 epoch đầu đứng yên ở 24.94% (đoán toàn Happy)",
        "Đạo hàm sigmoid ≤ 0.25: gradient qua 9 lớp biến mất",
        "LeakyReLU 66.29%, ngang ReLU (−0.14); ELU 64.89%",
      ], { size: 15 }),
      bar(7.3, 1.3, 5.4, 3.6, { title: "E6: val_acc (%)", cats: ["BASE", "Không dropout", "Dropout 0.4", "CE trọng số"],
          vals: [66.43, 62.55, 65.78, 62.19], min: 55, max: 70, major: 5, colors: [COL.ink, COL.coral, COL.teal, COL.coral] }),
      bullets(7.3, 5.05, 5.4, 1.7, [
        "Bỏ dropout: val_loss tăng sau epoch 17",
        "Dropout 0.4: train − val chỉ 1.30, chưa hội tụ",
        "Trọng số lớp: −4.24 accuracy, cần F1 từng lớp",
      ], { size: 15 }),
    ] },

  { section: "Thực nghiệm", by: "truc", notes:
    "Tổng hợp 18 run: chỉ Small vượt baseline, không run nào có val_loss thấp hơn baseline. Xếp theo mức ảnh hưởng: hàm kích hoạt, augmentation, optimizer, BatchNorm, regularization, cuối cùng là kiến trúc. Mô hình cuối vì vậy là baseline với kiến trúc Small.",
    els: [
      title("18 run: chỉ kiến trúc Small vượt baseline"),
      img("figures/tong_hop_val_acc.png", 0.6, 1.3, 7.6, 3.2),
      text(0.6, 4.55, 7.6, 0.45, "Cột gạch chéo: chạm trần 60 epoch, chưa hội tụ (E2-1, E3-1, E5-3, E6-2)", { size: 13, color: COL.muted }),
      box(0.6, 5.15, 7.6, 1.1, COL.card, { radius: true }),
      text(0.8, 5.28, 7.2, 0.9, "Mô hình cuối = baseline + kiến trúc Small (E1-1, +0.59 điểm).\nCác thay đổi khác đều thấp hơn baseline nên không gộp vào.", { size: 16, bold: true, lineSpacing: 1.2 }),
      text(8.5, 1.3, 4.2, 0.45, "Mức ảnh hưởng (chênh val_acc trong nhóm)", { size: 15, bold: true, color: COL.teal }),
      table(8.5, 1.85, 4.2, [2.6, 1.6], [
        ["Nhóm", "Chênh (điểm)"], ["E5 Kích hoạt", "9.84"], ["E4 Augmentation", "6.55"], ["E3 Optimizer", "5.27"],
        ["E2 BatchNorm", "4.65"], ["E6 Regularization", "4.24"], ["E1 Kiến trúc", "1.98"],
      ], { size: 14 }),
    ] },

  { section: "Mô hình cuối và phân tích", by: "truc", notes:
    "Mô hình cuối chạy 3 seed: 66.43 cộng trừ 0.57 trên validation, bằng đúng baseline Medium, nên lợi thế của Small ở E1 nhiều khả năng do seed. Chọn F-1 theo val_loss trước khi đọc test. PrivateTest đánh giá một lần: 66.09%, macro F1 0.6389. Trên ảnh test chưa từng thấy (loại 288 ảnh trùng train) chỉ còn 64.59%.",
    els: [
      title("Mô hình cuối: 66.09% trên PrivateTest, đánh giá một lần"),
      ...stat(0.6, 1.3, 2.9, 1.7, "66.43%", "val_acc 3 seed, ± 0.57", COL.ink),
      ...stat(3.67, 1.3, 2.9, 1.7, "66.09%", "accuracy trên PrivateTest", COL.teal),
      ...stat(6.73, 1.3, 2.9, 1.7, "0.6389", "macro F1 trên PrivateTest", COL.teal),
      ...stat(9.8, 1.3, 2.9, 1.7, "64.59%", "trên ảnh test không trùng train", COL.coral),
      img("final/figures/curves.png", 0.6, 3.3, 7.3, 2.65),
      text(0.6, 6.0, 7.3, 0.75, "Đường học F-0, F-1, F-2 (Small, seed 0, 1, 2). F-1 được chọn vì val_loss nhỏ nhất (0.9515), trước khi đọc test.", { size: 13, color: COL.muted }),
      bullets(8.2, 3.35, 4.5, 3.4, [
        "Trung bình Small = BASE Medium (66.43%): lợi thế +0.59 của E1-1 là may mắn của seed 0",
        "Kém tốt nhất thế giới (71%) 4.91 điểm; ngang báo cáo mẫu",
        "Ảnh test trùng train: đúng 83.33%; ảnh mới: 64.59%",
      ], { size: 15 }),
    ] },

  { section: "Mô hình cuối và phân tích", by: "khang", notes:
    "Lớp conv cuối có 256 filter. Ảnh tổng hợp cho thấy các mẫu hình lặp đều: sọc, lưới, tổ ong, vì mỗi vị trí chỉ nhìn vùng 22x22 pixel. Ảnh thật kích hoạt mạnh nhất cho thấy filter có ý nghĩa: miệng há, mắt mở to, tay che mặt; một filter chỉ phản ứng với ảnh tối.",
    els: [
      title("Filter lớp conv cuối: mẫu hình cục bộ và bộ phận khuôn mặt"),
      img("final/figures/last_conv_filters.png", 0.6, 1.3, 5.0, 5.55),
      img("final/figures/last_conv_top_images.png", 5.75, 1.3, 3.1, 5.55),
      bullets(9.1, 1.4, 3.6, 5.4, [
        "256 filter × (256 kênh × 3×3); 17 filter không kích hoạt trên validation",
        "Trái: ảnh làm từng filter kích hoạt mạnh nhất, mẫu lặp vì mỗi vị trí chỉ nhìn vùng 22×22",
        "Phải: ảnh thật kích hoạt mạnh nhất. Filter 191: miệng há; 155, 187: mắt mở to; 223: tay che mặt; 59: ảnh tối",
      ], { size: 15 }),
    ] },

  { section: "Mô hình cuối và phân tích", by: "nhan", notes:
    "Grad-CAM tại 3 lớp conv, mỗi cảm xúc một ảnh đoán đúng. Conv đầu chỉ bắt cạnh, chưa theo lớp. Conv giữa khoanh vùng mắt, miệng, tay. Conv cuối tập trung vào miệng với Angry, Happy và vào mắt với Surprise, Neutral, Fear, Disgust.",
    els: [
      title("Grad-CAM: lớp conv cuối nhìn vào mắt và miệng"),
      img("final/figures/gradcam.png", 0.6, 1.3, 8.5, 5.45),
      bullets(9.35, 1.4, 3.35, 5.35, [
        "Conv đầu: cạnh, viền, chưa phân biệt lớp",
        "Conv giữa: khoanh vùng mắt, miệng, tay",
        "Conv cuối: miệng với Angry, Happy; mắt với Surprise, Neutral, Fear, Disgust",
        "Mỗi lớp minh họa bằng một ảnh đoán đúng tự tin nhất",
      ], { size: 15 }),
    ] },

  { section: "Mô hình cuối và phân tích", by: "trung", notes:
    "Happy và Surprise dễ nhất. Lỗi tập trung trong nhóm Fear, Sad, Neutral, Angry: Fear bị nhầm thành Sad 110 ảnh, recall Fear chỉ 42.23%. Disgust hay bị nhầm thành Angry. Nhiều ca sai tự tin nhất là ảnh đang cười nhưng mang nhãn Fear, Surprise hay Neutral, tức nhãn nhiễu.",
    els: [
      title("Lỗi tập trung ở nhóm Fear, Sad, Neutral, Angry"),
      img("final/figures/confusion_matrix.png", 0.6, 1.3, 8.6, 3.7),
      text(0.6, 5.05, 8.6, 0.4, "Confusion matrix của mô hình cuối: validation (trái), test (phải); màu theo tỷ lệ trên mỗi hàng", { size: 13, color: COL.muted }),
      bullets(0.6, 5.55, 8.6, 1.25, [
        "Fear → Sad 110/528 ảnh test (20.83%); Disgust → Angry 27.27%",
        "Ca sai tự tin nhất: nhiều ảnh đang cười mang nhãn Fear, Surprise, Neutral (nhãn nhiễu)",
      ], { size: 15 }),
      table(9.5, 1.3, 3.2, [1.8, 1.4], [
        ["Lớp (test)", "F1"], ["Happy", "0.8624"], ["Surprise", "0.7678"], ["Neutral", "0.6453"],
        ["Disgust", "0.6139"], ["Angry", "0.5946"], ["Sad", "0.5231"], ["Fear", "0.4651"],
      ], { size: 14 }),
      text(9.5, 5.55, 3.2, 1.2, "Fear khó nhất: recall 42.23%", { size: 16, bold: true, color: COL.coral }),
    ] },

  { section: "Kết luận", dark: true, by: "truc", notes:
    "Kết luận và phân công. Mỗi thành viên 25%. Hướng tiếp theo: huấn luyện lâu hơn các cấu hình chưa hội tụ, thử mô hình học trước như VGG16, ghép nhiều mô hình; mọi lựa chọn vẫn chỉ dựa trên validation.",
    els: [
      text(0.6, 0.38, 12.1, 0.8, "Kết luận và phân công", { size: 30, bold: true, color: COL.white, valign: "middle" }),
      bullets(0.6, 1.4, 6.4, 3.4, [
        "Mô hình cuối: 66.43% ± 0.57 trên validation, 66.09% trên PrivateTest, ngang báo cáo mẫu",
        "Augmentation và hàm kích hoạt ảnh hưởng mạnh nhất; kiến trúc ít ảnh hưởng",
        "Khởi tạo He quyết định mạng có học được hay không",
        "Filter và Grad-CAM: mô hình dựa vào mắt và miệng",
        "Hạn chế: 1 seed mỗi thực nghiệm, nhãn nhiễu, 8% ảnh test trùng train",
      ], { size: 16, color: COL.white }),
      table(7.3, 1.4, 5.4, [2.55, 2.1, 0.75], [
        ["Thành viên", "Phần chính", "Tỷ lệ"],
        ["Huỳnh Ngọc Trúc", "Dữ liệu, train, mô hình cuối", "25%"],
        ["Phạm Trần Minh Khang", "Kiến trúc, BN, filter", "25%"],
        ["Tạ Quang Trung", "Optimizer, augmentation, lỗi", "25%"],
        ["Ngô Nguyễn Thành Nhân", "Kích hoạt, dropout, Grad‑CAM", "25%"],
      ], { size: 14, light: true }),
      box(0.6, 4.95, 12.1, 1.75, "33355A", { radius: true }),
      text(0.85, 5.1, 11.6, 0.45, "Hướng tiếp theo", { size: 18, bold: true, color: COL.amber }),
      text(0.85, 5.58, 11.6, 1.0, "Huấn luyện lâu hơn các cấu hình chưa hội tụ (dropout 0.4, BatchNorm) · thử mô hình học trước như VGG16 (E7) · ghép nhiều mô hình.\nMọi lựa chọn vẫn chỉ dựa trên validation: PrivateTest đã dùng một lần.", { size: 15, color: COL.white, lineSpacing: 1.2 }),
    ] },
];

// ------------------------------------------------------------------ PowerPoint
function buildPptx() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = "Nhận dạng cảm xúc khuôn mặt trên FER2013 bằng CNN";
  pres.author = "Nhóm 05";
  pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
  let section = null;
  SLIDES.forEach((sd, i) => {
    if (sd.section !== section) { pres.addSection({ title: sd.section }); section = sd.section; }
    const slide = pres.addSlide({ sectionTitle: sd.section });
    slide.background = { color: sd.dark ? COL.ink : COL.white };
    for (const e of sd.els) addPptx(pres, slide, e, sd.dark);
    footer(sd, i).forEach((e) => addPptx(pres, slide, e, sd.dark));
    slide.addNotes(sd.notes);
  });
  return pres.writeFile({ fileName: OUT + ".pptx" });
}

function footer(sd, i) {
  const c = sd.dark ? COL.soft : COL.muted;
  return [text(0.6, 7.02, 8, 0.3, `Trình bày: ${MEMBERS[sd.by]}`, { size: 11, color: c }),
          text(10.7, 7.02, 2.0, 0.3, `${i + 1} / ${SLIDES.length}`, { size: 11, color: c, align: "right" })];
}

function addPptx(pres, slide, e, dark) {
  const pos = { x: e.x, y: e.y, w: e.w, h: e.h };
  if (e.t === "rect") {
    slide.addShape(e.radius ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE,
                   { ...pos, fill: { color: dark ? "33355A" : e.fill }, line: { type: "none" }, rectRadius: e.radius ? 0.08 : undefined });
  } else if (e.t === "circle") {
    slide.addShape(pres.shapes.OVAL, { ...pos, fill: { color: e.fill }, line: { type: "none" } });
  } else if (e.t === "text") {
    slide.addText(e.s.split("\n").map((line, k, arr) => ({ text: line, options: { breakLine: k < arr.length - 1 } })), {
      ...pos, fontFace: FONT, fontSize: e.size, bold: !!e.bold, italic: !!e.italic, color: e.color,
      align: e.align || "left", valign: e.valign || "top", margin: 0, isTextBox: true,
      lineSpacingMultiple: e.lineSpacing || 1.0, fit: "none" });
  } else if (e.t === "bullets") {
    slide.addText(e.items.map((s, k) => ({ text: s, options: { bullet: { indent: 16 }, breakLine: k < e.items.length - 1 } })), {
      ...pos, fontFace: FONT, fontSize: e.size, color: e.color, valign: "top", margin: 0, isTextBox: true,
      paraSpaceAfter: 8, fit: "none" });
  } else if (e.t === "img") {
    slide.addImage({ path: e.file, ...pos });
  } else if (e.t === "bar") {
    slide.addChart(pres.charts.BAR, [{ name: e.title, labels: e.cats, values: e.vals }], {
      ...pos, barDir: "col", chartColors: e.colors, showTitle: true, title: e.title, titleFontSize: 14,
      titleColor: COL.ink, titleFontFace: FONT, showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.00", dataLabelFontSize: 12, dataLabelFontFace: FONT, dataLabelColor: COL.ink,
      valAxisMinVal: e.min, valAxisMaxVal: e.max, valAxisMajorUnit: e.major, valAxisLabelFontSize: 11, catAxisLabelFontSize: 12,
      valAxisLabelFontFace: FONT, catAxisLabelFontFace: FONT, valAxisLabelColor: COL.muted, catAxisLabelColor: COL.ink,
      valGridLine: { color: "E3E6EB", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 60 });
  } else if (e.t === "table") {
    const fg = e.light ? COL.white : COL.ink, line = e.light ? "6B6E8F" : "C9CEDA";
    const rows = e.rows.map((r, ri) => r.map((c, ci) => ({ text: c, options: {
      bold: ri === 0, color: fg, fill: { color: ri === 0 ? (e.light ? "33355A" : COL.card) : (e.light ? COL.ink : COL.white) },
      align: ci > 0 && /^[\d.%]+$/.test(c) ? "right" : "left" } })));
    slide.addTable(rows, { x: e.x, y: e.y, w: e.w, colW: e.colW, fontFace: FONT, fontSize: e.size,
                           border: { type: "solid", pt: 0.75, color: line }, margin: [3, 5, 3, 5], valign: "middle" });
  }
}

// ------------------------------------------------------------------ HTML -> PDF (cùng tọa độ)
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function htmlEl(e, dark) {
  const pos = `left:${e.x}in;top:${e.y}in;width:${e.w}in;height:${e.h}in;`;
  if (e.t === "rect") return `<div class="el" style="${pos}background:#${dark ? "33355A" : e.fill};border-radius:${e.radius ? "0.08in" : 0}"></div>`;
  if (e.t === "circle") return `<div class="el" style="${pos}background:#${e.fill};border-radius:50%"></div>`;
  if (e.t === "text") {
    const jc = { top: "flex-start", middle: "center", bottom: "flex-end" }[e.valign || "top"];
    return `<div class="el tx" style="${pos}justify-content:${jc};font-size:${e.size}pt;color:#${e.color};font-weight:${e.bold ? 700 : 400};` +
           `font-style:${e.italic ? "italic" : "normal"};text-align:${e.align || "left"};line-height:${1.17 * (e.lineSpacing || 1)}">` +
           e.s.split("\n").map((l) => `<div>${esc(l)}</div>`).join("") + `</div>`;
  }
  if (e.t === "bullets") return `<ul class="el" style="${pos}font-size:${e.size}pt;color:#${e.color}">` +
    e.items.map((s) => `<li>${esc(s)}</li>`).join("") + `</ul>`;
  if (e.t === "img") return `<img class="el" style="${pos}" src="${new URL("file:///" + e.file.replace(/\\/g, "/")).href}">`;
  if (e.t === "bar") return `<div class="el" style="${pos}">${barSvg(e)}</div>`;
  if (e.t === "table") {
    const fg = e.light ? "#FFFFFF" : "#" + COL.ink, line = e.light ? "#6B6E8F" : "#C9CEDA";
    const head = e.light ? "#33355A" : "#" + COL.card, body = e.light ? "#" + COL.ink : "#FFFFFF";
    return `<table class="el" style="left:${e.x}in;top:${e.y}in;width:${e.w}in;font-size:${e.size}pt;color:${fg}">` +
      e.rows.map((r, ri) => "<tr>" + r.map((c, ci) => `<td style="width:${e.colW[ci]}in;border:0.75pt solid ${line};` +
        `background:${ri === 0 ? head : body};font-weight:${ri === 0 ? 700 : 400};text-align:${ci > 0 && /^[\d.%]+$/.test(c) ? "right" : "left"}">${esc(c)}</td>`).join("") + "</tr>").join("") + `</table>`;
  }
  return "";
}

function barSvg(e) {  // biểu đồ cột tương tự biểu đồ native của PowerPoint
  const Wp = e.w * 96, Hp = e.h * 96, top = 46, bottom = 46, left = 40, right = 10;
  const ph = Hp - top - bottom, pw = Wp - left - right, n = e.vals.length, slot = pw / n, bw = slot / 1.6;
  const yv = (v) => top + ph * (1 - (v - e.min) / (e.max - e.min));
  let s = `<svg width="${Wp}" height="${Hp}" font-family="${FONT}">`;
  s += `<text x="${Wp / 2}" y="22" text-anchor="middle" font-size="18.7" font-weight="700" fill="#${COL.ink}">${esc(e.title)}</text>`;
  const step = e.major, nStep = Math.round((e.max - e.min) / step);
  for (let k = 0; k <= nStep; k++) {
    const v = e.min + k * step, y = yv(v);
    s += `<line x1="${left}" x2="${Wp - right}" y1="${y}" y2="${y}" stroke="#E3E6EB" stroke-width="0.7"/>`;
    s += `<text x="${left - 6}" y="${y + 4}" text-anchor="end" font-size="14.7" fill="#${COL.muted}">${+v.toFixed(2)}</text>`;
  }
  e.vals.forEach((v, i) => {
    const x = left + slot * i + (slot - bw) / 2, y = yv(v);
    s += `<rect x="${x}" y="${y}" width="${bw}" height="${top + ph - y}" fill="#${e.colors[i]}"/>`;
    s += `<text x="${x + bw / 2}" y="${y - 5}" text-anchor="middle" font-size="16" fill="#${COL.ink}">${v.toFixed(2)}</text>`;
    s += `<text x="${x + bw / 2}" y="${top + ph + 20}" text-anchor="middle" font-size="16" fill="#${COL.ink}">${esc(e.cats[i])}</text>`;
  });
  return s + `</svg>`;
}

function buildPdf() {
  const css = `@page { size: ${W}in ${H}in; margin: 0 }
    body { margin: 0; font-family: ${FONT}, sans-serif }
    .slide { position: relative; width: ${W}in; height: ${H}in; overflow: hidden; page-break-after: always }
    .el { position: absolute; box-sizing: border-box; margin: 0 }
    .tx { display: flex; flex-direction: column; overflow: visible }
    ul.el { padding-left: 0.22in; line-height: 1.17 } ul.el li { margin-bottom: 8pt }
    table.el { border-collapse: collapse } td { padding: 3pt 5pt; line-height: 1.15 }`;
  const pages = SLIDES.map((sd, i) => `<div class="slide" style="background:#${sd.dark ? COL.ink : COL.white}">` +
    [...sd.els, ...footer(sd, i)].map((e) => htmlEl(e, sd.dark)).join("") + `</div>`).join("");
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "slide-"));
  const html = path.join(tmp, "slides.html");
  fs.writeFileSync(html, `<!doctype html><html lang="vi"><head><meta charset="utf-8"><style>${css}</style></head><body>${pages}</body></html>`);
  const browser = [String.raw`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`,
                   String.raw`C:\Program Files\Microsoft\Edge\Application\msedge.exe`,
                   String.raw`C:\Program Files\Google\Chrome\Application\chrome.exe`].find((p) => fs.existsSync(p));
  if (!browser) throw new Error("Không tìm thấy Edge hoặc Chrome để in PDF");
  execFileSync(browser, ["--headless", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                         `--print-to-pdf=${OUT}.pdf`, new URL("file:///" + html.replace(/\\/g, "/")).href], { stdio: "ignore" });
}

// Bỏ các mục "thư mục" trong zip của .pptx: file có chúng từng không mở được khi đưa qua
// dịch vụ đám mây (OneDrive, Google Drive). Thứ tự các file còn lại giữ nguyên.
async function dropFolderEntries(file) {
  const JSZip = require("jszip");
  const src = await JSZip.loadAsync(fs.readFileSync(file));
  const out = new JSZip();
  for (const [name, entry] of Object.entries(src.files)) {
    if (entry.dir) continue;
    out.file(name, await entry.async("uint8array"), { createFolders: false, date: entry.date });
  }
  fs.writeFileSync(file, await out.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await buildPptx();
  await dropFolderEntries(OUT + ".pptx");
  buildPdf();
  for (const ext of [".pptx", ".pdf"]) console.log(`Đã ghi ${OUT + ext} (${Math.round(fs.statSync(OUT + ext).size / 1024)} KB)`);
})();

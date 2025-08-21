#working

import os
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import colorsys
import copy

# === Globals ===
model = None
CLASS_NAMES = {}
CLASS_IDS = []
image_dir = ''
label_dir = ''
image_files = []
current_image_index = 0
boxes = []  # (cls_id, xc, yc, w, h)
history = []  # for undo support


def get_class_color(cls_id, total=None):
    total = total or len(CLASS_IDS)
    hue = (cls_id * 1.0 / max(total, 1)) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 1, 1)
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'


class LabelMeYoloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAAD Labeling Tool")
        self.root.geometry("1280x800")
        self.root.configure(bg="#f8f9fa")

        toolbar = tk.Frame(self.root, bg="#f0f2f5", bd=1, relief="solid")
        toolbar.pack(side=tk.LEFT, fill="y", padx=6, pady=6)

        def styled_button(text, command):
            return tk.Button(toolbar, text=text, command=command,
                             font=("Segoe UI", 11, "bold"), bg="#1d3557", fg="white",
                             relief="flat", padx=14, pady=10, cursor="hand2",
                             activebackground="#457b9d", activeforeground="white",
                             bd=0, highlightthickness=0)

        styled_button("Load Weights", self.load_weights).pack(pady=8, fill='x')
        styled_button("Load Images", self.load_images).pack(pady=8, fill='x')
        styled_button("Load Labels", self.load_labels).pack(pady=8, fill='x')
        styled_button("Prev", self.prev_image).pack(pady=8, fill='x')
        styled_button("Next", self.next_image).pack(pady=8, fill='x')
        styled_button("Save", self.save_labels).pack(pady=8, fill='x')
        styled_button("Zoom In", self.zoom_in).pack(pady=8, fill='x')
        styled_button("Zoom Out", self.zoom_out).pack(pady=8, fill='x')
        styled_button("Clear All", self.clear_boxes).pack(pady=8, fill='x')

        self.auto_save_enabled = True
        self.auto_save_button = styled_button("Auto Save: ON", self.toggle_auto_save)
        self.auto_save_button.pack(pady=8, fill='x')

        # === Layout ===
        self.layout_frame = tk.Frame(self.root, bg="#f8f9fa")
        self.layout_frame.pack(side=tk.LEFT, fill="both", expand=True)

        self.canvas = tk.Canvas(self.layout_frame, cursor="cross", bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill="both", expand=True)

        self.sidebar = tk.Frame(self.layout_frame, width=260, bg="#e9ecef", padx=8, bd=1, relief="solid")
        self.sidebar.pack(side=tk.RIGHT, fill="y")

        tk.Label(self.sidebar, text="Class List", font=("Segoe UI", 12, "bold"),
                 bg="#e9ecef", fg="#1d3557", anchor="center", pady=8).pack(pady=(10, 6), fill='x')

        self.class_list_frame = tk.Frame(self.sidebar, bg="#1d3557", bd=0)
        self.class_list_frame.pack(fill="both", padx=8)

        self.scrollbar = tk.Scrollbar(self.class_list_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.class_listbox = tk.Listbox(
            self.class_list_frame,
            yscrollcommand=self.scrollbar.set,
            font=("Segoe UI", 11),
            activestyle="none",
            bd=0,
            highlightthickness=0,
            relief="flat",
            selectbackground="#457b9d",
            selectforeground="white",
            bg="#1d3557",
            fg="white"
        )
        self.class_listbox.pack(fill="both", expand=True, padx=8, pady=4)
        self.scrollbar.config(command=self.class_listbox.yview)

        tk.Label(self.sidebar, text="Box List", font=("Segoe UI", 12, "bold"),
                 bg="#e9ecef", fg="#1d3557", anchor="center", pady=8).pack(pady=(10, 6), fill='x')

        box_list_frame = tk.Frame(self.sidebar, bg="#e9ecef")
        box_list_frame.pack(fill="x", padx=8)

        self.box_listbox = tk.Listbox(
            box_list_frame,
            height=6,
            font=("Segoe UI", 10),
            selectbackground="#e63946",
            selectforeground="white",
            activestyle="none",
            relief="flat",
            bg="#ffffff",
            fg="#212529"
        )
        self.box_listbox.pack(side=tk.LEFT, fill="both", expand=True)

        box_scrollbar = tk.Scrollbar(box_list_frame, command=self.box_listbox.yview)
        box_scrollbar.pack(side=tk.RIGHT, fill="y")
        self.box_listbox.config(yscrollcommand=box_scrollbar.set)

        btn_frame = tk.Frame(self.sidebar, bg="#e9ecef")
        btn_frame.pack(pady=(6, 10), fill='x')

        tk.Button(btn_frame, text="Edit", command=self.edit_selected_box,
                  font=("Segoe UI", 10), bg="#1d3557", fg="white",
                  relief="flat", padx=6, pady=4).pack(side=tk.LEFT, padx=4, fill="x", expand=True)

        tk.Button(btn_frame, text="Delete", command=self.delete_selected_box,
                  font=("Segoe UI", 10), bg="#e63946", fg="white",
                  relief="flat", padx=6, pady=4).pack(side=tk.LEFT, padx=4, fill="x", expand=True)

        self.frame_label = tk.Label(self.sidebar, text="", font=("Segoe UI", 10), bg="#e9ecef", fg="#6c757d")
        self.frame_label.pack(pady=(5, 4))

        # === Key Bindings ===
        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("s", self.save_labels)
        self.root.bind("<Control-z>", self.undo_box)
        self.root.bind("+", self.zoom_in)
        self.root.bind("-", self.zoom_out)

        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<Button-3>", self.right_click_box)
        self.canvas.bind("<MouseWheel>", self.mouse_zoom)
        self.canvas.bind("<Button-4>", self.mouse_zoom_linux)
        self.canvas.bind("<Button-5>", self.mouse_zoom_linux)

        self.start_x = self.start_y = 0
        self.temp_rect = None
        self.tk_boxes = []
        self.zoom_scale = 1.0
        self.image_offset = (0, 0)
        self.tk_image = None

    def load_weights(self):
        global model, CLASS_NAMES, CLASS_IDS
        path = filedialog.askopenfilename(filetypes=[("YOLO Weights", "*.pt")])
        if path:
            try:
                model = YOLO(path)
                CLASS_NAMES = model.names
                CLASS_IDS.clear()
                CLASS_IDS.extend(CLASS_NAMES.keys())
                self.class_listbox.delete(0, tk.END)
                for cls_id in CLASS_IDS:
                    self.class_listbox.insert(tk.END, f"{cls_id}: {CLASS_NAMES[cls_id]}")
                messagebox.showinfo("Success", f"Model loaded from:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def load_images(self):
        global image_dir, image_files, current_image_index
        path = filedialog.askdirectory()
        if path:
            image_dir = path
            image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            current_image_index = 0
            self.load_image()

    def load_labels(self):
        global label_dir
        path = filedialog.askdirectory()
        if path:
            label_dir = path
            self.load_image()

    def toggle_auto_save(self):
        self.auto_save_enabled = not self.auto_save_enabled
        self.auto_save_button.config(text=f"Auto Save: {'ON' if self.auto_save_enabled else 'OFF'}")

    def load_image(self):
        global boxes, current_image_index
        if not image_files or current_image_index >= len(image_files):
            return
        boxes.clear()
        history.clear()
        self.tk_boxes.clear()
        filename = image_files[current_image_index]
        self.image_path = os.path.join(image_dir, filename)
        self.label_path = os.path.join(label_dir, filename.rsplit(".", 1)[0] + ".txt")

        image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        self.original_image = Image.fromarray(image)

        self.frame_label.config(text=f"Image: {current_image_index + 1}/{len(image_files)}")
        self.root.after(1, self.apply_zoom)

        if os.path.exists(self.label_path):
            with open(self.label_path, 'r') as f:
                for line in f:
                    cls_id, xc, yc, w, h = map(float, line.strip().split())
                    boxes.append((int(cls_id), xc, yc, w, h))

    def apply_zoom(self):
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if self.zoom_scale == 1.0:
            scale_x = canvas_width / self.original_image.width
            scale_y = canvas_height / self.original_image.height
            self.zoom_scale = min(scale_x, scale_y, 1.0)
        scaled_width = int(self.original_image.width * self.zoom_scale)
        scaled_height = int(self.original_image.height * self.zoom_scale)
        image_resized = self.original_image.resize((scaled_width, scaled_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(image_resized)
        x_offset = max((canvas_width - scaled_width) // 2, 0)
        y_offset = max((canvas_height - scaled_height) // 2, 0)
        self.image_offset = (x_offset, y_offset)
        self.canvas.delete("image")
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_image, tags="image")
        self.redraw_boxes()

    def draw_box(self, cls_id, xc, yc, w, h):
        scaled_width = int(self.original_image.width * self.zoom_scale)
        scaled_height = int(self.original_image.height * self.zoom_scale)
        x_offset, y_offset = self.image_offset
        x1 = int((xc - w / 2) * scaled_width) + x_offset
        y1 = int((yc - h / 2) * scaled_height) + y_offset
        x2 = int((xc + w / 2) * scaled_width) + x_offset
        y2 = int((yc + h / 2) * scaled_height) + y_offset
        color = get_class_color(cls_id)
        rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="box")
        label = self.canvas.create_text(x1 + 4, y1 - 8, text=CLASS_NAMES[cls_id], anchor="nw", fill=color,
                                        font=("Helvetica", 10, "bold"), tags="box")
        self.tk_boxes.append((rect, label))

    def redraw_boxes(self):
        self.canvas.delete("box")
        for box in boxes:
            self.draw_box(*box)
        self.update_box_list()

    def update_box_list(self):
        self.box_listbox.delete(0, tk.END)
        for i, (cls_id, xc, yc, w, h) in enumerate(boxes):
            self.box_listbox.insert(tk.END, f"{i}: {CLASS_NAMES[cls_id]} ({cls_id})")

    def mouse_down(self, e): self.start_x, self.start_y = e.x, e.y; self.temp_rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="green")
    def mouse_drag(self, e): self.canvas.coords(self.temp_rect, self.start_x, self.start_y, e.x, e.y)

    def mouse_up(self, e):
        if not CLASS_IDS: return
        x1, y1 = min(self.start_x, e.x), min(self.start_y, e.y)
        x2, y2 = max(self.start_x, e.x), max(self.start_y, e.y)
        x_offset, y_offset = self.image_offset
        selected = self.class_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "Select a class before drawing.")
            self.canvas.delete(self.temp_rect)
            return
        cls_id = int(self.class_listbox.get(selected[0]).split(":")[0])
        x1 -= x_offset; x2 -= x_offset; y1 -= y_offset; y2 -= y_offset
        xc = ((x1 + x2) / 2) / (self.original_image.width * self.zoom_scale)
        yc = ((y1 + y2) / 2) / (self.original_image.height * self.zoom_scale)
        w = abs(x2 - x1) / (self.original_image.width * self.zoom_scale)
        h = abs(y2 - y1) / (self.original_image.height * self.zoom_scale)
        history.append(copy.deepcopy(boxes))  # Add current state to history
        boxes.append((cls_id, xc, yc, w, h))
        self.draw_box(cls_id, xc, yc, w, h)
        self.update_box_list()
        self.canvas.delete(self.temp_rect)

    def right_click_box(self, e):
        x, y = e.x, e.y
        x_offset, y_offset = self.image_offset
        for i, (cls_id, xc, yc, w, h) in enumerate(boxes):
            scaled_width = int(self.original_image.width * self.zoom_scale)
            scaled_height = int(self.original_image.height * self.zoom_scale)
            x1 = int((xc - w / 2) * scaled_width) + x_offset
            y1 = int((yc - h / 2) * scaled_height) + y_offset
            x2 = int((xc + w / 2) * scaled_width) + x_offset
            y2 = int((yc + h / 2) * scaled_height) + y_offset
            if x1 <= x <= x2 and y1 <= y <= y2:
                if messagebox.askyesno("Delete", f"Delete box for {CLASS_NAMES[cls_id]}?"):
                    boxes.pop(i)
                    self.redraw_boxes()
                return

    def undo_box(self, event=None):
        global boxes
        if history:
            boxes.clear()
            boxes.extend(history.pop())
            self.redraw_boxes()

    def save_labels(self, event=None):
        if not hasattr(self, 'label_path'): return
        with open(self.label_path, 'w') as f:
            for cls_id, xc, yc, w, h in boxes:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        print(f"[Saved] {self.label_path}")

    def clear_boxes(self): boxes.clear(); self.redraw_boxes()
    def next_image(self, event=None): global current_image_index; self.save_labels() if self.auto_save_enabled else None; current_image_index += 1; self.zoom_scale = 1.0; self.load_image()
    def prev_image(self, event=None): global current_image_index; self.save_labels() if self.auto_save_enabled else None; current_image_index = max(current_image_index - 1, 0); self.zoom_scale = 1.0; self.load_image()
    def zoom_in(self, event=None): self.zoom_scale *= 1.2; self.apply_zoom()
    def zoom_out(self, event=None): self.zoom_scale /= 1.2; self.apply_zoom()
    def mouse_zoom(self, e): self.zoom_in() if e.delta > 0 else self.zoom_out()
    def mouse_zoom_linux(self, e): self.zoom_in() if e.num == 4 else self.zoom_out()
    def edit_selected_box(self):
        selection = self.box_listbox.curselection()
        if not selection: return
        idx = selection[0]
        old_cls_id, xc, yc, w, h = boxes[idx]
        new_cls_id = simpledialog.askinteger("Edit Box", f"Change class ID (was {old_cls_id}):", initialvalue=old_cls_id)
        if new_cls_id is not None and new_cls_id in CLASS_IDS:
            boxes[idx] = (new_cls_id, xc, yc, w, h)
            self.redraw_boxes()
    def delete_selected_box(self):
        selection = self.box_listbox.curselection()
        if not selection: return
        idx = selection[0]
        if messagebox.askyesno("Delete Box", f"Are you sure you want to delete this box?"):
            boxes.pop(idx)
            self.redraw_boxes()

if __name__ == "__main__":
    root = tk.Tk()

    # Set custom window icon using uploaded PNG
    try:
        icon_path = "/home/imad/Research/vid/WENS.png"
        icon_img = tk.PhotoImage(file=icon_path)
        root.iconphoto(False, icon_img)
    except Exception as e:
        print(f"[Warning] Failed to load icon: {e}")

    app = LabelMeYoloGUI(root)
    root.mainloop()

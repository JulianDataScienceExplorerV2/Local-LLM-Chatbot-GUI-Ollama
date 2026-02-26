import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import time
import re
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
import requests

# ──────────────────────────────────────────────
#  State definition
# ──────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]


# ──────────────────────────────────────────────
#  Palette & fonts
# ──────────────────────────────────────────────
COLORS = {
    "bg":           "#0D1117",   # main background
    "sidebar":      "#161B22",   # sidebar
    "surface":      "#1C2128",   # chat area bg
    "bubble_user":  "#1F6FEB",   # user bubble
    "bubble_ai":    "#21262D",   # AI bubble
    "input_bg":     "#21262D",   # input box bg
    "border":       "#30363D",   # borders
    "accent":       "#388BFD",   # primary accent (blue)
    "accent_glow":  "#1F6FEB",   # button hover
    "text":         "#E6EDF3",   # main text
    "text_dim":     "#8B949E",   # muted/secondary text
    "text_code":    "#79C0FF",   # code colour
    "dot1":         "#388BFD",
    "dot2":         "#58A6FF",
    "dot3":         "#79C0FF",
    "success":      "#3FB950",
    "warning":      "#D29922",
    "danger":       "#F85149",
    "header":       "#0D1117",
}

FONTS = {
    "title":    ("Segoe UI Semibold", 14),
    "subtitle": ("Segoe UI", 9),
    "body":     ("Segoe UI", 11),
    "body_bold":("Segoe UI Semibold", 11),
    "small":    ("Segoe UI", 9),
    "mono":     ("Consolas", 10),
    "icon":     ("Segoe UI Emoji", 14),
}

# ──────────────────────────────────────────────
#  Helper: rounded rectangle on canvas
# ──────────────────────────────────────────────
def rounded_rect(canvas, x1, y1, x2, y2, r=12, **kw):
    pts = [
        x1+r, y1,  x2-r, y1,
        x2, y1,    x2, y1+r,
        x2, y2-r,  x2, y2,
        x2-r, y2,  x1+r, y2,
        x1, y2,    x1, y2-r,
        x1, y1+r,  x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


# ──────────────────────────────────────────────
#  Custom scrollable frame
# ──────────────────────────────────────────────
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, bg, **kw):
        super().__init__(parent, bg=bg, **kw)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                      bg=COLORS["sidebar"], troughcolor=COLORS["bg"],
                                      activebackground=COLORS["accent"])
        self.inner = tk.Frame(self.canvas, bg=bg)

        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.inner_id, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def scroll_bottom(self):
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)


# ──────────────────────────────────────────────
#  Typing indicator (animated dots)
# ──────────────────────────────────────────────
class TypingIndicator(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=COLORS["bubble_ai"], **kw)
        self._dots = []
        self._job = None
        self._step = 0

        prefix = tk.Label(self, text="✦ Ollama  ", bg=COLORS["bubble_ai"],
                          fg=COLORS["accent"], font=FONTS["small"])
        prefix.pack(side="left")

        for _ in range(3):
            d = tk.Label(self, text="●", bg=COLORS["bubble_ai"],
                         fg=COLORS["text_dim"], font=("Segoe UI", 9))
            d.pack(side="left", padx=1)
            self._dots.append(d)

    def start(self):
        self._animate()

    def stop(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None

    def _animate(self):
        colors = [COLORS["dot1"], COLORS["dot2"], COLORS["dot3"]]
        for i, dot in enumerate(self._dots):
            dot.config(fg=colors[i] if i == self._step % 3 else COLORS["text_dim"])
        self._step += 1
        self._job = self.after(450, self._animate)


# ──────────────────────────────────────────────
#  Chat bubble widget
# ──────────────────────────────────────────────
class ChatBubble(tk.Frame):
    def __init__(self, parent, role, text, timestamp, **kw):
        super().__init__(parent, bg=COLORS["surface"], **kw)
        is_user = role == "user"

        self.columnconfigure(0, weight=1)

        # Avatar + meta row
        meta_frame = tk.Frame(self, bg=COLORS["surface"])
        if is_user:
            meta_frame.pack(anchor="e", padx=(60, 14), pady=(10, 2))
            avatar_text = "You"
            avatar_color = COLORS["accent"]
        else:
            meta_frame.pack(anchor="w", padx=(14, 60), pady=(10, 2))
            avatar_text = "✦ Ollama"
            avatar_color = COLORS["text_dim"]

        tk.Label(meta_frame, text=avatar_text, bg=COLORS["surface"],
                 fg=avatar_color, font=FONTS["small"]).pack(side="left")
        tk.Label(meta_frame, text=f"  {timestamp}", bg=COLORS["surface"],
                 fg=COLORS["text_dim"], font=FONTS["small"]).pack(side="left")

        # Bubble
        bubble_bg = COLORS["bubble_user"] if is_user else COLORS["bubble_ai"]
        bubble_fg = COLORS["text"]

        bubble = tk.Frame(self, bg=bubble_bg,
                          highlightthickness=1,
                          highlightbackground=COLORS["border"] if not is_user else bubble_bg)
        if is_user:
            bubble.pack(anchor="e", padx=(60, 14), pady=(0, 6))
        else:
            bubble.pack(anchor="w", padx=(14, 60), pady=(0, 6))

        # Render text with basic markdown-like formatting
        self._render_text(bubble, text, bubble_bg, bubble_fg, is_user)

    def _render_text(self, parent, text, bg, fg, is_user):
        """Render text with basic markdown: **bold**, `code`, ```blocks```."""
        # Split into code blocks vs normal lines
        parts = re.split(r'(```[\s\S]*?```)', text)

        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                # Code block
                code = part[3:-3].strip()
                # Remove language hint if present
                if "\n" in code:
                    first_line, rest = code.split("\n", 1)
                    if re.match(r'^[a-zA-Z]+$', first_line.strip()):
                        code = rest
                code_frame = tk.Frame(parent, bg="#0D1117",
                                      highlightthickness=1,
                                      highlightbackground=COLORS["border"])
                code_frame.pack(fill="x", padx=10, pady=6)
                tk.Label(code_frame, text=code, bg="#0D1117", fg=COLORS["text_code"],
                         font=FONTS["mono"], justify="left", anchor="w",
                         wraplength=520).pack(padx=10, pady=8, anchor="w")
            else:
                # Normal text with inline formatting
                if part.strip():
                    lines = part.strip().split("\n")
                    for line in lines:
                        if not line.strip():
                            tk.Frame(parent, bg=bg, height=4).pack()
                            continue
                        self._render_inline(parent, line.strip(), bg, fg)

    def _render_inline(self, parent, line, bg, fg):
        """Render a line with inline bold/code formatting."""
        # Simple approach: just find **...** and `...`
        frame = tk.Frame(parent, bg=bg)
        frame.pack(fill="x", padx=12, pady=1, anchor="w")

        pattern = re.compile(r'(\*\*(.+?)\*\*|`([^`]+)`)')
        pos = 0
        col = 0
        for m in pattern.finditer(line):
            # Text before match
            before = line[pos:m.start()]
            if before:
                tk.Label(frame, text=before, bg=bg, fg=fg,
                         font=FONTS["body"], wraplength=520,
                         justify="left").grid(row=0, column=col, sticky="w")
                col += 1
            if m.group(1).startswith("**"):
                tk.Label(frame, text=m.group(2), bg=bg, fg=fg,
                         font=FONTS["body_bold"], wraplength=520,
                         justify="left").grid(row=0, column=col, sticky="w")
            else:
                tk.Label(frame, text=m.group(3), bg="#0D1117", fg=COLORS["text_code"],
                         font=FONTS["mono"], wraplength=520,
                         justify="left").grid(row=0, column=col, sticky="w")
            col += 1
            pos = m.end()

        # Remaining text
        tail = line[pos:]
        if tail:
            tk.Label(frame, text=tail, bg=bg, fg=fg,
                     font=FONTS["body"], wraplength=520,
                     justify="left").grid(row=0, column=col, sticky="w")


# ──────────────────────────────────────────────
#  Icon button (flat, no border)
# ──────────────────────────────────────────────
class IconButton(tk.Label):
    def __init__(self, parent, text, command, tooltip_text="", **kw):
        defaults = dict(bg=COLORS["sidebar"], fg=COLORS["text_dim"],
                        font=FONTS["icon"], cursor="hand2", padx=6, pady=4)
        defaults.update(kw)
        super().__init__(parent, text=text, **defaults)
        self._cmd = command
        self._normal_bg = defaults["bg"]
        self._tooltip_text = tooltip_text

        self.bind("<Button-1>", lambda _: command())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _):
        self.config(fg=COLORS["text"], bg=COLORS["surface"])

    def _on_leave(self, _):
        self.config(fg=COLORS["text_dim"], bg=self._normal_bg)


# ──────────────────────────────────────────────
#  Session item in sidebar
# ──────────────────────────────────────────────
class SessionItem(tk.Frame):
    def __init__(self, parent, title, on_click, on_delete, **kw):
        super().__init__(parent, bg=COLORS["sidebar"], cursor="hand2", **kw)
        self._on_click = on_click
        self._normal_bg = COLORS["sidebar"]
        self._active = False

        self.lbl = tk.Label(self, text=title, bg=COLORS["sidebar"],
                            fg=COLORS["text"], font=FONTS["small"],
                            anchor="w", wraplength=160, justify="left")
        self.lbl.pack(side="left", fill="x", expand=True, padx=(8, 0), pady=6)

        self.del_btn = tk.Label(self, text="✕", bg=COLORS["sidebar"],
                                fg=COLORS["text_dim"], font=FONTS["small"],
                                cursor="hand2")
        self.del_btn.pack(side="right", padx=(0, 6))
        self.del_btn.bind("<Button-1>", lambda _: on_delete())

        for w in (self, self.lbl):
            w.bind("<Button-1>", lambda _: on_click())
            w.bind("<Enter>", self._hover_on)
            w.bind("<Leave>", self._hover_off)

    def set_active(self, active):
        self._active = active
        color = COLORS["surface"] if active else COLORS["sidebar"]
        self._normal_bg = color
        self.config(bg=color)
        self.lbl.config(bg=color)
        self.del_btn.config(bg=color)

    def _hover_on(self, _):
        if not self._active:
            for w in (self, self.lbl, self.del_btn):
                w.config(bg="#1C2128")

    def _hover_off(self, _):
        bg = COLORS["surface"] if self._active else COLORS["sidebar"]
        for w in (self, self.lbl, self.del_btn):
            w.config(bg=bg)


# ──────────────────────────────────────────────
#  Main Application
# ──────────────────────────────────────────────
class OllamaInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama Chat")
        self.root.geometry("1200x740")
        self.root.minsize(900, 580)
        self.root.configure(bg=COLORS["bg"])
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.window_open = True
        self.response_queue = queue.Queue()
        self.llm = None
        self.checkpointer = MemorySaver()
        self.conversation_graph = self._build_graph()
        self.typing_indicator = None
        self._initialized = False

        # Sessions: list of dicts {id, title, thread_id, history: [(role, text, ts)]}
        self.sessions = []
        self.active_session_idx = None
        self._session_counter = 0

        # UI
        self.models = self._fetch_models()
        if not self.models:
            messagebox.showerror("Error", "No se pudieron cargar modelos. Verifica que Ollama esté corriendo.")
            self.root.after(0, self.root.destroy)
            return

        self._build_ui()
        self._new_session()
        self._poll_queue()
        self._initialized = True

    # ── LangGraph ──────────────────────────────
    def _build_graph(self):
        builder = StateGraph(ChatState)
        builder.add_node("chatbot", self._invoke_model)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _invoke_model(self, state: ChatState):
        response = self.llm.invoke(state["messages"])
        return {"messages": [AIMessage(content=response)]}

    # ── Fetch models ────────────────────────────
    def _fetch_models(self):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    # ── Build UI ────────────────────────────────
    def _build_ui(self):
        # ── Root paned layout: sidebar | main
        self.paned = tk.PanedWindow(self.root, orient="horizontal",
                                    bg=COLORS["border"], sashwidth=1,
                                    sashrelief="flat", handlesize=0)
        self.paned.grid(row=0, column=0, sticky="nsew")

        # ── Sidebar ──────────────────────────────
        sidebar = tk.Frame(self.paned, bg=COLORS["sidebar"], width=220)
        sidebar.pack_propagate(False)
        self.paned.add(sidebar, minsize=180)

        # Sidebar header
        sb_header = tk.Frame(sidebar, bg=COLORS["sidebar"])
        sb_header.pack(fill="x", padx=12, pady=(14, 8))

        tk.Label(sb_header, text="💬  Ollama Chat", bg=COLORS["sidebar"],
                 fg=COLORS["text"], font=FONTS["body_bold"]).pack(side="left")

        new_btn = tk.Label(sb_header, text="＋", bg=COLORS["sidebar"],
                           fg=COLORS["accent"], font=("Segoe UI", 16),
                           cursor="hand2")
        new_btn.pack(side="right")
        new_btn.bind("<Button-1>", lambda _: self._new_session())

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=8)

        # Session list
        self.session_list_frame = tk.Frame(sidebar, bg=COLORS["sidebar"])
        self.session_list_frame.pack(fill="both", expand=True, pady=4)

        # Sidebar footer: model selector
        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=8)
        sb_footer = tk.Frame(sidebar, bg=COLORS["sidebar"])
        sb_footer.pack(fill="x", padx=10, pady=10)

        tk.Label(sb_footer, text="MODEL", bg=COLORS["sidebar"],
                 fg=COLORS["text_dim"], font=FONTS["small"]).pack(anchor="w")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TCombobox",
                        fieldbackground=COLORS["input_bg"],
                        background=COLORS["input_bg"],
                        foreground=COLORS["text"],
                        arrowcolor=COLORS["accent"],
                        selectbackground=COLORS["input_bg"],
                        selectforeground=COLORS["text"])
        style.map("Dark.TCombobox",
                  fieldbackground=[("readonly", COLORS["input_bg"])],
                  foreground=[("readonly", COLORS["text"])])

        self.model_var = tk.StringVar(value=self.models[0])
        self.model_combo = ttk.Combobox(sb_footer, textvariable=self.model_var,
                                        values=self.models, state="readonly",
                                        style="Dark.TCombobox")
        self.model_combo.pack(fill="x", pady=(4, 0))

        # ── Main panel ───────────────────────────
        main_panel = tk.Frame(self.paned, bg=COLORS["surface"])
        self.paned.add(main_panel, minsize=560)

        main_panel.grid_rowconfigure(0, weight=0)  # topbar
        main_panel.grid_rowconfigure(1, weight=1)  # chat scroll
        main_panel.grid_rowconfigure(2, weight=0)  # input row
        main_panel.grid_columnconfigure(0, weight=1)

        # Top bar
        self.topbar = tk.Frame(main_panel, bg=COLORS["bg"], height=48)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_propagate(False)
        self.topbar.grid_columnconfigure(0, weight=1)

        self.session_title_var = tk.StringVar(value="New Chat")
        self.session_title_lbl = tk.Label(self.topbar, textvariable=self.session_title_var,
                                          bg=COLORS["bg"], fg=COLORS["text"],
                                          font=FONTS["body_bold"])
        self.session_title_lbl.grid(row=0, column=0, padx=16, pady=12, sticky="w")

        # Action buttons in topbar
        actions = tk.Frame(self.topbar, bg=COLORS["bg"])
        actions.grid(row=0, column=1, padx=10, sticky="e")

        IconButton(actions, "💾", self._export_chat, bg=COLORS["bg"]).pack(side="left")
        IconButton(actions, "🗑", self._clear_chat, bg=COLORS["bg"]).pack(side="left", padx=(2, 0))

        # Separator
        tk.Frame(main_panel, bg=COLORS["border"], height=1).grid(row=0, column=0, sticky="sew")

        # Chat scroll area
        self.chat_scroll = ScrollableFrame(main_panel, bg=COLORS["surface"])
        self.chat_scroll.grid(row=1, column=0, sticky="nsew")

        # Input row
        input_row = tk.Frame(main_panel, bg=COLORS["bg"], pady=10)
        input_row.grid(row=2, column=0, sticky="ew", padx=16)
        input_row.grid_columnconfigure(0, weight=1)

        input_wrap = tk.Frame(input_row, bg=COLORS["input_bg"],
                              highlightthickness=1,
                              highlightbackground=COLORS["border"])
        input_wrap.grid(row=0, column=0, sticky="ew")
        input_wrap.grid_columnconfigure(0, weight=1)

        self.prompt_entry = tk.Text(input_wrap, height=3, bg=COLORS["input_bg"],
                                    fg=COLORS["text"], insertbackground=COLORS["accent"],
                                    font=FONTS["body"], relief="flat",
                                    highlightthickness=0, padx=12, pady=10,
                                    wrap="word")
        self.prompt_entry.grid(row=0, column=0, sticky="ew")
        self.prompt_entry.bind("<Control-Return>", self._send)
        self.prompt_entry.bind("<FocusIn>",
                               lambda _: input_wrap.config(highlightbackground=COLORS["accent"]))
        self.prompt_entry.bind("<FocusOut>",
                               lambda _: input_wrap.config(highlightbackground=COLORS["border"]))

        # Placeholder
        self._placeholder_active = True
        ph_text = "Escribe tu mensaje...  (Ctrl+Enter para enviar)"
        self.prompt_entry.insert("1.0", ph_text)
        self.prompt_entry.config(fg=COLORS["text_dim"])
        self.prompt_entry.bind("<FocusIn>", lambda e: (self._clear_placeholder(), input_wrap.config(highlightbackground=COLORS["accent"])))
        self.prompt_entry.bind("<FocusOut>", lambda e: (self._restore_placeholder(), input_wrap.config(highlightbackground=COLORS["border"])))

        # Send button
        self.send_btn = tk.Button(input_row, text="Send  ➤",
                                  command=self._send,
                                  bg=COLORS["accent_glow"], fg=COLORS["text"],
                                  font=FONTS["body_bold"],
                                  relief="flat", cursor="hand2",
                                  padx=16, pady=10, bd=0,
                                  activebackground=COLORS["accent"],
                                  activeforeground=COLORS["text"])
        self.send_btn.grid(row=0, column=1, padx=(8, 0), sticky="s")
        self.send_btn.bind("<Enter>", lambda _: self.send_btn.config(bg=COLORS["accent"]))
        self.send_btn.bind("<Leave>", lambda _: self.send_btn.config(bg=COLORS["accent_glow"]))

        # Status bar
        self.status_var = tk.StringVar(value="")
        tk.Label(input_row, textvariable=self.status_var, bg=COLORS["bg"],
                 fg=COLORS["text_dim"], font=FONTS["small"]).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    # ── Placeholder helpers ──────────────────────
    def _clear_placeholder(self):
        if self._placeholder_active:
            self.prompt_entry.delete("1.0", "end")
            self.prompt_entry.config(fg=COLORS["text"])
            self._placeholder_active = False

    def _restore_placeholder(self):
        if not self.prompt_entry.get("1.0", "end").strip():
            self.prompt_entry.insert("1.0", "Escribe tu mensaje...  (Ctrl+Enter para enviar)")
            self.prompt_entry.config(fg=COLORS["text_dim"])
            self._placeholder_active = True

    # ── Sessions ────────────────────────────────
    def _new_session(self):
        self._session_counter += 1
        session = {
            "id": self._session_counter,
            "title": f"Chat {self._session_counter}",
            "thread_id": f"thread_{time.time()}_{self._session_counter}",
            "history": [],
        }
        self.sessions.append(session)
        idx = len(self.sessions) - 1
        self._add_session_widget(len(self.sessions) - 1)
        self._switch_session(idx)

    def _add_session_widget(self, idx):
        session = self.sessions[idx]

        def on_click():
            self._switch_session(idx)

        def on_delete():
            self._delete_session(idx)

        item = SessionItem(self.session_list_frame,
                           title=session["title"],
                           on_click=on_click,
                           on_delete=on_delete)
        item.pack(fill="x", padx=4, pady=2)
        session["_widget"] = item

    def _switch_session(self, idx):
        if self.active_session_idx is not None and self.active_session_idx < len(self.sessions):
            self.sessions[self.active_session_idx]["_widget"].set_active(False)

        self.active_session_idx = idx
        session = self.sessions[idx]
        session["_widget"].set_active(True)
        self.session_title_var.set(session["title"])

        # Rebuild chat area
        self._rebuild_chat(session)

    def _delete_session(self, idx):
        if len(self.sessions) == 1:
            messagebox.showinfo("Info", "Debe haber al menos una sesión.")
            return
        session = self.sessions[idx]
        session["_widget"].destroy()
        self.sessions.pop(idx)

        # Rebuild widgets (indices changed)
        for w in self.session_list_frame.winfo_children():
            w.destroy()
        for i in range(len(self.sessions)):
            self._add_session_widget(i)

        new_idx = min(idx, len(self.sessions) - 1)
        self.active_session_idx = None
        self._switch_session(new_idx)

    def _rebuild_chat(self, session):
        for w in self.chat_scroll.inner.winfo_children():
            w.destroy()
        if self.typing_indicator:
            self.typing_indicator = None

        if not session["history"]:
            # Welcome message
            welcome = tk.Frame(self.chat_scroll.inner, bg=COLORS["surface"])
            welcome.pack(fill="x", expand=True, pady=60)
            tk.Label(welcome, text="✦", bg=COLORS["surface"],
                     fg=COLORS["accent"], font=("Segoe UI Emoji", 32)).pack()
            tk.Label(welcome, text="¿En qué puedo ayudarte hoy?",
                     bg=COLORS["surface"], fg=COLORS["text"],
                     font=FONTS["title"]).pack(pady=(8, 0))
            tk.Label(welcome, text=f"Modelo activo: {self.model_var.get()}",
                     bg=COLORS["surface"], fg=COLORS["text_dim"],
                     font=FONTS["small"]).pack(pady=(4, 0))
        else:
            for role, text, ts in session["history"]:
                ChatBubble(self.chat_scroll.inner, role, text, ts).pack(fill="x")

        self.chat_scroll.scroll_bottom()

    # ── Send message ────────────────────────────
    def _send(self, event=None):
        if self._placeholder_active:
            return
        prompt = self.prompt_entry.get("1.0", "end").strip()
        if not prompt:
            return

        self.prompt_entry.delete("1.0", "end")
        self._placeholder_active = False

        session = self.sessions[self.active_session_idx]
        ts = time.strftime("%H:%M")

        # Update title from first message
        if not session["history"]:
            title = prompt[:30] + ("…" if len(prompt) > 30 else "")
            session["title"] = title
            session["_widget"].lbl.config(text=title)
            self.session_title_var.set(title)

        # Add user bubble
        session["history"].append(("user", prompt, ts))
        # Remove welcome screen if present
        for w in self.chat_scroll.inner.winfo_children():
            w.destroy()
        for role, text, t in session["history"]:
            ChatBubble(self.chat_scroll.inner, role, text, t).pack(fill="x")

        # Typing indicator
        self.typing_indicator = TypingIndicator(self.chat_scroll.inner,
                                                bd=0, padx=14, pady=10)
        self.typing_indicator.pack(anchor="w", padx=14, pady=6)
        self.typing_indicator.start()
        self.chat_scroll.scroll_bottom()

        # Disable input
        self.send_btn.config(state="disabled")
        self.prompt_entry.config(state="disabled")
        self.status_var.set("Generando respuesta…")

        selected_model = self.model_var.get()
        self.llm = OllamaLLM(model=selected_model)

        threading.Thread(target=self._generate,
                         args=(prompt, session["thread_id"]),
                         daemon=True).start()

    def _generate(self, prompt, thread_id):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            initial_message = HumanMessage(content=prompt)
            response_text = ""
            for event in self.conversation_graph.stream(
                {"messages": [initial_message]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in event:
                    response_text = event["messages"][-1].content
            self.response_queue.put(("ok", response_text))
        except Exception as e:
            self.response_queue.put(("error", str(e)))

    def _poll_queue(self):
        if not self.window_open:
            return
        try:
            result_type, content = self.response_queue.get_nowait()

            # Remove typing indicator
            if self.typing_indicator:
                self.typing_indicator.stop()
                self.typing_indicator.destroy()
                self.typing_indicator = None

            session = self.sessions[self.active_session_idx]
            ts = time.strftime("%H:%M")

            if result_type == "error":
                content = f"⚠️  **Error:** {content}"

            session["history"].append(("ai", content, ts))
            ChatBubble(self.chat_scroll.inner, "ai", content, ts).pack(fill="x")
            self.chat_scroll.scroll_bottom()

            # Re-enable input
            self.send_btn.config(state="normal")
            self.prompt_entry.config(state="normal")
            self.status_var.set("")
            self.prompt_entry.focus_set()

        except queue.Empty:
            pass
        finally:
            if self.window_open:
                self.root.after(100, self._poll_queue)

    # ── Actions ─────────────────────────────────
    def _export_chat(self):
        session = self.sessions[self.active_session_idx]
        if not session["history"]:
            messagebox.showinfo("Info", "No hay mensajes para exportar.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("Markdown", "*.md")],
            initialfile=f"{session['title']}.txt"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# {session['title']}\n\n")
                    for role, text, ts in session["history"]:
                        label = "You" if role == "user" else "Ollama"
                        f.write(f"**[{ts}] {label}:**\n{text}\n\n")
                messagebox.showinfo("✅ Éxito", "Chat exportado correctamente.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _clear_chat(self):
        session = self.sessions[self.active_session_idx]
        if not session["history"]:
            return
        if messagebox.askyesno("Confirmar", "¿Borrar el historial de esta sesión?"):
            session["history"].clear()
            session["title"] = f"Chat {session['id']}"
            session["thread_id"] = f"thread_{time.time()}_{session['id']}"
            session["_widget"].lbl.config(text=session["title"])
            self.session_title_var.set(session["title"])
            self._rebuild_chat(session)

    def on_close(self):
        self.window_open = False
        self.root.destroy()


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.iconbitmap(default="")
    except Exception:
        pass
    app = OllamaInterface(root)
    if getattr(app, "_initialized", False):
        root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

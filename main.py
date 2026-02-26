import customtkinter as ctk
from tkinter import messagebox, filedialog
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

# ── CustomTkinter global config ─────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── State definition ─────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]


# ── Palette (Synthwave / Neon) ───────────────────────
C = {
    # Backgrounds
    "bg":           "#07080F",   # casi negro azulado
    "sidebar":      "#0C0D1A",   # sidebar oscuro
    "surface":      "#0F1020",   # área de chat
    "card":         "#171829",   # cards/burbujas AI
    "input_bg":     "#12132B",   # fondo del input

    # Bubbles
    "bubble_user":  "#3D2FBF",   # violeta profundo
    "bubble_user2": "#5B4AE8",   # borde/gradiente usuario

    # Accent colors
    "purple":       "#8B5CF6",   # violeta vibrante
    "cyan":         "#06B6D4",   # cyan eléctrico
    "pink":         "#EC4899",   # rosa neón
    "green":        "#10B981",   # verde menta
    "yellow":       "#F59E0B",   # ámbar

    # Text
    "text":         "#EEF0FF",   # blanco azulado
    "text_dim":     "#7B82B0",   # gris violeta
    "text_code":    "#67E8F9",   # cyan para código

    # Borders
    "border":       "#1E2048",
    "border_glow":  "#4C3DD4",

    # Buttons
    "btn_primary":  "#5C4AE4",
    "btn_hover":    "#7B6AF0",
    "btn_delete":   "#991B1B",
    "btn_delete_h": "#DC2626",
}

FONTS = {
    "title":     ("Segoe UI Semibold", 15),
    "subtitle":  ("Segoe UI", 10),
    "body":      ("Segoe UI", 12),
    "body_bold": ("Segoe UI Semibold", 12),
    "small":     ("Segoe UI", 10),
    "mono":      ("Cascadia Code", 10),
    "nano":      ("Segoe UI", 9),
}


# ── Typing indicator ─────────────────────────────────
class TypingIndicator(ctk.CTkFrame):
    def __init__(self, parent, **kw):
        super().__init__(parent, fg_color=C["card"],
                         corner_radius=14, **kw)
        self._job = None
        self._step = 0
        self._dots = []

        ctk.CTkLabel(self, text="✦  Ollama", text_color=C["purple"],
                     font=FONTS["small"], fg_color="transparent").pack(
            side="left", padx=(12, 6), pady=10)

        dot_frame = ctk.CTkFrame(self, fg_color="transparent")
        dot_frame.pack(side="left", pady=10, padx=(0, 12))
        for _ in range(3):
            d = ctk.CTkLabel(dot_frame, text="●", width=10,
                             text_color=C["text_dim"],
                             font=("Segoe UI", 10),
                             fg_color="transparent")
            d.pack(side="left", padx=2)
            self._dots.append(d)

    def start(self):
        self._animate()

    def stop(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None

    def _animate(self):
        colors = [C["purple"], C["cyan"], C["pink"]]
        for i, dot in enumerate(self._dots):
            dot.configure(text_color=colors[i] if i == self._step % 3 else C["text_dim"])
        self._step += 1
        self._job = self.after(400, self._animate)


# ── Chat bubble ──────────────────────────────────────
class ChatBubble(ctk.CTkFrame):
    def __init__(self, parent, role: str, text: str, timestamp: str, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        is_user = role == "user"

        # Meta row
        meta = ctk.CTkFrame(self, fg_color="transparent")
        if is_user:
            meta.pack(anchor="e", padx=(80, 14), pady=(10, 3))
            ctk.CTkLabel(meta, text=f"{timestamp}  ", text_color=C["text_dim"],
                         font=FONTS["nano"], fg_color="transparent").pack(side="left")
            ctk.CTkLabel(meta, text="Tú", text_color=C["purple"],
                         font=FONTS["small"], fg_color="transparent").pack(side="left")
        else:
            meta.pack(anchor="w", padx=(14, 80), pady=(10, 3))
            ctk.CTkLabel(meta, text="✦ Ollama", text_color=C["cyan"],
                         font=FONTS["small"], fg_color="transparent").pack(side="left")
            ctk.CTkLabel(meta, text=f"  {timestamp}", text_color=C["text_dim"],
                         font=FONTS["nano"], fg_color="transparent").pack(side="left")

        # Bubble container
        bubble_color = C["bubble_user"] if is_user else C["card"]
        border_color = C["bubble_user2"] if is_user else C["border"]

        bubble = ctk.CTkFrame(self, fg_color=bubble_color,
                               corner_radius=16,
                               border_width=1,
                               border_color=border_color)
        if is_user:
            bubble.pack(anchor="e", padx=(80, 14), pady=(0, 8))
        else:
            bubble.pack(anchor="w", padx=(14, 80), pady=(0, 8))

        self._render(bubble, text)

    def _render(self, parent, text):
        """Render with basic markdown: ```blocks```, **bold**, `code`."""
        parts = re.split(r'(```[\s\S]*?```)', text)
        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                raw = part[3:-3].strip()
                # Strip language hint
                if "\n" in raw:
                    first, rest = raw.split("\n", 1)
                    if re.match(r'^[a-zA-Z0-9+#-]+$', first.strip()):
                        raw = rest

                code_frame = ctk.CTkFrame(parent, fg_color="#050710",
                                           corner_radius=8,
                                           border_width=1,
                                           border_color=C["cyan"])
                code_frame.pack(fill="x", padx=12, pady=6)
                ctk.CTkLabel(code_frame, text=raw,
                             text_color=C["text_code"],
                             font=FONTS["mono"],
                             justify="left", anchor="w",
                             wraplength=520,
                             fg_color="transparent").pack(
                    padx=12, pady=8, anchor="w")
            else:
                stripped = part.strip()
                if stripped:
                    for line in stripped.split("\n"):
                        if not line.strip():
                            ctk.CTkFrame(parent, fg_color="transparent",
                                         height=4).pack()
                            continue
                        self._render_line(parent, line.strip())

    def _render_line(self, parent, line):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=1, anchor="w")

        pattern = re.compile(r'(\*\*(.+?)\*\*|`([^`]+)`)')
        pos = 0
        col = 0
        for m in pattern.finditer(line):
            before = line[pos:m.start()]
            if before:
                ctk.CTkLabel(row, text=before, text_color=C["text"],
                             font=FONTS["body"], fg_color="transparent",
                             wraplength=520, justify="left").grid(
                    row=0, column=col, sticky="w")
                col += 1
            if m.group(1).startswith("**"):
                ctk.CTkLabel(row, text=m.group(2), text_color=C["text"],
                             font=FONTS["body_bold"], fg_color="transparent",
                             wraplength=520, justify="left").grid(
                    row=0, column=col, sticky="w")
            else:
                ctk.CTkLabel(row, text=m.group(3), text_color=C["text_code"],
                             font=FONTS["mono"], fg_color="#050710",
                             corner_radius=4,
                             wraplength=520, justify="left").grid(
                    row=0, column=col, sticky="w", padx=2)
            col += 1
            pos = m.end()
        tail = line[pos:]
        if tail:
            ctk.CTkLabel(row, text=tail, text_color=C["text"],
                         font=FONTS["body"], fg_color="transparent",
                         wraplength=520, justify="left").grid(
                row=0, column=col, sticky="w")


# ── Session item in sidebar ──────────────────────────
class SessionItem(ctk.CTkFrame):
    def __init__(self, parent, title, on_click, on_delete, **kw):
        super().__init__(parent, fg_color="transparent",
                         corner_radius=10, cursor="hand2", **kw)
        self._active = False
        self._on_click = on_click

        self.lbl = ctk.CTkLabel(self, text=title, text_color=C["text_dim"],
                                 font=FONTS["small"], anchor="w",
                                 wraplength=150, justify="left",
                                 fg_color="transparent")
        self.lbl.pack(side="left", fill="x", expand=True, padx=(10, 0), pady=8)

        self.del_btn = ctk.CTkButton(self, text="✕", width=22, height=22,
                                      fg_color="transparent",
                                      hover_color=C["btn_delete"],
                                      text_color=C["text_dim"],
                                      font=FONTS["nano"],
                                      corner_radius=6,
                                      command=on_delete)
        self.del_btn.pack(side="right", padx=(0, 6))

        self.bind("<Button-1>", lambda _: on_click())
        self.lbl.bind("<Button-1>", lambda _: on_click())
        self.bind("<Enter>", self._hover_on)
        self.bind("<Leave>", self._hover_off)
        self.lbl.bind("<Enter>", self._hover_on)
        self.lbl.bind("<Leave>", self._hover_off)

    def set_active(self, active: bool):
        self._active = active
        color = C["border"] if active else "transparent"
        self.configure(fg_color=color)
        self.lbl.configure(text_color=C["text"] if active else C["text_dim"],
                           font=FONTS["body_bold"] if active else FONTS["small"])

    def _hover_on(self, _):
        if not self._active:
            self.configure(fg_color=C["border"])

    def _hover_off(self, _):
        if not self._active:
            self.configure(fg_color="transparent")


# ── Main Application ─────────────────────────────────
class OllamaInterface:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Ollama Chat")
        self.root.geometry("1260x780")
        self.root.minsize(960, 600)
        self.root.configure(fg_color=C["bg"])

        self.window_open = True
        self.response_queue: queue.Queue = queue.Queue()
        self.llm = None
        self.checkpointer = MemorySaver()
        self.conversation_graph = self._build_graph()
        self.typing_indicator = None
        self._initialized = False

        self.sessions: list[dict] = []
        self.active_idx: int | None = None
        self._session_cnt = 0

        self.models = self._fetch_models()
        if not self.models:
            messagebox.showerror(
                "Sin conexión",
                "No se encontraron modelos.\nAsegúrate de que Ollama esté corriendo:\n\n  ollama serve"
            )
            self.root.after(0, self.root.destroy)
            return

        self._build_ui()
        self._new_session()
        self._poll_queue()
        self._initialized = True

    # ── LangGraph ─────────────────────────────────
    def _build_graph(self):
        builder = StateGraph(ChatState)
        builder.add_node("chatbot", self._invoke_model)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _invoke_model(self, state: ChatState):
        response = self.llm.invoke(state["messages"])
        return {"messages": [AIMessage(content=response)]}

    # ── Fetch models ───────────────────────────────
    def _fetch_models(self):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    # ── Build UI ───────────────────────────────────
    def _build_ui(self):
        # Root 2-column grid
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──────────────────────────────
        sidebar = ctk.CTkFrame(self.root, width=230,
                                fg_color=C["sidebar"],
                                corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(2, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        # Logo / title
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.grid(row=0, column=0, sticky="ew", padx=14, pady=(18, 8))
        logo_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(logo_frame, text="✦  Ollama Chat",
                     text_color=C["purple"], font=FONTS["title"],
                     fg_color="transparent").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(logo_frame, text="Local · Privado · Rápido",
                     text_color=C["text_dim"], font=FONTS["nano"],
                     fg_color="transparent").grid(row=1, column=0, sticky="w", pady=(2, 0))

        # New chat button
        new_btn = ctk.CTkButton(sidebar, text="＋  Nuevo Chat",
                                 fg_color=C["btn_primary"],
                                 hover_color=C["btn_hover"],
                                 text_color=C["text"],
                                 font=FONTS["body_bold"],
                                 corner_radius=10, height=38,
                                 command=self._new_session)
        new_btn.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        # Separator line
        ctk.CTkFrame(sidebar, height=1, fg_color=C["border"],
                     corner_radius=0).grid(row=2, column=0, sticky="ew",
                                           padx=8, pady=(0, 6))

        # Session list
        self.session_scroll = ctk.CTkScrollableFrame(
            sidebar, fg_color="transparent",
            scrollbar_button_color=C["border"],
            scrollbar_button_hover_color=C["purple"])
        self.session_scroll.grid(row=3, column=0, sticky="nsew",
                                  padx=6, pady=(0, 6))
        self.session_scroll.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(3, weight=1)

        # Model selector footer
        ctk.CTkFrame(sidebar, height=1, fg_color=C["border"],
                     corner_radius=0).grid(row=4, column=0, sticky="ew", padx=8)

        footer = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer.grid(row=5, column=0, sticky="ew", padx=12, pady=10)
        footer.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(footer, text="🤖  MODELO", text_color=C["text_dim"],
                     font=FONTS["nano"], fg_color="transparent").grid(
            row=0, column=0, sticky="w", pady=(0, 4))

        self.model_var = ctk.StringVar(value=self.models[0])
        self.model_combo = ctk.CTkComboBox(footer,
                                            values=self.models,
                                            variable=self.model_var,
                                            fg_color=C["input_bg"],
                                            border_color=C["border"],
                                            button_color=C["purple"],
                                            button_hover_color=C["btn_hover"],
                                            dropdown_fg_color=C["card"],
                                            dropdown_hover_color=C["border"],
                                            text_color=C["text"],
                                            font=FONTS["small"],
                                            state="readonly")
        self.model_combo.grid(row=1, column=0, sticky="ew")

        # ── Main panel ────────────────────────────
        main = ctk.CTkFrame(self.root, fg_color=C["surface"], corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Top bar
        topbar = ctk.CTkFrame(main, fg_color=C["sidebar"],
                               corner_radius=0, height=52)
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.grid_propagate(False)
        topbar.grid_columnconfigure(0, weight=1)

        self.title_var = ctk.StringVar(value="Nuevo Chat")
        ctk.CTkLabel(topbar, textvariable=self.title_var,
                     text_color=C["text"], font=FONTS["title"],
                     fg_color="transparent").grid(
            row=0, column=0, padx=20, pady=14, sticky="w")

        # Action buttons
        actions = ctk.CTkFrame(topbar, fg_color="transparent")
        actions.grid(row=0, column=1, padx=12, sticky="e")

        ctk.CTkButton(actions, text="💾  Exportar",
                       fg_color="transparent",
                       hover_color=C["border"],
                       border_width=1, border_color=C["border"],
                       text_color=C["text_dim"], font=FONTS["small"],
                       corner_radius=8, height=30, width=100,
                       command=self._export_chat).pack(side="left", padx=(0, 6))

        ctk.CTkButton(actions, text="🗑  Borrar",
                       fg_color="transparent",
                       hover_color=C["btn_delete"],
                       border_width=1, border_color=C["border"],
                       text_color=C["text_dim"], font=FONTS["small"],
                       corner_radius=8, height=30, width=90,
                       command=self._clear_chat).pack(side="left")

        # Separator
        ctk.CTkFrame(main, height=1, fg_color=C["border"],
                     corner_radius=0).grid(row=0, column=0, sticky="sew")

        # Chat scroll area
        self.chat_scroll = ctk.CTkScrollableFrame(
            main, fg_color=C["surface"],
            scrollbar_button_color=C["border"],
            scrollbar_button_hover_color=C["purple"])
        self.chat_scroll.grid(row=1, column=0, sticky="nsew")
        self.chat_scroll.grid_columnconfigure(0, weight=1)

        # ── Input area ────────────────────────────
        input_panel = ctk.CTkFrame(main, fg_color=C["sidebar"],
                                    corner_radius=0)
        input_panel.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        input_panel.grid_columnconfigure(0, weight=1)

        # Separator above input
        ctk.CTkFrame(input_panel, height=1, fg_color=C["border"],
                     corner_radius=0).grid(row=0, column=0, columnspan=2, sticky="ew")

        # Textbox
        self.prompt_box = ctk.CTkTextbox(input_panel,
                                          height=90,
                                          fg_color=C["input_bg"],
                                          text_color=C["text"],
                                          font=FONTS["body"],
                                          border_width=1,
                                          border_color=C["border"],
                                          corner_radius=12,
                                          wrap="word",
                                          scrollbar_button_color=C["border"])
        self.prompt_box.grid(row=1, column=0, padx=(16, 8),
                              pady=12, sticky="ew")

        # Placeholder
        self._ph_active = True
        self._ph_text = "Escribe tu mensaje...  (Ctrl+Enter para enviar)"
        self.prompt_box.insert("0.0", self._ph_text)
        self.prompt_box.configure(text_color=C["text_dim"])
        self.prompt_box.bind("<FocusIn>", self._clear_ph)
        self.prompt_box.bind("<FocusOut>", self._restore_ph)
        self.prompt_box.bind("<Control-Return>", self._send)

        # Send button
        self.send_btn = ctk.CTkButton(input_panel,
                                       text="Enviar\n➤",
                                       width=90, height=90,
                                       fg_color=C["btn_primary"],
                                       hover_color=C["btn_hover"],
                                       text_color=C["text"],
                                       font=FONTS["body_bold"],
                                       corner_radius=12,
                                       command=self._send)
        self.send_btn.grid(row=1, column=1, padx=(0, 16), pady=12)

        # Status bar
        self.status_var = ctk.StringVar(value="")
        self._status_lbl = ctk.CTkLabel(input_panel,
                                         textvariable=self.status_var,
                                         text_color=C["text_dim"],
                                         font=FONTS["nano"],
                                         fg_color="transparent")
        self._status_lbl.grid(row=2, column=0, columnspan=2,
                               sticky="w", padx=18, pady=(0, 8))

    # ── Placeholder ────────────────────────────────
    def _clear_ph(self, _=None):
        if self._ph_active:
            self.prompt_box.delete("0.0", "end")
            self.prompt_box.configure(text_color=C["text"],
                                       border_color=C["purple"])
            self._ph_active = False

    def _restore_ph(self, _=None):
        if not self.prompt_box.get("0.0", "end").strip():
            self.prompt_box.insert("0.0", self._ph_text)
            self.prompt_box.configure(text_color=C["text_dim"],
                                       border_color=C["border"])
            self._ph_active = True

    # ── Sessions ────────────────────────────────────
    def _new_session(self):
        self._session_cnt += 1
        session = {
            "id":        self._session_cnt,
            "title":     f"Chat {self._session_cnt}",
            "thread_id": f"thread_{time.time()}_{self._session_cnt}",
            "history":   [],
            "_widget":   None,
        }
        self.sessions.append(session)
        idx = len(self.sessions) - 1
        self._add_session_widget(idx)
        self._switch_session(idx)

    def _add_session_widget(self, idx):
        session = self.sessions[idx]

        item = SessionItem(
            self.session_scroll,
            title=session["title"],
            on_click=lambda i=idx: self._switch_session(i),
            on_delete=lambda i=idx: self._delete_session(i),
        )
        item.grid(row=idx, column=0, sticky="ew", pady=2)
        session["_widget"] = item

    def _switch_session(self, idx):
        if self.active_idx is not None and self.active_idx < len(self.sessions):
            self.sessions[self.active_idx]["_widget"].set_active(False)

        self.active_idx = idx
        s = self.sessions[idx]
        s["_widget"].set_active(True)
        self.title_var.set(s["title"])
        self._rebuild_chat(s)

    def _delete_session(self, idx):
        if len(self.sessions) == 1:
            messagebox.showinfo("Info", "Debe haber al menos una sesión.")
            return
        # Destroy widget
        self.sessions[idx]["_widget"].destroy()
        self.sessions.pop(idx)

        # Rebuild all widgets with correct indices
        for w in self.session_scroll.winfo_children():
            w.destroy()
        for i in range(len(self.sessions)):
            self._add_session_widget(i)

        new_idx = min(idx, len(self.sessions) - 1)
        self.active_idx = None
        self._switch_session(new_idx)

    def _rebuild_chat(self, session):
        for w in self.chat_scroll.winfo_children():
            w.destroy()
        self.typing_indicator = None

        if not session["history"]:
            self._show_welcome()
        else:
            for role, text, ts in session["history"]:
                ChatBubble(self.chat_scroll, role, text, ts).pack(
                    fill="x", pady=0)
        self._scroll_bottom()

    def _show_welcome(self):
        frame = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        frame.pack(expand=True, pady=80)

        # Glowing icon
        ctk.CTkLabel(frame, text="✦",
                     text_color=C["purple"],
                     font=("Segoe UI Emoji", 48),
                     fg_color="transparent").pack()
        ctk.CTkLabel(frame, text="¿En qué puedo ayudarte hoy?",
                     text_color=C["text"],
                     font=FONTS["title"],
                     fg_color="transparent").pack(pady=(10, 4))
        ctk.CTkLabel(frame, text=f"Modelo activo: {self.model_var.get()}",
                     text_color=C["text_dim"],
                     font=FONTS["small"],
                     fg_color="transparent").pack()

        # Quick tip chips
        tips_frame = ctk.CTkFrame(frame, fg_color="transparent")
        tips_frame.pack(pady=(24, 0))
        tips = ["💡  Haz una pregunta", "🐍  Escribe código Python",
                "📖  Resume un texto", "🔍  Explica un concepto"]
        for t in tips:
            chip = ctk.CTkFrame(tips_frame, fg_color=C["card"],
                                 corner_radius=20,
                                 border_width=1, border_color=C["border"])
            chip.pack(side="left", padx=6)
            ctk.CTkLabel(chip, text=t, text_color=C["text_dim"],
                          font=FONTS["small"],
                          fg_color="transparent").pack(padx=14, pady=8)

    # ── Send ─────────────────────────────────────────
    def _send(self, _=None):
        if self._ph_active:
            return
        prompt = self.prompt_box.get("0.0", "end").strip()
        if not prompt:
            return

        self.prompt_box.delete("0.0", "end")
        self._ph_active = False

        session = self.sessions[self.active_idx]
        ts = time.strftime("%H:%M")

        # Auto-title
        if not session["history"]:
            title = prompt[:28] + ("…" if len(prompt) > 28 else "")
            session["title"] = title
            session["_widget"].lbl.configure(text=title)
            self.title_var.set(title)

        # User bubble
        session["history"].append(("user", prompt, ts))
        for w in self.chat_scroll.winfo_children():
            w.destroy()
        for role, text, t in session["history"]:
            ChatBubble(self.chat_scroll, role, text, t).pack(fill="x")

        # Typing indicator
        self.typing_indicator = TypingIndicator(self.chat_scroll)
        self.typing_indicator.pack(anchor="w", padx=14, pady=8)
        self.typing_indicator.start()
        self._scroll_bottom()

        # Lock input
        self.send_btn.configure(state="disabled",
                                 fg_color=C["border"],
                                 text_color=C["text_dim"])
        self.prompt_box.configure(state="disabled")
        self.status_var.set("⏳  Generando respuesta…")

        self.llm = OllamaLLM(model=self.model_var.get())
        threading.Thread(target=self._generate,
                         args=(prompt, session["thread_id"]),
                         daemon=True).start()

    def _generate(self, prompt, thread_id):
        try:
            config = {"configurable": {"thread_id": thread_id}}
            msg = HumanMessage(content=prompt)
            response_text = ""
            for event in self.conversation_graph.stream(
                {"messages": [msg]}, config=config, stream_mode="values"
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
            kind, content = self.response_queue.get_nowait()

            if self.typing_indicator:
                self.typing_indicator.stop()
                self.typing_indicator.destroy()
                self.typing_indicator = None

            session = self.sessions[self.active_idx]
            ts = time.strftime("%H:%M")

            if kind == "error":
                content = f"⚠️  **Error:**\n{content}"

            session["history"].append(("ai", content, ts))
            ChatBubble(self.chat_scroll, "ai", content, ts).pack(fill="x")
            self._scroll_bottom()

            # Unlock input
            self.send_btn.configure(state="normal",
                                     fg_color=C["btn_primary"],
                                     text_color=C["text"])
            self.prompt_box.configure(state="normal")
            self.status_var.set("")
            self.prompt_box.focus_set()

        except queue.Empty:
            pass
        finally:
            if self.window_open:
                self.root.after(100, self._poll_queue)

    def _scroll_bottom(self):
        self.chat_scroll.update_idletasks()
        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    # ── Actions ──────────────────────────────────────
    def _export_chat(self):
        session = self.sessions[self.active_idx]
        if not session["history"]:
            messagebox.showinfo("Info", "No hay mensajes para exportar.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt")],
            initialfile=f"{session['title']}.md"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# {session['title']}\n\n")
                    for role, text, ts in session["history"]:
                        label = "**Tú**" if role == "user" else "**Ollama**"
                        f.write(f"### {label} · {ts}\n{text}\n\n---\n\n")
                messagebox.showinfo("✅  Exportado", f"Guardado en:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _clear_chat(self):
        session = self.sessions[self.active_idx]
        if not session["history"]:
            return
        if messagebox.askyesno("Confirmar", "¿Borrar el historial de este chat?"):
            session["history"].clear()
            session["title"] = f"Chat {session['id']}"
            session["thread_id"] = f"thread_{time.time()}_{session['id']}"
            session["_widget"].lbl.configure(text=session["title"])
            self.title_var.set(session["title"])
            self._rebuild_chat(session)

    def on_close(self):
        self.window_open = False
        self.root.destroy()


# ── Entry point ──────────────────────────────────────
if __name__ == "__main__":
    root = ctk.CTk()
    app = OllamaInterface(root)
    if getattr(app, "_initialized", False):
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()

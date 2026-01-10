import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import time
from pygments import lex
from pygments.lexers import PythonLexer
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
import requests

# State definition with persistence
class ChatState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # Reducer to append messages

class OllamaInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama Chat Interface")
        self.root.geometry("1100x700")
        self.root.minsize(980, 620)
        self.colors = {
            "bg": "#1F2633",
            "surface": "#2A3242",
            "surface_alt": "#313A4D",
            "border": "#3B4252",
            "accent": "#88C0D0",
            "accent_hover": "#7FBBC5",
            "text": "#E5E9F0",
            "muted": "#A7B0C0",
        }
        self.fonts = {
            "title": ("Segoe UI Semibold", 18),
            "subtitle": ("Segoe UI", 10),
            "body": ("Segoe UI", 11),
            "body_bold": ("Segoe UI Semibold", 11),
            "mono": ("Consolas", 11),
        }
        self.root.configure(bg=self.colors["bg"])

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.models = self.get_models()
        if not self.models:
            messagebox.showerror("Error", "Could not load models. Please check your connection to Ollama.")
            self.root.destroy()
            return

        self.llm = None
        self.checkpointer = MemorySaver()  # Checkpointer for persistence
        self.conversation_graph = self.create_conversation_graph()  # LangGraph graph
        self.response_queue = queue.Queue()
        self.animation_active = False
        self.current_thread = None  # Current conversation thread
        self.window_open = True  # Flag to check if the window is open

        self.create_interface()
        self.check_response()

    def get_models(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return [model["name"] for model in response.json()["models"]]
            else:
                messagebox.showerror("Error", f"Could not fetch models: {response.text}")
                return []
        except Exception as e:
            messagebox.showerror("Error", f"Connection error: {str(e)}")
            return []

    def create_interface(self):
        main_frame = ttk.Frame(self.root, padding=16, style="TFrame")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1, uniform="content")
        main_frame.grid_columnconfigure(1, weight=1, uniform="content")
        main_frame.grid_rowconfigure(1, weight=1)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Card.TFrame", background=self.colors["surface"])
        style.configure("TLabel", foreground=self.colors["text"], background=self.colors["bg"], font=self.fonts["body"])
        style.configure("Header.TLabel", foreground=self.colors["text"], background=self.colors["bg"], font=self.fonts["title"])
        style.configure("Subtle.TLabel", foreground=self.colors["muted"], background=self.colors["bg"], font=self.fonts["subtitle"])
        style.configure("CardTitle.TLabel", foreground=self.colors["text"], background=self.colors["surface"], font=self.fonts["body_bold"])
        style.configure("Field.TLabel", foreground=self.colors["muted"], background=self.colors["surface"], font=self.fonts["subtitle"])
        style.configure(
            "Primary.TButton",
            foreground=self.colors["bg"],
            background=self.colors["accent"],
            padding=(12, 6),
            font=self.fonts["body_bold"],
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.colors["accent_hover"]), ("disabled", "#5F6777")],
            foreground=[("disabled", "#C0C6D4")],
        )
        style.configure(
            "Secondary.TButton",
            foreground=self.colors["text"],
            background=self.colors["surface_alt"],
            padding=(10, 5),
            font=self.fonts["body"],
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#3C455A"), ("disabled", "#3A4150")],
            foreground=[("disabled", "#8B93A4")],
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["surface_alt"],
            background=self.colors["surface_alt"],
            foreground=self.colors["text"],
            arrowcolor=self.colors["accent"],
            padding=6,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.colors["surface_alt"])],
            foreground=[("readonly", self.colors["text"])],
        )
        style.configure("TSeparator", background=self.colors["border"])

        header_frame = ttk.Frame(main_frame, style="TFrame")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header_frame.grid_columnconfigure(0, weight=1)
        ttk.Label(header_frame, text="Ollama Chat", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header_frame,
            text="Fast, local responses with a clean workspace.",
            style="Subtle.TLabel",
        ).grid(row=1, column=0, sticky="w")
        ttk.Separator(header_frame, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=(10, 0))

        prompt_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=14)
        prompt_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        prompt_frame.grid_columnconfigure(0, weight=1)
        prompt_frame.grid_rowconfigure(3, weight=1)

        ttk.Label(prompt_frame, text="MODEL", style="Field.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        self.model_combobox = ttk.Combobox(prompt_frame, values=self.models, width=40, state="readonly")
        self.model_combobox.grid(row=1, column=0, pady=(0, 12), sticky="ew")
        self.model_combobox.current(0)

        ttk.Label(prompt_frame, text="PROMPT", style="Field.TLabel").grid(row=2, column=0, sticky=tk.W, pady=(0, 4))
        self.prompt_entry = tk.Text(
            prompt_frame,
            height=10,
            width=50,
            bg=self.colors["surface_alt"],
            fg=self.colors["text"],
            insertbackground=self.colors["accent"],
            font=self.fonts["body"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
        )
        self.prompt_entry.grid(row=3, column=0, pady=(0, 8), sticky="nsew")
        self.prompt_entry.bind("<Control-Return>", self.start_response_generation)

        ttk.Label(prompt_frame, text="Tip: Ctrl+Enter to send", style="Subtle.TLabel").grid(
            row=4, column=0, sticky="w", pady=(0, 10)
        )

        self.generate_button = ttk.Button(
            prompt_frame, text="Generate Response", command=self.start_response_generation, style="Primary.TButton"
        )
        self.generate_button.grid(row=5, column=0, pady=(0, 6), sticky="e")

        self.status_label = ttk.Label(prompt_frame, text="Ready", style="Subtle.TLabel")
        self.status_label.grid(row=6, column=0, sticky="w", pady=(4, 0))

        response_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=14)
        response_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        response_frame.grid_columnconfigure(0, weight=1)
        response_frame.grid_rowconfigure(1, weight=1)

        response_header = ttk.Frame(response_frame, style="Card.TFrame")
        response_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        response_header.grid_columnconfigure(0, weight=1)
        ttk.Label(response_header, text="Model Response", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")

        self.export_button = ttk.Button(
            response_header, text="Export Response", command=self.export_response, state=tk.DISABLED, style="Secondary.TButton"
        )
        self.export_button.grid(row=0, column=1, sticky="e")

        self.response_area = scrolledtext.ScrolledText(
            response_frame,
            wrap=tk.WORD,
            width=70,
            height=20,
            state=tk.DISABLED,
            bg=self.colors["surface_alt"],
            fg=self.colors["text"],
            insertbackground=self.colors["accent"],
            font=self.fonts["body"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
        )
        self.response_area.grid(row=1, column=0, sticky="nsew")
        self.response_area.config(state=tk.NORMAL)
        self.response_area.insert(tk.END, "Please select a model and enter a prompt to begin.")
        self.response_area.config(state=tk.DISABLED)

    def create_conversation_graph(self):
        # Create the LangGraph graph
        builder = StateGraph(ChatState)

        # Node to generate a response using the model
        builder.add_node("chatbot", self.invoke_model)
        builder.add_edge(START, "chatbot")  # Entry point
        builder.add_edge("chatbot", END)    # Exit point

        # Compile the graph with persistence
        return builder.compile(checkpointer=self.checkpointer)

    def invoke_model(self, state: ChatState):
        # Generate a response using the model
        response = self.llm.invoke(state["messages"])
        return {"messages": [AIMessage(content=response)]}

    def start_response_generation(self, event=None):
        # Disable the button while generating the response
        self.generate_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
        self.response_area.config(state=tk.NORMAL)
        self.response_area.delete("1.0", tk.END)
        self.response_area.insert(tk.END, "Thinking...")  # Show "Thinking..." while generating the response
        self.response_area.config(state=tk.DISABLED)

        selected_model = self.model_combobox.get()
        prompt = self.prompt_entry.get("1.0", tk.END).strip()

        if not selected_model or not prompt:
            messagebox.showwarning("Warning", "Please select a model and enter a prompt.")
            self.generate_button.config(state=tk.NORMAL)
            self.status_label.config(text="")
            return

        self.llm = OllamaLLM(model=selected_model)

        # Create a new thread if necessary
        if not self.current_thread:
            self.current_thread = f"thread_{time.time()}"

        # Run in a separate thread
        threading.Thread(
            target=self.generate_response,
            args=(prompt, self.current_thread),
            daemon=True
        ).start()

    def generate_response(self, prompt, thread_id):
        try:
            # Thread configuration for persistence
            config = {"configurable": {"thread_id": thread_id}}

            # Create the initial message
            initial_message = HumanMessage(content=prompt)

            # Run the graph with persistence
            for event in self.conversation_graph.stream(
                {"messages": [initial_message]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in event:
                    response = event["messages"][-1].content
                    self.response_queue.put(response)  # Only put the response in the queue

        except Exception as e:
            self.response_queue.put(f"Error: {str(e)}")
        finally:
            self.animation_active = False

    def check_response(self):
        if not self.window_open:
            return  # Stop if the window is closed

        try:
            response = self.response_queue.get_nowait()
            self.show_response(response)
            self.generate_button.config(state=tk.NORMAL)  # Enable the button
            self.export_button.config(state=tk.NORMAL)  # Enable the export button
            self.status_label.config(text="")
        except queue.Empty:
            pass
        finally:
            if self.window_open:
                self.root.after(100, self.check_response)

    def show_response(self, response):
        self.response_area.config(state=tk.NORMAL)
        self.response_area.delete("1.0", tk.END)  # Clear the "Thinking..." message

        formatted_response = response

        if "python" in self.model_combobox.get().lower():
            lexer = PythonLexer()
            tokens = lex(formatted_response, lexer)
            for token_type, value in tokens:
                self.response_area.insert(tk.END, value, token_type)
        else:
            self.response_area.insert(tk.END, formatted_response)  # Insert only the model's response

        self.response_area.config(state=tk.DISABLED)

    def export_response(self):
        response = self.response_area.get("1.0", tk.END).strip()
        if not response:
            messagebox.showwarning("Warning", "No response to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "w") as file:
                    file.write(response)
                messagebox.showinfo("Success", "Response exported successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export response: {str(e)}")

    def on_close(self):
        self.window_open = False  # Set the flag to False when the window is closed
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaInterface(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Handle window close event
    root.mainloop()

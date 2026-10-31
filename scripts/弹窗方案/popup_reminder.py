"""
光源智能体 —— 定时调光提醒弹窗
================================
启动后会在后台计时，每隔指定分钟数弹出一次调光建议窗口。

弹窗内容：
  - 显示当前光参数（只读）与推荐光参数
  - 三个按钮：是 / 否 / 自定义
  - 点击「是」或「否」后弹窗关闭
  - 点击「自定义」打开参数调节子窗口

自定义窗口：
  - 当前参数为灰色只读，不可修改
  - 可修改照度与色温
  - 「确认」应用修改，「重置」恢复推荐值
"""

import tkinter as tk
from tkinter import ttk
import time
import threading

# ──────────────────── 默认光参数 ────────────────────
DEFAULT_CURRENT_ILLUMINANCE = 1000   # lx
DEFAULT_CURRENT_COLOR_TEMP = 7000    # K
DEFAULT_REC_ILLUMINANCE = 600        # lx
DEFAULT_REC_COLOR_TEMP = 4000        # K

# 定时间隔（秒），可按需调整
POPUP_INTERVAL_SECONDS = 20


class CustomLightWindow(tk.Toplevel):
    """自定义光参数调节子窗口"""

    def __init__(self, parent, rec_illuminance, rec_color_temp, on_confirm):
        super().__init__(parent)
        self.title("自定义光参数")
        self.resizable(False, False)
        self.configure(bg="#f5f5f5")
        self.on_confirm = on_confirm

        # 推荐值（重置时用）
        self.rec_illuminance = rec_illuminance
        self.rec_color_temp = rec_color_temp

        # 当前显示值（用户可修改）
        self.current_illuminance = tk.IntVar(value=rec_illuminance)
        self.current_color_temp = tk.IntVar(value=rec_color_temp)

        self._build_ui()
        self._center_window(380, 340)
        
        # 确保自定义窗口始终在最上层
        self.attributes("-topmost", True)
        
        # 绑定点击事件，点击时置顶
        self.bind("<Button-1>", lambda e: self._bring_to_front())
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _bring_to_front(self):
        """将窗口置顶"""
        self.lift()
        self.attributes("-topmost", True)
        self.after(100, lambda: self.attributes("-topmost", False))

    # ── UI 构建 ──────────────────────────────────────
    def _build_ui(self):
        pad = {"padx": 20, "pady": 6}

        # 标题
        tk.Label(
            self, text="自定义光参数调节", font=("Microsoft YaHei UI", 14, "bold"),
            bg="#f5f5f5", fg="#333"
        ).pack(pady=(18, 10))

        # ── 当前参数（只读灰色区） ──
        readonly_frame = tk.LabelFrame(
            self, text=" 当前光参数（不可修改） ",
            font=("Microsoft YaHei UI", 10),
            bg="#e8e8e8", fg="#999", bd=1, relief="groove"
        )
        readonly_frame.pack(fill="x", **pad)

        tk.Label(
            readonly_frame, text="照度：400 lx", font=("Microsoft YaHei UI", 10),
            bg="#e8e8e8", fg="#aaa"
        ).pack(anchor="w", padx=16, pady=(6, 2))
        tk.Label(
            readonly_frame, text="色温：2000 K", font=("Microsoft YaHei UI", 10),
            bg="#e8e8e8", fg="#aaa"
        ).pack(anchor="w", padx=16, pady=(2, 8))

        # ── 调整后参数（可编辑） ──
        edit_frame = tk.LabelFrame(
            self, text=" 调整后光参数 ",
            font=("Microsoft YaHei UI", 10),
            bg="#f5f5f5", fg="#333", bd=1, relief="groove"
        )
        edit_frame.pack(fill="x", **pad)

        # 照度行
        row1 = tk.Frame(edit_frame, bg="#f5f5f5")
        row1.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(row1, text="照度 (lx)：", font=("Microsoft YaHei UI", 10),
                 bg="#f5f5f5", fg="#333").pack(side="left")
        self.entry_illuminance = tk.Spinbox(
            row1, from_=100, to=2000, increment=50, width=10,
            textvariable=self.current_illuminance,
            font=("Microsoft YaHei UI", 10)
        )
        self.entry_illuminance.pack(side="left", padx=(4, 0))

        # 色温行
        row2 = tk.Frame(edit_frame, bg="#f5f5f5")
        row2.pack(fill="x", padx=12, pady=(4, 12))
        tk.Label(row2, text="色温 (K)：", font=("Microsoft YaHei UI", 10),
                 bg="#f5f5f5", fg="#333").pack(side="left")
        self.entry_color_temp = tk.Spinbox(
            row2, from_=1800, to=10000, increment=100, width=10,
            textvariable=self.current_color_temp,
            font=("Microsoft YaHei UI", 10)
        )
        self.entry_color_temp.pack(side="left", padx=(4, 0))

        # ── 按钮区 ──
        btn_frame = tk.Frame(self, bg="#f5f5f5")
        btn_frame.pack(pady=(8, 16))

        tk.Button(
            btn_frame, text="确  认", width=10, font=("Microsoft YaHei UI", 10),
            bg="#4CAF50", fg="white", relief="flat", cursor="hand2",
            command=self._confirm
        ).pack(side="left", padx=12)

        tk.Button(
            btn_frame, text="重  置", width=10, font=("Microsoft YaHei UI", 10),
            bg="#ff9800", fg="white", relief="flat", cursor="hand2",
            command=self._reset
        ).pack(side="left", padx=12)

    # ── 事件 ─────────────────────────────────────────
    def _confirm(self):
        """确认 → 把用户自定义值回传"""
        ill = self.current_illuminance.get()
        ct = self.current_color_temp.get()
        self.on_confirm(ill, ct)
        self.destroy()

    def _reset(self):
        """重置 → 恢复推荐值"""
        self.current_illuminance.set(self.rec_illuminance)
        self.current_color_temp.set(self.rec_color_temp)

    def _on_close(self):
        """关闭窗口等同于取消，不做任何应用"""
        self.destroy()

    def _center_window(self, w, h):
        self.update_idletasks()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")


class LightReminderApp:
    """主应用：后台计时 + 弹窗"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("光源智能体 —— 调光提醒")
        self.root.configure(bg="#fafafa")

        # 光参数状态
        self.current_illuminance = DEFAULT_CURRENT_ILLUMINANCE
        self.current_color_temp = DEFAULT_CURRENT_COLOR_TEMP
        self.rec_illuminance = DEFAULT_REC_ILLUMINANCE
        self.rec_color_temp = DEFAULT_REC_COLOR_TEMP

        # 弹窗引用
        self.popup = None
        self.custom_win = None

        # 计时器
        self.interval_seconds = POPUP_INTERVAL_SECONDS
        self._timer_running = True

        self._build_main_ui()
        self._center_main(520, 360)
        self._schedule_popup()

    # ── 主界面（简易状态面板） ─────────────────────────
    def _build_main_ui(self):
        tk.Label(
            self.root, text="💡 光源智能体 · 调光提醒服务",
            font=("Microsoft YaHei UI", 15, "bold"),
            bg="#fafafa", fg="#333"
        ).pack(pady=(24, 8))

        self.status_label = tk.Label(
            self.root, text="运行中 · 下次提醒将在计时结束后弹出",
            font=("Microsoft YaHei UI", 10),
            bg="#fafafa", fg="#666"
        )
        self.status_label.pack()

        # 当前参数展示
        info_frame = tk.Frame(self.root, bg="#fafafa")
        info_frame.pack(pady=16)
        self.lbl_current = tk.Label(
            info_frame,
            text=f"当前光参数   照度：{self.current_illuminance} lx  色温：{self.current_color_temp} K",
            font=("Microsoft YaHei UI", 10), bg="#fafafa", fg="#555"
        )
        self.lbl_current.pack()
        self.lbl_recommend = tk.Label(
            info_frame,
            text=f"推荐光参数   照度：{self.rec_illuminance} lx  色温：{self.rec_color_temp} K",
            font=("Microsoft YaHei UI", 10), bg="#fafafa", fg="#555"
        )
        self.lbl_recommend.pack(pady=(4, 0))

        # 倒计时
        self.countdown_label = tk.Label(
            self.root, text="", font=("Microsoft YaHei UI", 20, "bold"),
            bg="#fafafa", fg="#1976D2"
        )
        self.countdown_label.pack(pady=8)

        # 手动触发按钮（方便测试）
        btn_frame = tk.Frame(self.root, bg="#fafafa")
        btn_frame.pack(pady=4)
        tk.Button(
            btn_frame, text="立即弹出提醒（测试用）",
            font=("Microsoft YaHei UI", 9),
            bg="#1976D2", fg="white", relief="flat", cursor="hand2",
            command=self._show_popup
        ).pack(side="left", padx=8)
        tk.Button(
            btn_frame, text="重置计时",
            font=("Microsoft YaHei UI", 9),
            bg="#ff9800", fg="white", relief="flat", cursor="hand2",
            command=self._reset_timer
        ).pack(side="left", padx=8)

        # 日志区
        self.log_text = tk.Text(
            self.root, height=5, width=50,
            font=("Consolas", 9), bg="#263238", fg="#aed581",
            relief="flat", state="disabled"
        )
        self.log_text.pack(pady=12, padx=20, fill="x")

    # ── 倒计时刷新 ────────────────────────────────────
    def _tick(self, remaining):
        if remaining < 0 or not self._timer_running:
            return
        mins, secs = divmod(remaining, 60)
        self.countdown_label.config(text=f"⏱  {mins:02d}:{secs:02d}")
        self.root.after(1000, self._tick, remaining - 1)

    # ── 定时调度 ──────────────────────────────────────
    def _schedule_popup(self):
        self._timer_running = True
        self._tick(self.interval_seconds)
        self.root.after(self.interval_seconds * 1000, self._show_popup)
        self._log(f"计时开始，{POPUP_INTERVAL_SECONDS} 秒后弹出提醒")

    def _reset_timer(self):
        self._timer_running = False
        self.countdown_label.config(text="")
        self._timer_running = True
        self._schedule_popup()
        self._log("计时已重置")

    # ── 弹窗 ──────────────────────────────────────────
    def _show_popup(self):
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.lift()
            return

        self._timer_running = False
        self.countdown_label.config(text="")
        self._log("弹出调光提醒")

        win = tk.Toplevel(self.root)
        win.title("调光提醒")
        win.resizable(False, False)
        win.configure(bg="#ffffff")
        self.popup = win

        # 居中
        w, h = 460, 320
        x = (win.winfo_screenwidth() - w) // 2
        y = (win.winfo_screenheight() - h) // 2
        win.geometry(f"{w}x{h}+{x}+{y}")
        win.attributes("-topmost", True)
        
        # 绑定点击事件，点击时置顶
        def popup_click_topmost(e):
            win.lift()
            win.attributes("-topmost", True)
            win.after(100, lambda: win.attributes("-topmost", False))
        win.bind("<Button-1>", popup_click_topmost)
        
        win.protocol("WM_DELETE_WINDOW", lambda: self._on_popup_close(win))

        # ── 内容 ──
        tk.Label(
            win, text="💡 调光建议", font=("Microsoft YaHei UI", 14, "bold"),
            bg="#ffffff", fg="#333"
        ).pack(pady=(20, 8))

        # 提示文字（静态部分）
        tk.Label(
            win, text="目前检测到你心情低落，是否使用推荐的调光策略：",
            font=("Microsoft YaHei UI", 10),
            bg="#ffffff", fg="#444", justify="left", wraplength=400
        ).pack(padx=24, pady=(4, 2))

        # 当前光参数（静态）
        tk.Label(
            win,
            text=f"  当前光参数  照度：{self.current_illuminance} lx，色温：{self.current_color_temp} K",
            font=("Microsoft YaHei UI", 10),
            bg="#ffffff", fg="#444", justify="left", wraplength=400
        ).pack(padx=24, pady=2, anchor="w")

        # 调整后光参数（动态，可被自定义窗口更新）
        self.popup_adjust_label = tk.Label(
            win,
            text=f"  调整后光参数  照度：{self.rec_illuminance} lx，色温：{self.rec_color_temp} K",
            font=("Microsoft YaHei UI", 10),
            bg="#ffffff", fg="#1976D2", justify="left", wraplength=400
        )
        self.popup_adjust_label.pack(padx=24, pady=2, anchor="w")

        # ── 按钮 ──
        btn_frame = tk.Frame(win, bg="#ffffff")
        btn_frame.pack(pady=20)

        tk.Button(
            btn_frame, text="是（应用推荐）", width=14,
            font=("Microsoft YaHei UI", 10),
            bg="#4CAF50", fg="white", relief="flat", cursor="hand2",
            command=lambda: self._on_yes(win)
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="否", width=8,
            font=("Microsoft YaHei UI", 10),
            bg="#bdbdbd", fg="#333", relief="flat", cursor="hand2",
            command=lambda: self._on_no(win)
        ).pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="自定义", width=8,
            font=("Microsoft YaHei UI", 10),
            bg="#1976D2", fg="white", relief="flat", cursor="hand2",
            command=lambda: self._on_custom(win)
        ).pack(side="left", padx=8)

    # ── 弹窗按钮回调 ──────────────────────────────────
    def _on_popup_close(self, win):
        win.destroy()
        self.popup = None
        self._restart_timer()

    def _on_yes(self, win):
        """应用推荐值"""
        self.current_illuminance = self.rec_illuminance
        self.current_color_temp = self.rec_color_temp
        self._log(f"已应用推荐参数 → 照度 {self.rec_illuminance} lx，色温 {self.rec_color_temp} K")
        win.destroy()
        self.popup = None
        self._update_info()
        self._restart_timer()

    def _on_no(self, win):
        """不应用"""
        self._log("用户拒绝调光建议")
        win.destroy()
        self.popup = None
        self._restart_timer()

    def _on_custom(self, win):
        """打开自定义窗口"""
        if self.custom_win is not None and self.custom_win.winfo_exists():
            self.custom_win.lift()
            return

        def on_confirm(ill, ct):
            self.current_illuminance = ill
            self.current_color_temp = ct
            self._log(f"已应用自定义参数 → 照度 {ill} lx，色温 {ct} K")
            # 只关闭自定义窗口，保留调光提醒弹窗
            self.custom_win = None
            # 更新弹窗中「调整后光参数」的显示
            if hasattr(self, 'popup_adjust_label') and self.popup_adjust_label.winfo_exists():
                self.popup_adjust_label.config(
                    text=f"  调整后光参数  照度：{ill} lx，色温：{ct} K"
                )
            self._update_info()

        self.custom_win = CustomLightWindow(
            win, self.rec_illuminance, self.rec_color_temp, on_confirm
        )

    # ── 辅助 ──────────────────────────────────────────
    def _restart_timer(self):
        """弹窗关闭后重新开始计时"""
        self._timer_running = True
        self._schedule_popup()

    def _update_info(self):
        """刷新主窗口参数显示"""
        self.lbl_current.config(
            text=f"当前光参数   照度：{self.current_illuminance} lx  色温：{self.current_color_temp} K"
        )

    def _log(self, msg):
        """写日志"""
        self.log_text.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _center_main(self, w, h):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def run(self):
        self.root.mainloop()


# ──────────────────── 入口 ────────────────────────────
if __name__ == "__main__":
    app = LightReminderApp()
    app.run()

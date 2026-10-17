"""Local human-versus-AI Koi-Koi, using the existing reference rules and agents."""

from __future__ import annotations

import argparse
import logging
import secrets
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.font import nametofont

import numpy as np

from .layout import (
    CONTINUE,
    DEALER,
    DISCARD,
    HISTORY,
    KOI,
    KOIS,
    PENDING,
    PHASE,
    PICK_DRAW,
    PICK_HAND,
    PLAYER,
    POINTS,
    ROUND,
    STOP,
    TERMINAL,
    TURN,
)
from .reference import BLUE, KASU, LIGHTS, RED, RIBBONS, TANE, ReferenceEnv

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/cold_ppo_screen/seed6710001_warm/update_00000520.pt"
CHECKPOINT_SHA256 = "37d06849261e0c1f0c31a9d2f228388fa1f391b670ce6772093a1c276091cc36"
ASSETS = ROOT / "external/KoiKoi-AI/resource/cardpng"
OPPONENTS = {
    "trained": "今回の学習済み AI",
    "rl-wp": "原論文 RL-WP",
    "rl-point": "原論文 RL-Point",
    "greedy": "練習用 AI",
}
MONTHS = ("松", "梅", "桜", "藤", "菖蒲", "牡丹", "萩", "芒", "菊", "紅葉", "柳", "桐")
BG, FELT, PANEL, TEXT, MUTED, GOLD = (
    "#132b28",
    "#1c3d36",
    "#112824",
    "#f6f1e3",
    "#b0c7ba",
    "#f1cc78",
)


def card_name(card):
    kind = (
        "光"
        if card in LIGHTS
        else "たね"
        if card in TANE
        else "短冊"
        if card in RIBBONS
        else "かす"
    )
    return f"{card // 4 + 1}月 {MONTHS[card // 4]}・{kind}"


def yaku_names(cards):
    """Labels only; all scoring remains in reference.score."""
    cards = set(cards)
    names = []
    lights = len(cards & LIGHTS)
    if lights == 5:
        names.append("五光")
    elif lights == 4:
        names.append("雨四光" if 40 in cards else "四光")
    elif lights == 3 and 40 not in cards:
        names.append("三光")
    for needed, name in (
        ({20, 24, 36}, "猪鹿蝶"),
        (RED, "赤短"),
        (BLUE, "青短"),
        (RED | BLUE, "赤短・青短の加点"),
        ({8, 32}, "花見酒"),
        ({28, 32}, "月見酒"),
    ):
        if needed <= cards:
            names.append(name)
    for group, minimum, name in (
        (TANE, 5, "たね"),
        (RIBBONS, 5, "短冊"),
        (KASU, 10, "かす"),
    ):
        if len(cards & group) >= minimum:
            names.append(name)
    return "・".join(names) or "役はまだありません"


def load_opponent(name):
    import torch

    from .agents import EndgameAgent, GreedyAgent, NeuralAgent
    from .upstream import PublishedAgent

    torch.set_num_threads(2)
    if name == "trained":
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"学習済みモデルがありません。\n{CHECKPOINT}\nREADME の復元手順を確認してください。"
            )
        agent = NeuralAgent(CHECKPOINT, device="cpu", deterministic=True)
        if agent.checkpoint_sha256 != CHECKPOINT_SHA256 or agent.profile != "paper8":
            raise ValueError(
                "採用モデルの識別情報が一致しません。元のチェックポイントを復元してください。"
            )
    elif name == "greedy":
        agent = GreedyAgent()
    else:
        agent = PublishedAgent("wp" if name == "rl-wp" else "point")
    return EndgameAgent(agent)


class PlaySession:
    """One match: human is seat 0. Pause at round boundaries, never auto-restart."""

    def __init__(self, agent, seed=None):
        self.agent = agent
        # Casual games are not evaluation records; avoid reserved final-test seeds.
        self.env = ReferenceEnv(
            seed if seed is not None else secrets.randbelow(90_000_000) + 1, "paper8"
        )
        self.paused = False
        self.result = ""
        self.finished_round = 0
        self.moves = 0
        self.log = ["8局・各30点で開始。親は「親」マークの側です。"]

    @property
    def finished(self):
        return self.env.state[PHASE] == TERMINAL

    @property
    def human_turn(self):
        return not self.paused and not self.finished and self.env.state[PLAYER] == 0

    def human_action(self, action):
        if not self.human_turn:
            raise ValueError("今はあなたの操作待ちではありません。")
        self._step(action)

    def ai_action(self):
        if self.paused or self.finished or self.env.state[PLAYER] != 1:
            raise ValueError("今はAIの手番ではありません。")
        # The same observation-only inference adapter used in the final evaluation.
        self._step(self.agent.act(self.env))

    def _step(self, action):
        before = self.env.state.copy()
        player, phase = int(before[PLAYER]), int(before[PHASE])
        who = "あなた" if player == 0 else "AI"
        result = self.env.step(action)  # Validates legality before any mutation.
        self.moves += 1
        verb = "を出しました" if phase == DISCARD else "を取りました"
        detail = (
            ("上がり" if action == STOP else "こいこい")
            if phase == KOI
            else card_name(action) + verb
        )
        self.log.append(f"{who}：{detail}")
        if result[1]:
            self.paused = True
            self.finished_round = int(before[ROUND]) + 1
            winner = "あなた" if result[3] > 0 else "AI"
            self.result = (
                f"第{self.finished_round}局：{winner}が {abs(int(result[3]))} 点獲得"
            )
            if self.finished:
                difference = int(self.env.state[POINTS] - self.env.state[POINTS + 1])
                outcome = (
                    "あなたの勝ち！"
                    if difference > 0
                    else "AIの勝ち"
                    if difference < 0
                    else "引き分け"
                )
                self.result += f"　／　対戦終了・{outcome}"
            self.log.append(self.result)
        else:
            history = HISTORY + int(before[TURN]) * 5
            drawn = int(self.env.state[history + 2]) - 1
            if drawn >= 0 and before[history + 2] == 0:
                self.log.append(f"{who}の山札：{card_name(drawn)}")

    def next_round(self):
        if not self.paused or self.finished:
            raise ValueError("次の局には進めません。")
        self.paused = False
        self.result = ""

    def view(self):
        """Explicit public/human view: never return AI hand IDs, stock or RNG."""
        s = self.env.state
        return {
            "hand": np.flatnonzero(s[:48] == 0).tolist(),
            "field": np.flatnonzero(s[:48] == 2).tolist(),
            "captured": [np.flatnonzero(s[:48] == 3 + p).tolist() for p in range(2)],
            "opponent_count": int(np.count_nonzero(s[:48] == 1)),
            "pending": int(s[PENDING]),
            "points": s[POINTS : POINTS + 2].tolist(),
            "koi": s[KOIS : KOIS + 2].tolist(),
            "yaku": [self.env.yaku(p) for p in range(2)],
            "dealer": int(s[DEALER]),
            "round": int(s[ROUND]) + 1,
            "phase": int(s[PHASE]),
            "legal": np.flatnonzero(self.env.legal()).tolist()
            if self.human_turn
            else [],
        }


class PlayWindow:
    def __init__(self, root, opponent="trained", seed=None, agent=None):
        self.root, self.seed, self.timer = root, seed, None
        self.session = None
        self.loaded_name = opponent
        self.agent = agent
        self.error = False
        root.title("こいこい — Human vs AI")
        width = min(1180, root.winfo_screenwidth() - 80)
        height = min(850, root.winfo_screenheight() - 120)
        root.geometry(f"{width}x{height}+30+30")
        root.minsize(1000, 750)
        root.configure(bg=BG)
        root.option_add("*Font", ("Yu Gothic UI", 10))
        nametofont("TkDefaultFont", root=root).configure(family="Yu Gothic UI", size=10)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TButton", padding=(14, 7), font=("Yu Gothic UI", 10))
        style.configure("TCombobox", padding=5)
        self.photos = {}
        for c in range(-1, 48):
            filename = "0-0.png" if c == -1 else f"{c // 4 + 1}-{c % 4 + 1}.png"
            self.photos[c] = tk.PhotoImage(master=root, file=str(ASSETS / filename))
        self.big = {c: im.zoom(3).subsample(2) for c, im in self.photos.items()}
        self.compact = {c: im.zoom(3).subsample(4) for c, im in self.photos.items()}
        self.small = {c: im.subsample(2) for c, im in self.photos.items()}

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=24, pady=(16, 10))
        tk.Label(
            header, text="こいこい", font=("Yu Gothic UI", 25, "bold"), fg=GOLD, bg=BG
        ).pack(side="left")
        ttk.Button(header, text="遊び方・役", command=self.help).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(header, text="最初から", command=self.restart).pack(
            side="right", padx=8
        )
        self.choice = ttk.Combobox(
            header, state="readonly", values=list(OPPONENTS.values()), width=23
        )
        self.choice.set(OPPONENTS[opponent])
        self.choice.pack(side="right")
        self.score = tk.Label(root, bg=BG, fg=TEXT, font=("Yu Gothic UI", 15, "bold"))
        self.score.pack(fill="x", padx=24, pady=(0, 8))

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=20)
        self.side = tk.Canvas(body, width=280, bg=PANEL, highlightthickness=0)
        self.side.pack(side="right", fill="y", padx=(12, 0))
        self.board = tk.Canvas(body, bg=FELT, highlightthickness=0)
        self.board.pack(side="left", fill="both", expand=True)
        self.board.bind("<Configure>", lambda event: self.render())
        self.board.bind("<Button-1>", self.click_card)
        self.board.bind("<Motion>", self.hover_card)
        self.board.bind("<Leave>", lambda event: self.board.config(cursor=""))
        self.side.bind("<Configure>", lambda event: self.render())

        controls = tk.Frame(root, bg=BG)
        controls.pack(fill="x", padx=24, pady=10)
        self.status = tk.Label(
            controls,
            text="AIを読み込んでいます…",
            bg=BG,
            fg=GOLD,
            anchor="w",
            wraplength=680,
        )
        self.status.pack(side="left", fill="x", expand=True)
        self.next_button = ttk.Button(controls, text="次の局へ", command=self.advance)
        self.next_button.pack(side="right", padx=(8, 0))
        self.koi_button = ttk.Button(
            controls, text="こいこい", command=lambda: self.play(CONTINUE)
        )
        self.koi_button.pack(side="right", padx=(8, 0))
        self.stop_button = ttk.Button(
            controls, text="上がり", command=lambda: self.play(STOP)
        )
        self.stop_button.pack(side="right")
        self.log_label = tk.Label(
            root, text="", bg=BG, fg=MUTED, anchor="w", justify="left", height=3
        )
        self.log_label.pack(fill="x", padx=24, pady=(0, 12))
        root.update_idletasks()
        self.start(opponent, seed)

    def start(self, opponent, seed=None):
        self.cancel_timer()
        try:
            if self.agent is None or opponent != self.loaded_name:
                self.status.config(text="AIを読み込んでいます…")
                self.root.update_idletasks()
                agent = load_opponent(opponent)
                self.agent, self.loaded_name = agent, opponent
            self.session = PlaySession(self.agent, seed)
            self.error = False
        except Exception as exc:
            self.choice.set(OPPONENTS[self.loaded_name])
            if self.session is None:
                raise
            messagebox.showerror("AIを読み込めません", str(exc), parent=self.root)
        self.render()
        self.schedule_ai()

    def cancel_timer(self):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None

    def schedule_ai(self):
        self.cancel_timer()
        if (
            self.session
            and not self.error
            and not self.session.paused
            and not self.session.finished
            and not self.session.human_turn
        ):
            self.timer = self.root.after(700, self.ai_step)

    def ai_step(self):
        self.timer = None
        try:
            self.session.ai_action()
        except Exception as exc:
            logger.exception("AI inference failed; match paused")
            self.error = True
            messagebox.showerror("対戦を停止しました", str(exc), parent=self.root)
        self.render()
        self.schedule_ai()

    def play(self, action):
        if not self.session.human_turn or self.error:
            return
        try:
            self.session.human_action(action)
        except ValueError:
            return
        self.render()
        self.schedule_ai()

    def advance(self):
        if self.session.paused and not self.session.finished:
            self.session.next_round()
            self.render()
            self.schedule_ai()

    def restart(self):
        self.cancel_timer()
        if (
            self.session.moves
            and not self.session.finished
            and not messagebox.askyesno(
                "新しい対戦",
                "現在の対戦を終了して、選択したAIと最初から遊びますか？",
                parent=self.root,
            )
        ):
            self.schedule_ai()
            return
        opponent = next(
            key for key, value in OPPONENTS.items() if value == self.choice.get()
        )
        self.start(opponent)

    def close(self):
        self.cancel_timer()
        self.root.destroy()

    def card_under_pointer(self):
        for item in self.board.find_withtag("current"):
            for tag in self.board.gettags(item):
                if tag.startswith("play-"):
                    return int(tag.removeprefix("play-"))
        return None

    def click_card(self, event):
        action = self.card_under_pointer()
        if action is not None:
            self.play(action)

    def hover_card(self, event):
        self.board.config(
            cursor="hand2" if self.card_under_pointer() is not None else ""
        )

    def draw_card(
        self,
        canvas,
        card,
        x,
        y,
        *,
        large=True,
        compact=False,
        active=False,
        label=False,
    ):
        photo = (
            self.compact[card]
            if compact
            else self.big[card]
            if large
            else self.photos[card]
        )
        tag = (f"card-{card}", f"play-{card}") if active else (f"card-{card}",)
        w, h = photo.width(), photo.height()
        canvas.create_rectangle(
            x - 3,
            y - 3,
            x + w + 3,
            y + h + 3,
            fill=GOLD if active else PANEL,
            outline="",
            tags=tag,
        )
        canvas.create_image(x, y, image=photo, anchor="nw", tags=tag)
        if label and card >= 0:
            canvas.create_text(
                x + w / 2,
                y + h + 11,
                text=f"{card // 4 + 1}月 {MONTHS[card // 4]}",
                fill=MUTED,
                font=("Yu Gothic UI", 9),
                tags=tag,
            )

    def render(self):
        if self.session is None:
            return
        game, v = self.session, self.session.view()
        b, side = self.board, self.side
        b.delete("all")
        b.config(cursor="")
        side.delete("all")
        w, h = max(b.winfo_width(), 660), max(b.winfo_height(), 470)
        active = set(v["legal"]) if not self.error else set()
        round_no = game.finished_round if game.paused else v["round"]
        self.score.config(
            text=f"第 {round_no} / 8 局     あなた {v['points'][0]} 点   ─   AI {v['points'][1]} 点     ·     {OPPONENTS[self.loaded_name]}"
        )
        self.log_label.config(text="\n".join(game.log[-3:]))
        for button, enabled in (
            (self.stop_button, STOP in active),
            (self.koi_button, CONTINUE in active),
            (self.next_button, game.paused and not game.finished),
        ):
            button.config(state="normal" if enabled else "disabled")

        if game.paused:
            b.create_text(
                w / 2,
                h / 2 - 35,
                text=game.result.replace("　／　", "\n"),
                width=w - 70,
                justify="center",
                fill=GOLD,
                font=("Yu Gothic UI", 21, "bold"),
            )
            b.create_text(
                w / 2,
                h / 2 + 45,
                text="「最初から」で新しい対戦を始められます。"
                if game.finished
                else "「次の局へ」で続けます。",
                fill=TEXT,
                font=("Yu Gothic UI", 12),
            )
            self.status.config(
                text="対戦終了"
                if game.finished
                else "局が終了しました。結果を確認して次へ進んでください。"
            )
            return

        b.create_text(
            18,
            16,
            anchor="nw",
            text="AI の手札" + ("  親" if v["dealer"] == 1 else ""),
            fill=MUTED,
        )
        for i in range(v["opponent_count"]):
            self.draw_card(b, -1, 20 + i * 55, 43, large=False)
        if v["pending"] >= 0:
            self.draw_card(b, v["pending"], w - 95, 28, large=False)
            b.create_text(
                w - 70, 117, text="選択中の札", fill=GOLD, font=("Yu Gothic UI", 9)
            )
        b.create_line(18, 140, w - 18, 140, fill="#416357")
        b.create_text(
            18, 152, anchor="nw", text="場の札  ·  同じ月の札を合わせます", fill=MUTED
        )

        field = v["field"]
        columns = max(1, int((w - 35) // 82))
        rows = max(1, (len(field) + columns - 1) // columns)
        field_space = h - 355
        large = rows * 136 <= field_space
        compact = False
        if not large:
            columns = max(1, int((w - 35) // 57))
            rows = max(1, (len(field) + columns - 1) // columns)
            compact = rows * 96 > field_space
            if compact:
                columns = max(1, int((w - 35) // 52))
        for i, c in enumerate(field):
            x = 22 + (i % columns) * (52 if compact else 82 if large else 57)
            y = 182 + (i // columns) * (78 if compact else 136 if large else 96)
            self.draw_card(
                b, c, x, y, large=large, compact=compact, active=c in active, label=True
            )

        hand_y = h - 136
        b.create_text(
            18,
            hand_y - 29,
            anchor="nw",
            text="あなたの手札" + ("  親" if v["dealer"] == 0 else ""),
            fill=TEXT,
        )
        for i, c in enumerate(v["hand"]):
            self.draw_card(b, c, 22 + i * 82, hand_y, active=c in active, label=True)

        for p, title in ((1, "AI の取り札"), (0, "あなたの取り札")):
            top = 12 + (0 if p == 1 else max(235, side.winfo_height() / 2))
            cards = set(v["captured"][p])
            side.create_text(
                14,
                top,
                anchor="nw",
                text=f"{title}   役 {v['yaku'][p]}点",
                fill=TEXT,
                font=("Yu Gothic UI", 12, "bold"),
            )
            side.create_text(
                14, top + 27, anchor="nw", text=f"こいこい {v['koi'][p]}回", fill=MUTED
            )
            for row, (group, name) in enumerate(
                ((LIGHTS, "光"), (TANE, "たね"), (RIBBONS, "短冊"), (KASU, "かす"))
            ):
                collected = sorted(cards & group)
                y = top + 48 + row * 38
                side.create_text(
                    14, y + 15, anchor="w", text=f"{name} {len(collected)}", fill=MUTED
                )
                spacing = min(27, 175 / max(1, len(collected) - 1))
                for i, c in enumerate(collected):
                    side.create_image(
                        76 + i * spacing, y, anchor="nw", image=self.small[c]
                    )
            side.create_text(
                14,
                top + 207,
                anchor="nw",
                text=yaku_names(cards),
                width=252,
                fill=GOLD,
                font=("Yu Gothic UI", 9),
            )

        if self.error:
            prompt = "エラーで停止しました。「最初から」で再開してください。"
        elif not game.human_turn:
            prompt = "AI の手番です…"
        elif v["phase"] == DISCARD:
            prompt = "出す手札をクリックしてください。金色の枠が選べる札です。"
        elif v["phase"] in (PICK_HAND, PICK_DRAW):
            prompt = "同じ月の札が2枚あります。場から取る札を選んでください。"
        else:
            prompt = f"役ができました！ 上がると {v['yaku'][0]} 点獲得。続けるなら「こいこい」。"
        self.status.config(text=prompt)

    def help(self):
        self.cancel_timer()
        messagebox.showinfo(
            "遊び方・このゲームのルール",
            (
                "① 金色の枠の手札をクリックして出します。\n"
                "② 同じ月の場札を取ります。候補が2枚なら1枚を選びます。\n"
                "③ 山札は自動でめくります。取る候補が2枚なら選びます。\n"
                "④ 役が増えたら「上がり」か「こいこい」を選びます。\n\n"
                "8局、各30点。0点以下で終了し、最終得点が多い側の勝ち。\n"
                "五光10、四光8、雨四光7、三光5、猪鹿蝶5。\n"
                "赤短5、青短5、両方で追加10。たね・短冊は5枚から、かすは10枚から加点。\n"
                "花見酒・月見酒は各1点、こいこい後は各3点。盃はたね・かす兼用。\n"
                "こいこい3回までは回数を加点、4回以降は基本点×(回数−2)。\n"
                "最終手番は役が増えれば自動で上がり。流局は親が1点獲得。\n"
                "7点倍付け・相手のこいこいによる倍付けはありません。\n\n"
                "AIの変更は上の一覧で選び「最初から」。通信・学習・棋譜保存は行いません。"
            ),
            parent=self.root,
        )
        self.schedule_ai()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", choices=OPPONENTS, default="trained")
    parser.add_argument(
        "--seed", type=int, help="Optional repeatable casual deal (1..99999999)"
    )
    args = parser.parse_args()
    if args.seed is not None and not 1 <= args.seed < 100_000_000:
        parser.error("--seed must be between 1 and 99999999")
    root = tk.Tk()
    try:
        PlayWindow(root, args.opponent, args.seed)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror(
            "こいこいを起動できません",
            f"{exc}\n\nREADME の「人とAIの対戦」を確認してください。",
            parent=root,
        )
        root.destroy()
        raise


if __name__ == "__main__":
    main()

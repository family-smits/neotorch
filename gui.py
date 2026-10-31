#!/usr/bin/env python3
"""NeoTorch Studio — Pygame GUI met token-bar, AI-chat en 3D-preview."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

from neotorch import Model3DGenerator, export_obj, export_stl
from neotorch.mesh import PRACTICAL_MAX_VERTS
from neotorch.paths import icon_path, output_dir
from neotorch.preview import MeshPreview
from neotorch.tokens import TokenWallet


# Palet: donkere studio + amber accent
BG = (12, 14, 18)
PANEL = (22, 26, 34)
PANEL_2 = (28, 33, 44)
STROKE = (48, 56, 72)
STROKE_SOFT = (38, 44, 58)
TEXT = (232, 236, 242)
TEXT_DIM = (140, 150, 168)
TEXT_MUTED = (96, 108, 128)
ACCENT = (232, 168, 74)
ACCENT_DIM = (168, 118, 42)
OK = (88, 196, 140)
WARN = (220, 120, 80)
DANGER = (210, 78, 78)

W, H = 1280, 760
TOKEN_W = 168
AI_W = 360
MARGIN = 16


@dataclass
class ChatLine:
    role: str  # "user" | "ai" | "sys"
    text: str


def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def draw_round_rect(
    surf: pygame.Surface,
    color: Tuple[int, int, int],
    rect: pygame.Rect,
    radius: int = 12,
    width: int = 0,
) -> None:
    pygame.draw.rect(surf, color, rect, width=width, border_radius=radius)


def lerp_color(c1, c2, t: float):
    t = clamp(t, 0.0, 1.0)
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Fonts:
    def __init__(self):
        self.title = pygame.font.SysFont("segoeui", 22, bold=True)
        self.head = pygame.font.SysFont("segoeui", 16, bold=True)
        self.body = pygame.font.SysFont("segoeui", 15)
        self.small = pygame.font.SysFont("segoeui", 13)
        self.tiny = pygame.font.SysFont("segoeui", 11)
        self.big_num = pygame.font.SysFont("segoeui", 36, bold=True)


class NeoTorchStudio:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("NeoTorch Studio")
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        try:
            ico = icon_path()
            if ico.exists():
                pygame.display.set_icon(pygame.image.load(str(ico)))
        except Exception:
            pass
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()
        self.wallet = TokenWallet()
        self.generator = Model3DGenerator(detail=20, deform_strength=0.05, max_verts=100_000)
        self.preview = MeshPreview()
        self.running = True
        self._preview_buf: Optional[pygame.Surface] = None
        self._preview_buf_size = (0, 0)

        self.input_text = ""
        self.caret_on = True
        self.caret_t = 0.0
        self.chat: List[ChatLine] = [
            ChatLine("ai", "Welkom bij NeoTorch Studio."),
            ChatLine("ai", "Typ ÁLLES — elk woord wordt 3D. Geen vaste lijst."),
            ChatLine("sys", "bv. 'paarse pizza-draak', 'laptop op een berg', 'ninja met vleugels'"),
        ]
        self.status = "Klaar"
        self.last_cost = 0
        self.last_shape = "—"
        self.last_diff = "—"
        self.last_verts_est = 0
        self.last_colors: list = []
        self.anim = 0.0
        self.gen_flash = 0.0
        self.scroll = 0
        self.export_msg = ""
        self.export_t = 0.0

        # Layout rects (bijgewerkt in layout())
        self.token_rect = pygame.Rect(0, 0, 0, 0)
        self.ai_rect = pygame.Rect(0, 0, 0, 0)
        self.preview_rect = pygame.Rect(0, 0, 0, 0)
        self.input_rect = pygame.Rect(0, 0, 0, 0)
        self.gen_btn = pygame.Rect(0, 0, 0, 0)
        self.export_obj_btn = pygame.Rect(0, 0, 0, 0)
        self.export_stl_btn = pygame.Rect(0, 0, 0, 0)
        self.photo_btn = pygame.Rect(0, 0, 0, 0)
        self.layout()

        self._update_estimate()

    def layout(self) -> None:
        w, h = self.screen.get_size()
        top = MARGIN + 8
        bottom = h - MARGIN
        left = MARGIN

        self.token_rect = pygame.Rect(left, top, TOKEN_W, bottom - top)
        self.ai_rect = pygame.Rect(self.token_rect.right + MARGIN, top, AI_W, bottom - top)
        self.preview_rect = pygame.Rect(
            self.ai_rect.right + MARGIN,
            top,
            w - self.ai_rect.right - MARGIN * 2,
            bottom - top,
        )

        pad = 14
        self.input_rect = pygame.Rect(
            self.ai_rect.left + pad,
            self.ai_rect.bottom - 108,
            self.ai_rect.width - pad * 2,
            44,
        )
        self.gen_btn = pygame.Rect(
            self.ai_rect.left + pad,
            self.ai_rect.bottom - 52,
            self.ai_rect.width - pad * 2,
            36,
        )
        bw = 96
        gap = 8
        right = self.preview_rect.right - 14
        self.photo_btn = pygame.Rect(right - bw, self.preview_rect.top + 14, bw, 30)
        self.export_stl_btn = pygame.Rect(self.photo_btn.left - gap - bw, self.preview_rect.top + 14, bw, 30)
        self.export_obj_btn = pygame.Rect(self.export_stl_btn.left - gap - bw, self.preview_rect.top + 14, bw, 30)

    def _update_estimate(self) -> None:
        prompt = self.input_text.strip() or "model"
        cost, shape, diff = self.generator.estimate_cost(prompt)
        plan = self.generator.parse(prompt)
        self.last_cost = cost
        self.last_shape = shape
        self.last_diff = diff
        self.last_verts_est = self.generator.estimate_verts(prompt)
        cols = list(plan.colors)
        for a in plan.actors:
            for c in a.colors:
                if c not in cols:
                    cols.append(c)
        # Toon ook object-palette preview kleuren
        from neotorch.colors import expand_palette
        if plan.actors:
            cols = expand_palette(cols, obj=plan.actors[0].obj, min_colors=5)
        self.last_colors = cols[:6]
        self.last_shape = plan.label + (" · SCENE" if plan.is_scene else "")

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(30) / 1000.0
            self.anim += dt
            self.caret_t += dt
            if self.caret_t >= 0.5:
                self.caret_t = 0.0
                self.caret_on = not self.caret_on
            if self.gen_flash > 0:
                self.gen_flash = max(0.0, self.gen_flash - dt)
            if self.export_t > 0:
                self.export_t = max(0.0, self.export_t - dt)

            for event in pygame.event.get():
                self.handle_event(event)

            self.preview.update(dt)
            self.draw()
            pygame.display.flip()

        pygame.quit()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((max(1000, event.w), max(600, event.h)), pygame.RESIZABLE)
            self.layout()
            return

        self.preview.handle_event(event, self.preview_rect)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.gen_btn.collidepoint(event.pos):
                self.generate()
            elif self.export_obj_btn.collidepoint(event.pos):
                self.export("obj")
            elif self.export_stl_btn.collidepoint(event.pos):
                self.export("stl")
            elif self.photo_btn.collidepoint(event.pos):
                self.render_photoreal()
            elif event.pos[1] > self.ai_rect.top + 50 and self.ai_rect.collidepoint(event.pos):
                # scroll zone
                pass

        if event.type == pygame.MOUSEWHEEL and self.ai_rect.collidepoint(pygame.mouse.get_pos()):
            if not self.preview_rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll = max(0, self.scroll - event.y * 28)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.generate()
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
                self._update_estimate()
            elif event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.unicode and event.unicode.isprintable() and len(self.input_text) < 120:
                self.input_text += event.unicode
                self._update_estimate()

    def generate(self) -> None:
        prompt = self.input_text.strip()
        if not prompt:
            self.chat.append(ChatLine("sys", "Typ eerst een prompt."))
            return

        cost, shape, diff = self.generator.estimate_cost(prompt)
        self.last_cost, self.last_shape, self.last_diff = cost, shape, diff

        if not self.wallet.can_afford(cost):
            self.chat.append(
                ChatLine(
                    "sys",
                    f"Niet genoeg tokens ({cost} nodig, {self.wallet.remaining} over).",
                )
            )
            self.status = "Tokens op"
            return

        self.chat.append(ChatLine("user", prompt))
        self.status = "Genereren…"
        self.draw()
        pygame.display.flip()

        plan = self.generator.parse(prompt)
        mesh = self.generator.generate(prompt)
        self.wallet.spend(cost)
        self.preview.set_mesh(mesh, photoreal=plan.photoreal or True)
        self.preview.auto_rotate = True
        self.gen_flash = 0.8
        bits = [f"acteurs={'+'.join(a.obj for a in plan.actors)}"]
        if plan.is_scene:
            bits.append(f"scene:{plan.environment or 'custom'}")
        if plan.parts:
            bits.append(f"delen={','.join(sorted(plan.parts))}")
        if plan.modifiers:
            bits.append(f"mods={','.join(sorted(plan.modifiers))}")
        n_col = sum(max(1, len(a.colors)) for a in plan.actors) + len(plan.colors)
        bits.append(f"multi-color×{max(n_col, 4)}")
        self.chat.append(
            ChatLine(
                "ai",
                f"Klaar: {mesh.name} · {diff} · −{cost} tok · "
                f"{mesh.vertex_count:,} verts · ".replace(",", ".")
                + " · ".join(bits),
            )
        )
        self.status = f"Model: {mesh.name}"
        self.input_text = ""
        self._update_estimate()
        self.scroll = 10_000  # spring naar onderkant

    def export(self, fmt: str) -> None:
        mesh = self.preview.full_mesh or self.preview.mesh
        if mesh is None:
            self.export_msg = "Geen model"
            self.export_t = 2.0
            return
        out = output_dir()
        path = out / f"{mesh.name}.{fmt}"
        if fmt == "obj":
            export_obj(mesh, path)
        else:
            export_stl(mesh, path)
        self.export_msg = f"Opgeslagen: {path.name}"
        self.export_t = 2.5
        self.chat.append(ChatLine("sys", f"Export → {path}"))

    def render_photoreal(self) -> None:
        """Ultra pipeline: JS · Java · Python · Java · C++ · Rust · C."""
        mesh = self.preview.full_mesh or self.preview.mesh
        if mesh is None:
            self.export_msg = "Geen model"
            self.export_t = 2.0
            self.chat.append(ChatLine("sys", "Genereer eerst een model voor fotoreal."))
            return
        self.status = "ULTRA photoreal…"
        self.chat.append(
            ChatLine(
                "sys",
                "Ultra fotoreal: C meshpack → C++ GGX → Rust denoise → Java×2 → JS viewer…",
            )
        )
        self.draw()
        pygame.display.flip()
        try:
            from neotorch.ultraphoto import render_ultra

            verts = mesh.vertex_count
            if verts > 40_000:
                w, h, spp = 640, 480, 48
            elif verts > 15_000:
                w, h, spp = 720, 540, 64
            else:
                w, h, spp = 800, 600, 96
            img, _obj = render_ultra(mesh, width=w, height=h, spp=spp, open_viewers=True)
            self.export_msg = f"ULTRA: {img.name}"
            self.export_t = 5.0
            self.status = "Ultra fotoreal klaar"
            self.chat.append(ChatLine("ai", f"ECHTE fotoreal klaar → {img}"))
        except Exception as e:
            self.status = "Fotoreal fout"
            self.export_msg = "Fotoreal mislukt"
            self.export_t = 3.0
            self.chat.append(ChatLine("sys", f"Fotoreal fout: {e}"))

    # ——— drawing ———

    def draw(self) -> None:
        self.screen.fill(BG)
        self._draw_header_glow()
        self._draw_token_panel()
        self._draw_ai_panel()
        self._draw_preview_panel()

    def _draw_header_glow(self) -> None:
        # Subtiele amber glow bovenaan
        glow = pygame.Surface((self.screen.get_width(), 120), pygame.SRCALPHA)
        for i in range(120):
            a = int(28 * (1 - i / 120))
            pygame.draw.line(glow, (*ACCENT_DIM, a), (0, i), (self.screen.get_width(), i))
        self.screen.blit(glow, (0, 0))
        brand = self.fonts.title.render("NeoTorch Studio", True, TEXT)
        self.screen.blit(brand, (MARGIN, 10))
        sub = self.fonts.tiny.render("lokaal · geen API-keys", True, TEXT_MUTED)
        self.screen.blit(sub, (MARGIN + brand.get_width() + 12, 20))

    def _draw_token_panel(self) -> None:
        r = self.token_rect
        draw_round_rect(self.screen, PANEL, r, 16)
        draw_round_rect(self.screen, STROKE_SOFT, r, 16, 1)

        # Title
        title = self.fonts.head.render("TOKENS", True, ACCENT)
        self.screen.blit(title, (r.left + 16, r.top + 16))
        day = self.fonts.tiny.render("50 / dag", True, TEXT_MUTED)
        self.screen.blit(day, (r.left + 16, r.top + 38))

        rem = self.wallet.remaining
        lim = self.wallet.limit
        used = self.wallet.used_today
        frac = rem / lim if lim else 0

        # Big number
        pulse = 0.5 + 0.5 * math.sin(self.anim * 2.2)
        num_col = lerp_color(OK, ACCENT, 1 - frac) if frac > 0.2 else DANGER
        if self.gen_flash > 0:
            num_col = lerp_color(num_col, TEXT, self.gen_flash)
        num = self.fonts.big_num.render(str(rem), True, num_col)
        self.screen.blit(num, (r.left + 16, r.top + 64))
        of = self.fonts.small.render(f"/ {lim}", True, TEXT_DIM)
        self.screen.blit(of, (r.left + 16 + num.get_width() + 6, r.top + 82))

        # Vertical battery bar
        bar = pygame.Rect(r.left + 16, r.top + 130, r.width - 32, 14)
        draw_round_rect(self.screen, (18, 20, 26), bar, 7)
        fill_w = int((bar.width - 4) * frac)
        if fill_w > 0:
            fill = pygame.Rect(bar.left + 2, bar.top + 2, fill_w, bar.height - 4)
            col = lerp_color(DANGER, OK, frac)
            # shimmer
            shimmer = lerp_color(col, TEXT, 0.15 * pulse)
            draw_round_rect(self.screen, shimmer, fill, 5)

        used_lbl = self.fonts.tiny.render(f"Gebruikt vandaag: {used}", True, TEXT_MUTED)
        self.screen.blit(used_lbl, (r.left + 16, r.top + 154))

        # Cost estimate card
        card = pygame.Rect(r.left + 12, r.top + 190, r.width - 24, 148)
        draw_round_rect(self.screen, PANEL_2, card, 12)
        draw_round_rect(self.screen, STROKE, card, 12, 1)
        self.screen.blit(self.fonts.small.render("Schatting", True, TEXT_DIM), (card.left + 12, card.top + 10))
        self.screen.blit(
            self.fonts.head.render(f"−{self.last_cost} tok", True, ACCENT),
            (card.left + 12, card.top + 32),
        )
        shape_lbl = self.last_shape[:22]
        self.screen.blit(
            self.fonts.tiny.render(f"Vorm: {shape_lbl}", True, TEXT_MUTED),
            (card.left + 12, card.top + 56),
        )
        diff_col = ACCENT if self.last_cost >= 12 else (OK if self.last_cost < 6 else TEXT_DIM)
        self.screen.blit(
            self.fonts.tiny.render(f"Niveau: {self.last_diff}", True, diff_col),
            (card.left + 12, card.top + 72),
        )
        verts_lbl = f"~{self.last_verts_est:,} verts".replace(",", ".")
        self.screen.blit(
            self.fonts.tiny.render(verts_lbl, True, TEXT_DIM),
            (card.left + 12, card.top + 90),
        )
        # Kleur-swatches
        sx = card.left + 12
        sy = card.top + 112
        if self.last_colors:
            for col in self.last_colors[:5]:
                rgb = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
                pygame.draw.rect(self.screen, rgb, pygame.Rect(sx, sy, 16, 16), border_radius=4)
                pygame.draw.rect(self.screen, STROKE, pygame.Rect(sx, sy, 16, 16), 1, border_radius=4)
                sx += 20
        else:
            self.screen.blit(self.fonts.tiny.render("kleur: auto", True, TEXT_MUTED), (sx, sy))

        # Cost legend — schaalt met promptcomplexiteit
        y = card.bottom + 18
        self.screen.blit(self.fonts.small.render("Kosten", True, TEXT_DIM), (r.left + 16, y))
        y += 22
        legend = [
            ("simpel", 3, OK),
            ("+ kleur / mods", 8, TEXT_DIM),
            ("detailed / complex", 14, ACCENT),
            ("ultra high-poly", 22, DANGER),
            (f"cap {PRACTICAL_MAX_VERTS // 1000}k verts", 0, WARN),
        ]
        for label, cost, dot in legend:
            if y > r.bottom - 28:
                break
            pygame.draw.circle(self.screen, dot, (r.left + 22, y + 6), 4)
            self.screen.blit(self.fonts.tiny.render(label, True, TEXT_MUTED), (r.left + 34, y))
            cost_s = self.fonts.tiny.render(f"~{cost}", True, TEXT_DIM)
            self.screen.blit(cost_s, (r.right - 28 - cost_s.get_width(), y))
            y += 18

    def _draw_ai_panel(self) -> None:
        r = self.ai_rect
        draw_round_rect(self.screen, PANEL, r, 16)
        draw_round_rect(self.screen, STROKE_SOFT, r, 16, 1)

        self.screen.blit(self.fonts.head.render("AI BAR", True, ACCENT), (r.left + 16, r.top + 16))
        self.screen.blit(
            self.fonts.tiny.render(self.status, True, TEXT_MUTED),
            (r.left + 16, r.top + 38),
        )

        # Chat area
        chat_area = pygame.Rect(r.left + 12, r.top + 58, r.width - 24, self.input_rect.top - r.top - 70)
        draw_round_rect(self.screen, (16, 18, 24), chat_area, 12)

        old_clip = self.screen.get_clip()
        self.screen.set_clip(chat_area.inflate(-4, -4))

        y = chat_area.top + 10 - self.scroll
        bubble_w = chat_area.width - 24
        for line in self.chat:
            color = TEXT if line.role == "ai" else (ACCENT if line.role == "user" else TEXT_MUTED)
            prefix = {"user": "Jij", "ai": "Neo", "sys": "·"}[line.role]
            header = self.fonts.tiny.render(prefix, True, TEXT_MUTED)
            wrapped = self._wrap(line.text, bubble_w - 16, self.fonts.body)
            bh = 22 + len(wrapped) * 18
            bubble = pygame.Rect(chat_area.left + 10, y, bubble_w, bh)
            bg = PANEL_2 if line.role != "user" else (36, 30, 22)
            if bubble.bottom > chat_area.top and bubble.top < chat_area.bottom:
                draw_round_rect(self.screen, bg, bubble, 10)
                self.screen.blit(header, (bubble.left + 10, bubble.top + 4))
                ty = bubble.top + 20
                for wline in wrapped:
                    self.screen.blit(self.fonts.body.render(wline, True, color), (bubble.left + 10, ty))
                    ty += 18
            y += bh + 8

        # Auto-clamp scroll
        content_h = y - (chat_area.top + 10 - self.scroll) + self.scroll
        max_scroll = max(0, content_h - chat_area.height + 20)
        self.scroll = int(clamp(self.scroll, 0, max_scroll))

        self.screen.set_clip(old_clip)

        # Input
        draw_round_rect(self.screen, PANEL_2, self.input_rect, 10)
        border = ACCENT if self.caret_on else STROKE
        draw_round_rect(self.screen, border, self.input_rect, 10, 1)
        placeholder = "Beschrijf je 3D-model…"
        shown = self.input_text if self.input_text else placeholder
        col = TEXT if self.input_text else TEXT_MUTED
        txt = self.fonts.body.render(shown[:54], True, col)
        self.screen.blit(txt, (self.input_rect.left + 12, self.input_rect.centery - txt.get_height() // 2))
        if self.caret_on and self.input_text:
            cx = self.input_rect.left + 12 + txt.get_width() + 1
            pygame.draw.line(
                self.screen,
                ACCENT,
                (cx, self.input_rect.top + 10),
                (cx, self.input_rect.bottom - 10),
                2,
            )

        # Generate button
        can = self.wallet.can_afford(self.last_cost) and bool(self.input_text.strip())
        btn_col = ACCENT if can else (70, 62, 48)
        draw_round_rect(self.screen, btn_col, self.gen_btn, 10)
        label = f"Genereer  (−{self.last_cost})"
        bt = self.fonts.head.render(label, True, (20, 16, 10) if can else TEXT_MUTED)
        self.screen.blit(bt, bt.get_rect(center=self.gen_btn.center))

    def _draw_preview_panel(self) -> None:
        r = self.preview_rect
        draw_round_rect(self.screen, PANEL, r, 16)
        draw_round_rect(self.screen, STROKE_SOFT, r, 16, 1)

        inner = r.inflate(-10, -10)
        inner.top += 44
        size = (inner.width, inner.height)
        if self._preview_buf is None or self._preview_buf_size != size:
            self._preview_buf = pygame.Surface(size)
            self._preview_buf_size = size
        self.preview.draw(self._preview_buf, self._preview_buf.get_rect())
        self.screen.blit(self._preview_buf, inner.topleft)
        draw_round_rect(self.screen, STROKE, inner, 12, 1)

        self.screen.blit(self.fonts.head.render("MODEL PREVIEW", True, ACCENT), (r.left + 16, r.top + 14))

        for btn, label, fill in (
            (self.export_obj_btn, "OBJ", PANEL_2),
            (self.export_stl_btn, "STL", PANEL_2),
            (self.photo_btn, "ULTRA", ACCENT_DIM),
        ):
            draw_round_rect(self.screen, fill, btn, 8)
            draw_round_rect(self.screen, STROKE, btn, 8, 1)
            col = TEXT if btn is self.photo_btn else TEXT_DIM
            t = self.fonts.tiny.render(label, True, col)
            self.screen.blit(t, t.get_rect(center=btn.center))

        show = self.preview.full_mesh or self.preview.mesh
        if show is not None:
            info = f"{show.name}  ·  {show.vertex_count} verts  ·  {show.face_count} faces"
            t = self.fonts.tiny.render(info, True, TEXT_MUTED)
            self.screen.blit(t, (r.left + 16, r.bottom - 28))

        if self.export_t > 0 and self.export_msg:
            msg = self.fonts.small.render(self.export_msg, True, OK)
            self.screen.blit(msg, (r.left + 16, r.top + 48))

        # Flash rim after generate
        if self.gen_flash > 0:
            a = int(120 * self.gen_flash)
            flash = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            pygame.draw.rect(flash, (*ACCENT, a), flash.get_rect(), width=3, border_radius=16)
            self.screen.blit(flash, r.topleft)

    def _wrap(self, text: str, max_w: int, font: pygame.font.Font) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        lines: List[str] = []
        cur = words[0]
        for w in words[1:]:
            test = f"{cur} {w}"
            if font.size(test)[0] <= max_w:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines


def main() -> None:
    NeoTorchStudio().run()


if __name__ == "__main__":
    main()

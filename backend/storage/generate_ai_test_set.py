import os
import math
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Curated harmonious color palette
BG_COLOR = (248, 250, 252)        # Slate 50
TEXT_COLOR = (15, 23, 42)         # Slate 900
BORDER_COLOR = (203, 213, 225)     # Slate 300
ARROW_COLOR = (71, 85, 105)        # Slate 600
COLOR_PRIMARY = (186, 230, 253)    # Sky 200
COLOR_SECONDARY = (187, 247, 208)  # Emerald 200
COLOR_ACCENT = (254, 243, 199)     # Amber 200
COLOR_PURPLE = (233, 213, 255)     # Purple 200
COLOR_ROSE = (254, 205, 211)       # Rose 200

def get_font(size=14, bold=False):
    font_names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "Calibrib.ttf" if bold else "Calibri.ttf",
        "arial.ttf"
    ]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            try:
                path = os.path.join("C:\\Windows\\Fonts", name)
                if os.path.exists(path):
                    return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw_arrow(draw, start, end, fill=ARROW_COLOR, width=2, arrow_len=10):
    x1, y1 = start
    x2, y2 = end
    draw.line([start, end], fill=fill, width=width)
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    if length > 0:
        dx /= length
        dy /= length
        px = -dy
        py = dx
        tip1 = (x2 - arrow_len * dx + arrow_len * 0.5 * px, y2 - arrow_len * dy + arrow_len * 0.5 * py)
        tip2 = (x2 - arrow_len * dx - arrow_len * 0.5 * px, y2 - arrow_len * dy - arrow_len * 0.5 * py)
        draw.polygon([end, tip1, tip2], fill=fill)

def draw_label(draw, text, x, y, bg_color=COLOR_PRIMARY, font=None):
    if font is None:
        font = get_font(12, bold=True)
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw = right - left
        th = bottom - top
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
    
    pad_x, pad_y = 8, 4
    box = [x - tw//2 - pad_x, y - th//2 - pad_y, x + tw//2 + pad_x, y + th//2 + pad_y]
    draw.rounded_rectangle(box, radius=4, fill=bg_color, outline=BORDER_COLOR, width=1)
    draw.text((x - tw//2, y - th//2), text, fill=TEXT_COLOR, font=font)
    return box

def create_gpt_diagram():
    # GPT-style biology mitochondrion diagram
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Mitochondrion Structure (GPT-Emulated Diagram)", fill=TEXT_COLOR, font=font_title)
    
    # Outer membrane (large red oval)
    draw.ellipse([250, 180, 550, 420], fill=COLOR_ROSE, outline=BORDER_COLOR, width=3)
    
    # Inner membrane folded cristae (nested orange folds)
    pts = [
        (280, 300), (320, 240), (340, 360), (380, 240), (400, 360),
        (440, 240), (460, 360), (500, 240), (520, 300)
    ]
    draw.line(pts, fill=COLOR_ACCENT, width=8, joint="round")
    
    # Text labels with arrows
    draw_label(draw, "Outer Membrane", 140, 160, COLOR_PRIMARY, font_labels)
    draw_arrow(draw, (200, 160), (280, 200))
    
    draw_label(draw, "Inner Membrane Fold", 140, 440, COLOR_PRIMARY, font_labels)
    draw_arrow(draw, (210, 440), (330, 340))
    
    draw_label(draw, "Matrix Fluid", 660, 300, COLOR_SECONDARY, font_labels)
    draw_arrow(draw, (610, 300), (450, 300))
    
    img.save(os.path.join(OUTPUT_DIR, "gpt_style_diagram.png"))
    print("   [OK] gpt_style_diagram.png created.")

def create_gemini_diagram():
    # Gemini-style Physics schematic (Refracting Prism)
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Prism Dispersion (Gemini-Emulated Schematic)", fill=TEXT_COLOR, font=font_title)
    
    # Glass prism (blue triangle)
    draw.polygon([(400, 150), (520, 380), (280, 380)], fill=COLOR_PRIMARY, outline=BORDER_COLOR, width=3)
    
    # Incident beam (white/yellow arrow)
    draw_arrow(draw, (150, 320), (340, 280), fill=(234, 179, 8), width=3)
    
    # Refracted rays (color spectrum)
    colors = [(239, 68, 68), (249, 115, 22), (234, 179, 8), (34, 197, 94), (59, 130, 246), (168, 85, 247)]
    for i, col in enumerate(colors):
        y_out = 250 + i * 15
        draw.line([(340, 280), (430, y_out)], fill=col, width=2)
        draw_arrow(draw, (430, y_out), (600, y_out - 40 + i*20), fill=col, width=2)
        
    draw_label(draw, "Light Dispersion", 680, 260, COLOR_ACCENT, font_labels)
    draw_label(draw, "Glass Prism", 400, 420, COLOR_SECONDARY, font_labels)
    
    img.save(os.path.join(OUTPUT_DIR, "gemini_style_schematic.png"))
    print("   [OK] gemini_style_schematic.png created.")

def create_midjourney_diagram():
    # Midjourney-style Geography illustration (Earth Layers)
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Layers of the Earth (Midjourney-Emulated Illustration)", fill=TEXT_COLOR, font=font_title)
    
    # Crust, Mantle, Core concentric circles
    draw.ellipse([250, 150, 550, 450], fill=COLOR_PRIMARY, outline=BORDER_COLOR, width=2)     # Crust
    draw.ellipse([280, 180, 520, 420], fill=COLOR_SECONDARY, outline=BORDER_COLOR, width=2)   # Mantle
    draw.ellipse([320, 220, 480, 380], fill=COLOR_ACCENT, outline=BORDER_COLOR, width=2)      # Outer Core
    draw.ellipse([360, 260, 440, 340], fill=COLOR_ROSE, outline=BORDER_COLOR, width=2)        # Inner Core
    
    # Labels
    draw_label(draw, "Crust", 140, 200, COLOR_PRIMARY, font_labels)
    draw_arrow(draw, (180, 200), (265, 200))
    
    draw_label(draw, "Mantle Layer", 140, 300, COLOR_SECONDARY, font_labels)
    draw_arrow(draw, (200, 300), (300, 300))
    
    draw_label(draw, "Outer Core", 660, 250, COLOR_ACCENT, font_labels)
    draw_arrow(draw, (610, 250), (450, 280))
    
    draw_label(draw, "Inner Core", 660, 350, COLOR_ROSE, font_labels)
    draw_arrow(draw, (610, 350), (400, 320))
    
    img.save(os.path.join(OUTPUT_DIR, "midjourney_style_illustration.png"))
    print("   [OK] midjourney_style_illustration.png created.")

def create_canva_diagram():
    # Canva-style multi-layer slide export
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Project Roadmap (Canva-Emulated Slide Layout)", fill=TEXT_COLOR, font=font_title)
    
    # 3 colored cards with overlapping icons
    cards = [
        ("Phase 1: Research", "Define target audience and explore tech stack.", COLOR_PRIMARY, 100),
        ("Phase 2: Prototype", "Implement amodal engines and build dashboards.", COLOR_SECONDARY, 320),
        ("Phase 3: Launch", "Deploy batch processors and run validations.", COLOR_PURPLE, 540)
    ]
    
    for title, desc, col, x in cards:
        # Background card
        draw.rounded_rectangle([x, 150, x + 180, 480], radius=10, fill=col, outline=BORDER_COLOR, width=2)
        
        # Header text
        draw.text((x + 15, 180), title.split(":")[0], fill=TEXT_COLOR, font=font_labels)
        draw.text((x + 15, 205), title.split(":")[1].strip(), fill=TEXT_COLOR, font=get_font(15, bold=True))
        
        # Separator line
        draw.line([(x + 15, 245), (x + 165, 245)], fill=BORDER_COLOR, width=1)
        
        # Description text word-wrapped
        words = desc.split()
        lines = []
        cur_line = []
        for w in words:
            if len(" ".join(cur_line + [w])) > 20:
                lines.append(" ".join(cur_line))
                cur_line = [w]
            else:
                cur_line.append(w)
        lines.append(" ".join(cur_line))
        
        y_text = 270
        for l in lines:
            draw.text((x + 15, y_text), l, fill=TEXT_COLOR, font=get_font(12))
            y_text += 20
            
    # Draw horizontal roadmap connector line across cards
    draw.line([(190, 400), (610, 400)], fill=ARROW_COLOR, width=4)
    draw_arrow(draw, (600, 400), (630, 400), fill=ARROW_COLOR, width=4, arrow_len=12)
    
    img.save(os.path.join(OUTPUT_DIR, "canva_style_slide.png"))
    print("   [OK] canva_style_slide.png created.")

def run():
    print("Generating AI validation diagrams...")
    create_gpt_diagram()
    create_gemini_diagram()
    create_midjourney_diagram()
    create_canva_diagram()
    print("AI validation set generated successfully.")

if __name__ == "__main__":
    run()

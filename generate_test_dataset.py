import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Define Output Directory
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Curated HSL-like Premium Palette (RGB values)
BG_COLOR = (250, 250, 252)        # Sleek off-white
TEXT_COLOR = (30, 41, 59)         # Dark slate
BORDER_COLOR = (148, 163, 184)     # Medium gray border
ARROW_COLOR = (100, 116, 139)      # Slate gray arrow
SHAPE_BLUE = (224, 242, 254)       # Light sky blue
SHAPE_GREEN = (220, 252, 231)      # Soft emerald green
SHAPE_RED = (254, 226, 226)        # Soft coral/red
SHAPE_ORANGE = (254, 243, 199)     # Soft amber/orange
SHAPE_PURPLE = (243, 232, 255)     # Soft lavender

# Fonts Helper
def get_font(size=14, bold=False):
    font_names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "Calibrib.ttf" if bold else "Calibri.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "arial.ttf",
        "tahoma.ttf"
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

# Drawing Helpers
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

def draw_label(draw, text, x, y, bg_color=SHAPE_BLUE, font=None):
    if font is None:
        font = get_font(12, bold=True)
    # Estimate text bounds
    try:
        # Pillow >= 10.0.0
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw = right - left
        th = bottom - top
    except AttributeError:
        # Pillow < 10.0.0
        tw, th = draw.textsize(text, font=font)
    
    pad_x, pad_y = 8, 4
    box = [x - tw//2 - pad_x, y - th//2 - pad_y, x + tw//2 + pad_x, y + th//2 + pad_y]
    draw.rounded_rectangle(box, radius=4, fill=bg_color, outline=BORDER_COLOR, width=1)
    
    text_y_offset = y - th//2
    # Adjust slightly for baseline
    draw.text((x - tw//2, text_y_offset), text, fill=TEXT_COLOR, font=font)
    return box

# ==========================================
# CATEGORY 1: BIOLOGY DIAGRAMS
# ==========================================

def create_bio_digestive():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Human Digestive System", fill=TEXT_COLOR, font=font_title)
    
    # Draw body outline / tract schematic (simplified)
    # Throat path
    draw.line([(400, 100), (400, 200)], fill=(226, 180, 180), width=16)
    # Stomach shape (soft red oval)
    draw.ellipse([360, 200, 460, 280], fill=(254, 200, 200), outline=(220, 100, 100), width=2)
    # Liver shape (green wedge)
    draw.polygon([(320, 210), (370, 210), (350, 250)], fill=(200, 240, 200), outline=(100, 180, 100), width=2)
    # Intestines (purple block)
    draw.rounded_rectangle([350, 290, 450, 390], radius=10, fill=(240, 220, 250), outline=(180, 120, 220), width=2)
    
    # Add label boxes & arrows
    labels_data = [
        ("Mouth", (400, 90), (180, 90), "right"),
        ("Esophagus", (400, 160), (180, 160), "right"),
        ("Liver", (340, 220), (180, 220), "right"),
        ("Stomach", (430, 240), (620, 240), "left"),
        ("Small Intestine", (400, 310), (620, 310), "left"),
        ("Large Intestine", (400, 360), (620, 360), "left")
    ]
    
    for label, target_pt, label_pt, direction in labels_data:
        draw_label(draw, label, label_pt[0], label_pt[1], SHAPE_BLUE, font_labels)
        if direction == "right":
            draw_arrow(draw, (label_pt[0] + 50, label_pt[1]), (target_pt[0] - 10, target_pt[1]))
        else:
            draw_arrow(draw, (label_pt[0] - 50, label_pt[1]), (target_pt[0] + 10, target_pt[1]))
            
    img.save(os.path.join(OUTPUT_DIR, "biology_digestive_system.png"))

def create_bio_heart():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Human Heart Structure", fill=TEXT_COLOR, font=font_title)
    
    # Draw Heart shape outline (a combination of arcs and polygons)
    draw.ellipse([320, 200, 480, 360], fill=SHAPE_RED, outline=BORDER_COLOR, width=2)
    draw.ellipse([300, 160, 390, 220], fill=SHAPE_BLUE, outline=BORDER_COLOR, width=2) # Vena Cava/Aorta tubes
    draw.ellipse([410, 150, 460, 220], fill=SHAPE_RED, outline=BORDER_COLOR, width=2)
    
    # Internal division indicator
    draw.line([(400, 200), (400, 350)], fill=BORDER_COLOR, width=2)
    
    # Add label boxes & arrows
    labels_data = [
        ("Superior Vena Cava", (340, 180), (160, 150), "right"),
        ("Right Atrium", (360, 240), (160, 240), "right"),
        ("Right Ventricle", (360, 310), (160, 330), "right"),
        ("Aorta", (435, 170), (640, 150), "left"),
        ("Left Atrium", (440, 240), (640, 240), "left"),
        ("Left Ventricle", (440, 310), (640, 330), "left")
    ]
    
    for label, target_pt, label_pt, direction in labels_data:
        draw_label(draw, label, label_pt[0], label_pt[1], SHAPE_PURPLE, font_labels)
        if direction == "right":
            draw_arrow(draw, (label_pt[0] + 60, label_pt[1]), (target_pt[0] - 5, target_pt[1]))
        else:
            draw_arrow(draw, (label_pt[0] - 60, label_pt[1]), (target_pt[0] + 5, target_pt[1]))
            
    img.save(os.path.join(OUTPUT_DIR, "biology_human_heart.png"))

def create_bio_plant_cell():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Plant Cell Structure Diagram", fill=TEXT_COLOR, font=font_title)
    
    # Draw Green Octagon Cell Wall
    pts = [(300, 150), (500, 150), (600, 250), (600, 450), (500, 550), (300, 550), (200, 450), (200, 250)]
    draw.polygon(pts, fill=(230, 250, 230), outline=(50, 150, 50), width=6) # wall
    draw.polygon(pts, fill=None, outline=(100, 200, 100), width=2) # membrane
    
    # Large Vacuole (blue oval)
    draw.ellipse([320, 220, 480, 420], fill=(220, 240, 255), outline=(100, 150, 250), width=2)
    # Nucleus (purple circle)
    draw.ellipse([450, 380, 530, 460], fill=SHAPE_PURPLE, outline=BORDER_COLOR, width=2)
    # Chloroplasts (small green ovals)
    draw.ellipse([240, 220, 280, 250], fill=(150, 220, 150), outline=(50, 120, 50), width=2)
    draw.ellipse([230, 360, 270, 390], fill=(150, 220, 150), outline=(50, 120, 50), width=2)
    draw.ellipse([520, 200, 560, 230], fill=(150, 220, 150), outline=(50, 120, 50), width=2)
    
    # Labels
    labels_data = [
        ("Cell Wall", (200, 300), (90, 200), "right"),
        ("Cell Membrane", (250, 280), (90, 280), "right"),
        ("Chloroplast", (250, 375), (90, 370), "right"),
        ("Large Vacuole", (360, 300), (360, 90), "down"),
        ("Nucleus", (490, 420), (700, 420), "left"),
        ("Cytoplasm", (320, 470), (700, 490), "left")
    ]
    
    for label, target_pt, label_pt, direction in labels_data:
        draw_label(draw, label, label_pt[0], label_pt[1], SHAPE_GREEN, font_labels)
        if direction == "right":
            draw_arrow(draw, (label_pt[0] + 50, label_pt[1]), (target_pt[0] - 5, target_pt[1]))
        elif direction == "left":
            draw_arrow(draw, (label_pt[0] - 50, label_pt[1]), (target_pt[0] + 5, target_pt[1]))
        else:
            draw_arrow(draw, (label_pt[0], label_pt[1] + 15), (target_pt[0], target_pt[1] - 5))
            
    img.save(os.path.join(OUTPUT_DIR, "biology_plant_cell.png"))

def create_bio_animal_cell():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Animal Cell Structure Diagram", fill=TEXT_COLOR, font=font_title)
    
    # Irregular animal cell border (circle-like)
    draw.ellipse([220, 150, 580, 510], fill=(255, 250, 240), outline=BORDER_COLOR, width=3)
    
    # Nucleus (large central purple circle)
    draw.ellipse([340, 260, 440, 360], fill=SHAPE_PURPLE, outline=(150, 100, 220), width=2)
    # Nucleolus (dark circle inside)
    draw.ellipse([370, 290, 410, 330], fill=(150, 100, 220), outline=None)
    # Mitochondria (orange ovals with inner lines)
    draw.ellipse([270, 210, 330, 250], fill=SHAPE_ORANGE, outline=(220, 120, 20), width=2)
    draw.ellipse([480, 390, 540, 430], fill=SHAPE_ORANGE, outline=(220, 120, 20), width=2)
    
    # Labels
    labels_data = [
        ("Cell Membrane", (230, 250), (90, 200), "right"),
        ("Mitochondrion", (280, 235), (90, 280), "right"),
        ("Lysosome", (300, 400), (90, 420), "right"),
        ("Nucleolus", (390, 310), (390, 90), "down"),
        ("Nucleus", (430, 320), (700, 280), "left"),
        ("Cytoplasm", (500, 250), (700, 360), "left"),
        ("Ribosome", (470, 210), (700, 440), "left")
    ]
    
    for label, target_pt, label_pt, direction in labels_data:
        draw_label(draw, label, label_pt[0], label_pt[1], SHAPE_BLUE, font_labels)
        if direction == "right":
            draw_arrow(draw, (label_pt[0] + 55, label_pt[1]), (target_pt[0] - 5, target_pt[1]))
        elif direction == "left":
            draw_arrow(draw, (label_pt[0] - 55, label_pt[1]), (target_pt[0] + 5, target_pt[1]))
        else:
            draw_arrow(draw, (label_pt[0], label_pt[1] + 15), (target_pt[0], target_pt[1] - 5))
            
    img.save(os.path.join(OUTPUT_DIR, "biology_animal_cell.png"))

# ==========================================
# CATEGORY 2: PHYSICS / EARTH SCIENCE
# ==========================================

def create_phys_solar():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "The Solar System", fill=TEXT_COLOR, font=font_title)
    
    # Sun (large curve on left)
    draw.ellipse([-300, 100, 100, 500], fill=(255, 230, 150), outline=(255, 180, 50), width=4)
    draw.text((40, 290), "Sun", fill=TEXT_COLOR, font=get_font(14, bold=True))
    
    # Orbit lines
    for r in [180, 260, 340, 420, 500, 600]:
        draw.arc([-300, 300 - r, -300 + 2*r, 300 + r], start=270, end=90, fill=(200, 200, 200), width=1)
        
    # Planets (spheres along orbits)
    planets = [
        ("Mercury", (180, 300), 10, SHAPE_ORANGE),
        ("Venus", (260, 250), 16, SHAPE_RED),
        ("Earth", (340, 320), 18, SHAPE_BLUE),
        ("Mars", (420, 270), 12, SHAPE_RED),
        ("Jupiter", (500, 330), 34, SHAPE_ORANGE),
        ("Saturn", (600, 260), 28, SHAPE_PURPLE)
    ]
    
    for name, pos, r, color in planets:
        x, y = pos
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=BORDER_COLOR, width=2)
        if name == "Saturn":
            draw.ellipse([x - r - 10, y - 5, x + r + 10, y + 5], fill=None, outline=(180, 180, 180), width=2)
        
        draw_label(draw, name, x, y + r + 20, SHAPE_BLUE, font_labels)
        
    img.save(os.path.join(OUTPUT_DIR, "physics_solar_system.png"))

def create_phys_water_cycle():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    
    draw.text((40, 30), "The Water Cycle", fill=TEXT_COLOR, font=font_title)
    
    # Ocean
    draw.rectangle([500, 450, 760, 550], fill=(220, 240, 255), outline=(100, 150, 220), width=2)
    # Land
    draw.polygon([(40, 550), (250, 350), (380, 460), (500, 450), (500, 550)], fill=(220, 250, 220), outline=(100, 180, 100), width=2)
    
    # Sun
    draw.ellipse([80, 80, 140, 140], fill=(255, 240, 150), outline=(255, 180, 50), width=2)
    # Clouds
    draw.ellipse([520, 100, 600, 150], fill=(240, 240, 245), outline=BORDER_COLOR)
    draw.ellipse([560, 90, 640, 140], fill=(240, 240, 245), outline=BORDER_COLOR)
    draw.ellipse([540, 110, 620, 160], fill=(240, 240, 245), outline=None)
    
    # Cycle Arrows
    draw_arrow(draw, (620, 430), (620, 200), width=3)
    draw_label(draw, "Evaporation", 620, 300, SHAPE_BLUE)
    
    draw_label(draw, "Condensation", 540, 65, SHAPE_PURPLE)
    
    draw_arrow(draw, (500, 160), (320, 350), width=3)
    for i in range(5):
        draw.line([(450 - i*20, 220 + i*25), (445 - i*20, 235 + i*25)], fill=(150, 150, 250), width=2)
    draw_label(draw, "Precipitation", 410, 240, SHAPE_RED)
    
    draw_arrow(draw, (260, 450), (480, 500), width=3)
    draw_label(draw, "Surface Runoff", 340, 520, SHAPE_GREEN)
    
    draw_arrow(draw, (180, 380), (120, 200), width=3)
    draw_label(draw, "Transpiration", 120, 280, SHAPE_ORANGE)
    
    img.save(os.path.join(OUTPUT_DIR, "physics_water_cycle.png"))

def create_phys_circuit():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Electric Circuit Diagram", fill=TEXT_COLOR, font=font_title)
    
    # Wire rectangle
    draw.rectangle([200, 150, 600, 450], fill=None, outline=BORDER_COLOR, width=4)
    
    # Battery Symbol
    draw.rectangle([380, 140, 420, 160], fill=BG_COLOR, outline=None)
    draw.line([(390, 130), (390, 170)], fill=TEXT_COLOR, width=4)
    draw.line([(410, 140), (410, 160)], fill=TEXT_COLOR, width=8)
    draw_label(draw, "Battery (Power Source)", 400, 100, SHAPE_ORANGE)
    
    # Bulb Symbol
    draw.rectangle([590, 270, 610, 330], fill=BG_COLOR, outline=None)
    draw.ellipse([575, 275, 625, 325], fill=SHAPE_BLUE, outline=BORDER_COLOR, width=3)
    draw.line([(585, 285), (615, 315)], fill=TEXT_COLOR, width=3)
    draw.line([(585, 315), (615, 285)], fill=TEXT_COLOR, width=3)
    draw_label(draw, "Light Bulb (Load)", 690, 300, SHAPE_BLUE)
    
    # Switch Symbol
    draw.rectangle([370, 440, 430, 460], fill=BG_COLOR, outline=None)
    draw.circle((380, 450), 4, fill=TEXT_COLOR)
    draw.circle((420, 450), 4, fill=TEXT_COLOR)
    draw.line([(380, 450), (415, 425)], fill=TEXT_COLOR, width=3)
    draw_label(draw, "Switch (Open State)", 400, 500, SHAPE_RED)
    
    draw_arrow(draw, (200, 300), (200, 280), width=3)
    draw_label(draw, "Current Flow", 110, 290, SHAPE_GREEN)
    
    img.save(os.path.join(OUTPUT_DIR, "physics_electric_circuit.png"))

def create_phys_reflection():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    font_labels = get_font(12, bold=True)
    
    draw.text((40, 30), "Reflection of Light", fill=TEXT_COLOR, font=font_title)
    
    mirror_y = 400
    draw.line([(100, mirror_y), (700, mirror_y)], fill=TEXT_COLOR, width=4)
    for x in range(120, 700, 20):
        draw.line([(x, mirror_y), (x - 10, mirror_y + 15)], fill=BORDER_COLOR, width=2)
    draw_label(draw, "Flat Mirror Surface", 400, mirror_y + 40, SHAPE_BLUE)
    
    for y in range(150, mirror_y, 15):
        draw.line([(400, y), (400, y + 8)], fill=ARROW_COLOR, width=2)
    draw_label(draw, "Normal", 400, 120, SHAPE_PURPLE)
    
    draw_arrow(draw, (150, 180), (400, mirror_y), width=3, arrow_len=15)
    draw_label(draw, "Incident Ray", 190, 150, SHAPE_RED)
    
    draw_arrow(draw, (400, mirror_y), (650, 180), width=3, arrow_len=15)
    draw_label(draw, "Reflected Ray", 610, 150, SHAPE_GREEN)
    
    draw.arc([360, 360, 440, 440], start=180, end=270, fill=TEXT_COLOR, width=2)
    draw.arc([360, 360, 440, 440], start=270, end=360, fill=TEXT_COLOR, width=2)
    
    draw.text((345, 335), "i", fill=TEXT_COLOR, font=get_font(14, bold=True))
    draw.text((445, 335), "r", fill=TEXT_COLOR, font=get_font(14, bold=True))
    
    draw_label(draw, "Angle of Incidence (i)", 270, 290, SHAPE_ORANGE)
    draw_label(draw, "Angle of Reflection (r)", 530, 290, SHAPE_ORANGE)
    
    img.save(os.path.join(OUTPUT_DIR, "physics_reflection_of_light.png"))

# ==========================================
# CATEGORY 3: FLOWCHARTS
# ==========================================

def create_flow_admission():
    img = Image.new("RGB", (850, 650), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    
    draw.text((40, 30), "Student Admission Process Flow", fill=TEXT_COLOR, font=font_title)
    
    draw.rounded_rectangle([350, 90, 470, 140], radius=20, fill=SHAPE_GREEN, outline=BORDER_COLOR, width=2)
    draw.text((385, 105), "START", fill=TEXT_COLOR, font=get_font(13, bold=True))
    
    draw.rectangle([330, 190, 490, 240], fill=SHAPE_BLUE, outline=BORDER_COLOR, width=2)
    draw.text((345, 205), "Submit Application", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    diamond_pts = [(410, 290), (490, 340), (410, 390), (330, 340)]
    draw.polygon(diamond_pts, fill=SHAPE_ORANGE, outline=BORDER_COLOR, width=2)
    draw.text((350, 330), "Eligible?", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.rectangle([560, 315, 710, 365], fill=SHAPE_BLUE, outline=BORDER_COLOR, width=2)
    draw.text((595, 330), "Pay Fees", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.rectangle([90, 315, 240, 365], fill=SHAPE_RED, outline=BORDER_COLOR, width=2)
    draw.text((115, 330), "Reject Student", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.rounded_rectangle([350, 460, 470, 510], radius=20, fill=SHAPE_PURPLE, outline=BORDER_COLOR, width=2)
    draw.text((395, 475), "END", fill=TEXT_COLOR, font=get_font(13, bold=True))
    
    draw_arrow(draw, (410, 140), (410, 190))
    draw_arrow(draw, (410, 240), (410, 290))
    
    draw_arrow(draw, (490, 340), (560, 340))
    draw.text((515, 320), "YES", fill=TEXT_COLOR, font=get_font(11, bold=True))
    
    draw_arrow(draw, (330, 340), (240, 340))
    draw.text((275, 320), "NO", fill=TEXT_COLOR, font=get_font(11, bold=True))
    
    draw_arrow(draw, (635, 365), (470, 485))
    draw_arrow(draw, (165, 365), (350, 485))
    
    img.save(os.path.join(OUTPUT_DIR, "flowchart_student_admission_process.png"))

def create_flow_sdlc():
    img = Image.new("RGB", (900, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    
    draw.text((40, 30), "Software Development Lifecycle (SDLC)", fill=TEXT_COLOR, font=font_title)
    
    stages = [
        ("Requirements", 50, SHAPE_RED),
        ("Design", 190, SHAPE_ORANGE),
        ("Development", 330, SHAPE_BLUE),
        ("Testing", 470, SHAPE_GREEN),
        ("Deployment", 610, SHAPE_PURPLE),
        ("Maintenance", 750, SHAPE_BLUE)
    ]
    
    y = 250
    w = 110
    h = 80
    
    for idx, (name, x, color) in enumerate(stages):
        draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=color, outline=BORDER_COLOR, width=2)
        draw.text((x + 10, y + 30), name, fill=TEXT_COLOR, font=get_font(11, bold=True))
        
        if idx < len(stages) - 1:
            draw_arrow(draw, (x + w, y + h//2), (stages[idx+1][1], y + h//2), arrow_len=8)
            
    draw.line([(805, y + h), (805, y + h + 80), (105, y + h + 80), (105, y + h)], fill=ARROW_COLOR, width=2)
    draw_arrow(draw, (105, y + h + 20), (105, y + h + 5))
    draw.text((420, y + h + 90), "Feedback & Iteration Loop", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    img.save(os.path.join(OUTPUT_DIR, "flowchart_software_development_lifecycle.png"))

def create_flow_order():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    
    draw.text((40, 30), "Order Processing Flowchart", fill=TEXT_COLOR, font=font_title)
    
    draw.rounded_rectangle([330, 80, 470, 130], radius=15, fill=SHAPE_GREEN, outline=BORDER_COLOR, width=2)
    draw.text((370, 95), "Receive Order", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    diamond = [(400, 180), (480, 230), (400, 280), (320, 230)]
    draw.polygon(diamond, fill=SHAPE_ORANGE, outline=BORDER_COLOR, width=2)
    draw.text((345, 220), "In Stock?", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.rectangle([330, 330, 470, 380], fill=SHAPE_BLUE, outline=BORDER_COLOR, width=2)
    draw.text((360, 345), "Ship Order", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.rectangle([560, 205, 700, 255], fill=SHAPE_RED, outline=BORDER_COLOR, width=2)
    draw.text((585, 220), "Reorder Item", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.rounded_rectangle([330, 460, 470, 510], radius=15, fill=SHAPE_PURPLE, outline=BORDER_COLOR, width=2)
    draw.text((375, 475), "Complete", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw_arrow(draw, (400, 130), (400, 180))
    draw_arrow(draw, (400, 280), (400, 330))
    draw.text((410, 295), "YES", fill=TEXT_COLOR, font=get_font(11, bold=True))
    
    draw_arrow(draw, (480, 230), (560, 230))
    draw.text((510, 210), "NO", fill=TEXT_COLOR, font=get_font(11, bold=True))
    
    draw_arrow(draw, (400, 380), (400, 460))
    draw_arrow(draw, (630, 255), (470, 485))
    
    img.save(os.path.join(OUTPUT_DIR, "flowchart_order_processing.png"))

def create_flow_temperature():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font_title = get_font(20, bold=True)
    
    draw.text((40, 30), "Temperature Control Loop", fill=TEXT_COLOR, font=font_title)
    
    draw.rectangle([330, 80, 470, 130], fill=SHAPE_BLUE, outline=BORDER_COLOR, width=2)
    draw.text((350, 95), "Read Temperature", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    diamond = [(400, 180), (490, 230), (400, 280), (310, 230)]
    draw.polygon(diamond, fill=SHAPE_ORANGE, outline=BORDER_COLOR, width=2)
    draw.text((350, 220), "Temp > 25 C?", fill=TEXT_COLOR, font=get_font(11, bold=True))
    
    draw.rectangle([150, 320, 290, 370], fill=SHAPE_RED, outline=BORDER_COLOR, width=2)
    draw.text((170, 335), "Turn On Cooling", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.rectangle([510, 320, 650, 370], fill=SHAPE_GREEN, outline=BORDER_COLOR, width=2)
    draw.text((530, 335), "Turn Off Cooling", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.rectangle([330, 440, 470, 490], fill=SHAPE_PURPLE, outline=BORDER_COLOR, width=2)
    draw.text((355, 455), "Wait 10 Seconds", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw_arrow(draw, (400, 130), (400, 180))
    draw_arrow(draw, (310, 230), (220, 320))
    draw.text((240, 240), "YES", fill=TEXT_COLOR, font=get_font(11, bold=True))
    
    draw_arrow(draw, (490, 230), (580, 320))
    draw.text((540, 240), "NO", fill=TEXT_COLOR, font=get_font(11, bold=True))
    
    draw_arrow(draw, (220, 370), (330, 465))
    draw_arrow(draw, (580, 370), (470, 465))
    
    draw.line([(330, 465), (100, 465), (100, 105), (330, 105)], fill=ARROW_COLOR, width=2)
    draw_arrow(draw, (320, 105), (328, 105))
    
    img.save(os.path.join(OUTPUT_DIR, "flowchart_temperature_control_loop.png"))

# ==========================================
# CATEGORY 4: INFOGRAPHICS
# ==========================================

def create_info_parts_of_speech():
    img = Image.new("RGB", (900, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    draw.text((40, 30), "Core Parts of Speech", fill=TEXT_COLOR, font=get_font(24, bold=True))
    
    cards = [
        ("NOUN", "Names a person, place, thing, or idea.", "Examples: dog, city, freedom", SHAPE_BLUE, 50),
        ("VERB", "Expresses action or state of being.", "Examples: run, exists, study", SHAPE_GREEN, 260),
        ("ADJECTIVE", "Modifies or describes a noun.", "Examples: beautiful, fast, smart", SHAPE_RED, 470),
        ("ADVERB", "Modifies a verb, adjective, or adverb.", "Examples: quickly, very, yesterday", SHAPE_ORANGE, 680)
    ]
    
    for title, desc, examples, color, x in cards:
        draw.rounded_rectangle([x, 120, x + 180, 480], radius=12, fill=color, outline=BORDER_COLOR, width=2)
        draw.text((x + 20, 150), title, fill=TEXT_COLOR, font=get_font(16, bold=True))
        draw.line([(x + 20, 185), (x + 160, 185)], fill=BORDER_COLOR, width=1)
        
        words = desc.split()
        lines = []
        cur_line = []
        for w in words:
            if len(" ".join(cur_line + [w])) > 22:
                lines.append(" ".join(cur_line))
                cur_line = [w]
            else:
                cur_line.append(w)
        lines.append(" ".join(cur_line))
        
        y_offset = 210
        for line in lines:
            draw.text((x + 20, y_offset), line, fill=TEXT_COLOR, font=get_font(12))
            y_offset += 20
            
        draw.text((x + 20, 350), examples, fill=TEXT_COLOR, font=get_font(11, bold=True))
        
    img.save(os.path.join(OUTPUT_DIR, "infographic_parts_of_speech.png"))

def create_info_learning_process():
    img = Image.new("RGB", (900, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    draw.text((40, 30), "The 4-Step Learning Process", fill=TEXT_COLOR, font=get_font(24, bold=True))
    
    steps = [
        ("1. Read", "Consume and absorb educational textbooks, documents, or video material.", SHAPE_RED, 60),
        ("2. Process", "Analyze structures, summarize concepts, and extract key core insights.", SHAPE_ORANGE, 260),
        ("3. Practice", "Build sample projects, solve quiz assignments, and run dry tests.", SHAPE_BLUE, 460),
        ("4. Teach", "Explain concepts to others to solidify knowledge and uncover gaps.", SHAPE_GREEN, 660)
    ]
    
    for title, desc, color, x in steps:
        draw.rounded_rectangle([x, 140, x + 170, 200], radius=8, fill=color, outline=BORDER_COLOR, width=2)
        draw.text((x + 20, 160), title, fill=TEXT_COLOR, font=get_font(15, bold=True))
        
        if x < 600:
            draw_arrow(draw, (x + 175, 170), (x + 255, 170), width=3)
            
        draw.rounded_rectangle([x - 5, 230, x + 175, 450], radius=6, fill=BG_COLOR, outline=BORDER_COLOR, width=1)
        
        words = desc.split()
        lines = []
        cur_line = []
        for w in words:
            if len(" ".join(cur_line + [w])) > 22:
                lines.append(" ".join(cur_line))
                cur_line = [w]
            else:
                cur_line.append(w)
        lines.append(" ".join(cur_line))
        
        y_offset = 260
        for line in lines:
            draw.text((x + 15, y_offset), line, fill=TEXT_COLOR, font=get_font(12))
            y_offset += 20
            
    img.save(os.path.join(OUTPUT_DIR, "infographic_learning_process.png"))

def create_info_photosynthesis():
    img = Image.new("RGB", (850, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    draw.text((40, 30), "Photosynthesis Process Model", fill=TEXT_COLOR, font=get_font(22, bold=True))
    
    draw.ellipse([60, 100, 160, 200], fill=(255, 235, 120), outline=(255, 170, 0), width=2)
    draw.text((95, 140), "Sunlight", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.polygon([(360, 520), (440, 520), (420, 440), (380, 440)], fill=SHAPE_ORANGE, outline=BORDER_COLOR, width=2)
    draw.line([(400, 440), (400, 250)], fill=(80, 180, 80), width=8)
    draw.ellipse([340, 320, 400, 360], fill=SHAPE_GREEN, outline=(50, 120, 50), width=2)
    draw.ellipse([400, 280, 460, 320], fill=SHAPE_GREEN, outline=(50, 120, 50), width=2)
    
    draw_arrow(draw, (220, 260), (360, 310), width=3)
    draw_label(draw, "CO2 (Carbon Dioxide)", 180, 240, SHAPE_RED)
    
    draw_arrow(draw, (300, 480), (380, 460), width=3)
    draw_label(draw, "H2O (Water)", 250, 490, SHAPE_BLUE)
    
    draw_arrow(draw, (440, 290), (620, 240), width=3)
    draw_label(draw, "O2 (Oxygen)", 650, 220, SHAPE_GREEN)
    
    draw_arrow(draw, (410, 360), (620, 420), width=3)
    draw_label(draw, "C6H12O6 (Glucose / Sugar)", 660, 440, SHAPE_PURPLE)
    
    draw_arrow(draw, (150, 180), (350, 280), fill=(255, 170, 0), width=3)
    
    img.save(os.path.join(OUTPUT_DIR, "infographic_photosynthesis.png"))

def create_info_water_conservation():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    draw.text((40, 30), "Water Conservation Strategies", fill=TEXT_COLOR, font=get_font(22, bold=True))
    
    draw.polygon([(400, 180), (460, 280), (460, 380), (340, 380), (340, 280)], fill=(224, 242, 254), outline=(100, 150, 250), width=3)
    draw.ellipse([340, 280, 460, 400], fill=(224, 242, 254), outline=None)
    draw.text((375, 330), "Save\nWater", fill=TEXT_COLOR, font=get_font(15, bold=True))
    
    strategies = [
        ("Fix Leaks Quickly", (260, 170), (140, 140)),
        ("Install Rain Barrels", (540, 170), (660, 140)),
        ("Take Short Showers", (240, 370), (120, 400)),
        ("Turn Off Active Taps", (560, 370), (680, 400))
    ]
    
    for title, line_start, box_pos in strategies:
        draw_label(draw, title, box_pos[0], box_pos[1], SHAPE_BLUE)
        draw.line([line_start, box_pos], fill=BORDER_COLOR, width=2)
        
    img.save(os.path.join(OUTPUT_DIR, "infographic_water_conservation.png"))

# ==========================================
# CATEGORY 5: DIFFICULT CASES (MIXED CONTENT / SCAN / PAGES)
# ==========================================

def create_diff_scan_1():
    img = Image.new("RGB", (800, 1000), (248, 246, 240))
    draw = ImageDraw.Draw(img)
    
    draw.line([(50, 0), (50, 1000)], fill=(230, 210, 210), width=1)
    
    draw.text((70, 40), "CHAPTER 3: CELLULAR BIOLOGY", fill=TEXT_COLOR, font=get_font(12))
    draw.text((70, 60), "3.1 The Animal Cell Membrane structure and function", fill=TEXT_COLOR, font=get_font(16, bold=True))
    draw.line([(70, 90), (730, 90)], fill=TEXT_COLOR, width=2)
    
    para1 = "The cell membrane (also known as the plasma membrane) is a biological membrane that separates the interior of all cells from the outside environment. It consists of a lipid bilayer with embedded proteins. The primary function is to protect the cell from its surroundings. It is selectively permeable to ions and organic molecules, controlling the movement of substances in and out of cells."
    words = para1.split()
    y = 110
    lines = []
    line = []
    for w in words:
        if len(" ".join(line + [w])) > 80:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    lines.append(" ".join(line))
    
    for l in lines:
        draw.text((70, y), l, fill=TEXT_COLOR, font=get_font(12))
        y += 20
        
    diagram_y = 300
    draw.rectangle([120, diagram_y, 680, diagram_y + 300], fill=(255, 255, 255), outline=BORDER_COLOR, width=1)
    
    for x in range(150, 650, 30):
        draw.ellipse([x - 10, diagram_y + 60, x + 10, diagram_y + 80], fill=SHAPE_ORANGE, outline=BORDER_COLOR)
        draw.line([(x, diagram_y + 80), (x - 5, diagram_y + 110)], fill=BORDER_COLOR, width=2)
        draw.line([(x, diagram_y + 80), (x + 5, diagram_y + 110)], fill=BORDER_COLOR, width=2)
        
        draw.ellipse([x - 10, diagram_y + 140, x + 10, diagram_y + 160], fill=SHAPE_ORANGE, outline=BORDER_COLOR)
        draw.line([(x, diagram_y + 140), (x - 5, diagram_y + 110)], fill=BORDER_COLOR, width=2)
        draw.line([(x, diagram_y + 140), (x + 5, diagram_y + 110)], fill=BORDER_COLOR, width=2)
        
    draw.rounded_rectangle([360, diagram_y + 40, 440, diagram_y + 180], radius=10, fill=SHAPE_PURPLE, outline=BORDER_COLOR)
    draw.text((375, diagram_y + 100), "Channel", fill=TEXT_COLOR, font=get_font(11, bold=True))
    
    draw_label(draw, "Hydrophilic Head", 220, diagram_y + 20, SHAPE_BLUE)
    draw_arrow(draw, (220, diagram_y + 35), (250, diagram_y + 65))
    
    draw_label(draw, "Hydrophobic Tail", 220, diagram_y + 260, SHAPE_BLUE)
    draw_arrow(draw, (220, diagram_y + 245), (250, diagram_y + 115))
    
    draw_label(draw, "Integral Protein", 550, diagram_y + 100, SHAPE_BLUE)
    draw_arrow(draw, (490, diagram_y + 100), (435, diagram_y + 100))
    
    para2 = "Active transport processes require the cell to expend energy to move substances against their concentration gradient. These processes are highly selective and rely on specialized transmembrane carrier proteins which act as pumps powered by ATP hydrolysis."
    y = 630
    words = para2.split()
    lines = []
    line = []
    for w in words:
        if len(" ".join(line + [w])) > 80:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    lines.append(" ".join(line))
    for l in lines:
        draw.text((70, y), l, fill=TEXT_COLOR, font=get_font(12))
        y += 20
        
    draw.text((70, 940), "Page 74 | Cell Biology Basics", fill=TEXT_COLOR, font=get_font(11))
    img.save(os.path.join(OUTPUT_DIR, "difficult_textbook_scan_1.png"))

def create_diff_scan_2():
    img = Image.new("RGB", (800, 1000), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    
    draw.text((60, 40), "SECTION 4: WAVE OPTICS", fill=TEXT_COLOR, font=get_font(18, bold=True))
    draw.line([(60, 70), (740, 70)], fill=TEXT_COLOR, width=2)
    
    col1_x = 60
    col1_y = 100
    
    col1_text = [
        "In optics, dispersion is the phenomenon",
        "in which the phase velocity of a wave",
        "depends on its frequency. Media having",
        "this common property are termed",
        "dispersive media.",
        "",
        "The relation governing index of",
        "refraction n as a function of wavelength",
        "lambda is given by the Cauchy equation:",
        "",
        "   n(L) = A + B / (L^2) + C / (L^4)",
        "",
        "Where A, B, and C are constant parameters",
        "fitted to experimental measurements.",
        "Similarly, Snell's Law states:",
        "",
        "   n_1 * sin(t_1) = n_2 * sin(t_2)"
    ]
    
    for line in col1_text:
        draw.text((col1_x, col1_y), line, fill=TEXT_COLOR, font=get_font(12))
        col1_y += 24
        
    col2_x = 420
    prism_pts = [(560, 150), (680, 350), (440, 350)]
    draw.polygon(prism_pts, fill=(240, 245, 255), outline=BORDER_COLOR, width=3)
    draw.text((535, 280), "Glass Prism\n(n > 1)", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw_arrow(draw, (380, 290), (490, 270), fill=(220, 220, 0), width=4)
    draw_label(draw, "Incident White Light", 400, 340, SHAPE_ORANGE)
    
    colors = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (75, 0, 130)]
    for i, color in enumerate(colors):
        y_offset = i * 4
        draw.line([(490, 270), (560, 275 + y_offset)], fill=color, width=2)
        draw_arrow(draw, (560, 275 + y_offset), (730, 290 + i*8), fill=color, width=2)
        
    draw_label(draw, "Refracted Spectrum (Rainbow)", 620, 400, SHAPE_BLUE)
    
    bottom_y = 500
    bottom_text = [
        "Newton's experiments in 1666 showed that white light is not simple but a compound of all spectral",
        "colors. By placing a second inverted prism in the path of the dispersed rays, he successfully",
        "recombined the colors back into a single ray of white light, proving color is intrinsic to light itself."
    ]
    for line in bottom_text:
        draw.text((60, bottom_y), line, fill=TEXT_COLOR, font=get_font(12))
        bottom_y += 24
        
    draw.text((60, 950), "Optics Lab Handbook | Dispersion Analysis", fill=TEXT_COLOR, font=get_font(11))
    img.save(os.path.join(OUTPUT_DIR, "difficult_textbook_scan_2.png"))

def create_diff_mixed_1():
    img = Image.new("RGB", (800, 900), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    draw.text((40, 30), "General Physics Quiz - Semester 1", fill=TEXT_COLOR, font=get_font(18, bold=True))
    draw.line([(40, 60), (760, 60)], fill=TEXT_COLOR, width=1)
    
    draw.text((40, 90), "Question 4: Identify the correct mathematical relation governing the loop current (I)", fill=TEXT_COLOR, font=get_font(14, bold=True))
    draw.text((40, 115), "in the schematic circuit diagram shown below. Assume all parts are ideal.", fill=TEXT_COLOR, font=get_font(12))
    
    draw.rectangle([150, 160, 650, 480], fill=(255, 255, 255), outline=BORDER_COLOR, width=2)
    draw.rectangle([250, 240, 550, 400], fill=None, outline=BORDER_COLOR, width=3)
    
    draw.rectangle([370, 230, 430, 250], fill=(255, 255, 255), outline=None)
    res_pts = [(370, 240), (380, 230), (390, 250), (400, 230), (410, 250), (420, 230), (430, 240)]
    draw.line(res_pts, fill=TEXT_COLOR, width=3)
    draw_label(draw, "Resistor (R = 10 Ohms)", 400, 200, SHAPE_BLUE)
    
    draw.rectangle([240, 300, 260, 340], fill=(255, 255, 255), outline=None)
    draw.line([(235, 320), (265, 320)], fill=TEXT_COLOR, width=3)
    draw.line([(245, 310), (245, 330)], fill=TEXT_COLOR, width=6)
    draw_label(draw, "Source (V = 5V)", 160, 320, SHAPE_ORANGE)
    
    draw.arc([350, 290, 450, 370], start=45, end=315, fill=ARROW_COLOR, width=3)
    draw_arrow(draw, (430, 345), (428, 355), width=3)
    draw.text((395, 320), "I", fill=TEXT_COLOR, font=get_font(14, bold=True))
    
    draw.text((40, 520), "Options:", fill=TEXT_COLOR, font=get_font(14, bold=True))
    
    options = [
        ("A)  I = V * R  (50 Amps)", 80, 560, SHAPE_GREEN),
        ("B)  I = V / R  (0.5 Amps)", 420, 560, SHAPE_GREEN),
        ("C)  I = R / V  (2 Amps)", 80, 620, SHAPE_GREEN),
        ("D)  I = V - R  (-5 Amps)", 420, 620, SHAPE_GREEN)
    ]
    
    for text, x, y, color in options:
        draw_label(draw, text, x + 150, y + 20, color)
        
    draw.rounded_rectangle([40, 700, 760, 830], radius=8, fill=SHAPE_PURPLE, outline=BORDER_COLOR, width=1)
    draw.text((60, 720), "Submitting Instructions:", fill=TEXT_COLOR, font=get_font(13, bold=True))
    draw.text((60, 750), "- Mark your answers on the digital OMR sheet provided by the examiner.", fill=TEXT_COLOR, font=get_font(12))
    draw.text((60, 775), "- Double check all calculations. No marks will be awarded for incorrect reasoning.", fill=TEXT_COLOR, font=get_font(12))
    
    img.save(os.path.join(OUTPUT_DIR, "difficult_mixed_content_1.png"))

def create_diff_mixed_2():
    img = Image.new("RGB", (900, 700), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    draw.text((40, 30), "Timeline of Modern Computing (1940 - 2000)", fill=TEXT_COLOR, font=get_font(20, bold=True))
    draw.line([(40, 65), (860, 65)], fill=TEXT_COLOR, width=2)
    
    axis_y = 250
    draw.line([(50, axis_y), (850, axis_y)], fill=TEXT_COLOR, width=4)
    
    nodes = [
        ("1946", "ENIAC", "First general purpose programmable computer.", 100, "up", SHAPE_RED),
        ("1971", "Microprocessor", "Intel 4004 changes computing landscape.", 280, "down", SHAPE_ORANGE),
        ("1981", "IBM PC", "Brings computers directly into corporate offices.", 460, "up", SHAPE_BLUE),
        ("1991", "World Wide Web", "Tim Berners-Lee invents HTTP/HTML sharing.", 640, "down", SHAPE_GREEN),
        ("1998", "Google Search", "Index-based search engine organizes web data.", 800, "up", SHAPE_PURPLE)
    ]
    
    for date, name, desc, x, direction, color in nodes:
        draw.line([(x, axis_y - 10), (x, axis_y + 10)], fill=TEXT_COLOR, width=3)
        draw.text((x - 20, axis_y + 15 if direction == "down" else axis_y - 30), date, fill=TEXT_COLOR, font=get_font(12, bold=True))
        
        box_y = axis_y + 60 if direction == "down" else axis_y - 180
        draw.rounded_rectangle([x - 70, box_y, x + 70, box_y + 90], radius=8, fill=color, outline=BORDER_COLOR, width=1)
        draw.text((x - 60, box_y + 10), name, fill=TEXT_COLOR, font=get_font(11, bold=True))
        
        words = desc.split()
        lines = []
        line = []
        for w in words:
            if len(" ".join(line + [w])) > 18:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        lines.append(" ".join(line))
        
        dy = box_y + 35
        for l in lines:
            draw.text((x - 60, dy), l, fill=TEXT_COLOR, font=get_font(9))
            dy += 15
            
        if direction == "down":
            draw_arrow(draw, (x, box_y), (x, axis_y + 12))
        else:
            draw_arrow(draw, (x, box_y + 90), (x, axis_y - 12))
            
    desc_para = "This timeline details the evolution of modern hardware and networking technologies. Within just 60 years, computers shrank from warehouse-sized scientific calculators (like ENIAC) into highly portable desktop devices capable of index-searching the entirety of human knowledge in milliseconds."
    
    y = 520
    words = desc_para.split()
    lines = []
    line = []
    for w in words:
        if len(" ".join(line + [w])) > 100:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    lines.append(" ".join(line))
    
    for l in lines:
        draw.text((80, y), l, fill=TEXT_COLOR, font=get_font(12))
        y += 22
        
    img.save(os.path.join(OUTPUT_DIR, "difficult_mixed_content_2.png"))

# ==========================================
# EXTRA SAMPLES TO FULFILL THE "AT LEAST 20" REQUIREMENT
# ==========================================

def create_extra_pendulum():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), "Simple Pendulum System", fill=TEXT_COLOR, font=get_font(20, bold=True))
    
    draw.line([(250, 100), (550, 100)], fill=TEXT_COLOR, width=4)
    for x in range(260, 550, 15):
        draw.line([(x, 100), (x + 10, 85)], fill=BORDER_COLOR, width=2)
    draw_label(draw, "Rigid Support", 400, 60, SHAPE_BLUE)
    
    draw.circle((400, 100), 5, fill=TEXT_COLOR)
    
    for y in range(110, 450, 15):
        draw.line([(400, y), (400, y + 8)], fill=ARROW_COLOR, width=2)
        
    angle_rad = math.radians(30)
    bob_x = int(400 + 300 * math.sin(angle_rad))
    bob_y = int(100 + 300 * math.cos(angle_rad))
    draw.line([(400, 100), (bob_x, bob_y)], fill=TEXT_COLOR, width=3)
    draw_label(draw, "Tension Force (T)", (400 + bob_x)//2 - 20, (100 + bob_y)//2, SHAPE_PURPLE)
    
    draw.ellipse([bob_x - 25, bob_y - 25, bob_x + 25, bob_y + 25], fill=SHAPE_RED, outline=BORDER_COLOR, width=2)
    draw_label(draw, "Bob (Mass m)", bob_x + 70, bob_y, SHAPE_RED)
    
    draw_arrow(draw, (bob_x, bob_y + 25), (bob_x, bob_y + 100), width=3)
    draw_label(draw, "Gravity (F_g = m * g)", bob_x, bob_y + 120, SHAPE_ORANGE)
    
    draw.arc([350, 50, 450, 150], start=60, end=90, fill=TEXT_COLOR, width=2)
    draw.text((385, 160), "theta", fill=TEXT_COLOR, font=get_font(12, bold=True))
    draw_label(draw, "Angle of Displacement", 280, 160, SHAPE_GREEN)
    
    img.save(os.path.join(OUTPUT_DIR, "physics_pendulum_system.png"))

def create_extra_chemistry():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), "Periodic Table Group 1: Alkali Metals", fill=TEXT_COLOR, font=get_font(20, bold=True))
    
    metals = [
        ("Li", "Lithium", "Atomic #3", SHAPE_RED, 80),
        ("Na", "Sodium", "Atomic #11", SHAPE_ORANGE, 220),
        ("K", "Potassium", "Atomic #19", SHAPE_BLUE, 360),
        ("Rb", "Rubidium", "Atomic #37", SHAPE_GREEN, 50),
        ("Cs", "Cesium", "Atomic #55", SHAPE_PURPLE, 190)
    ]
    
    grid_coords = [
        (80, 120), (310, 120), (540, 120),
        (80, 330), (310, 330)
    ]
    
    for idx, (sym, name, desc, color, x_pos) in enumerate(metals):
        x, y = grid_coords[idx]
        draw.rounded_rectangle([x, y, x + 180, y + 160], radius=10, fill=color, outline=BORDER_COLOR, width=2)
        draw.text((x + 20, y + 20), sym, fill=TEXT_COLOR, font=get_font(32, bold=True))
        draw.text((x + 20, y + 75), name, fill=TEXT_COLOR, font=get_font(16, bold=True))
        draw.text((x + 20, y + 110), desc, fill=TEXT_COLOR, font=get_font(12))
        
    draw_label(draw, "Extremely reactive with water. Stored in oil.", 590, 410, SHAPE_ORANGE)
    
    img.save(os.path.join(OUTPUT_DIR, "infographic_chemistry_alkali_metals.png"))

def create_extra_study_methods():
    img = Image.new("RGB", (850, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), "Scientific Study Methods", fill=TEXT_COLOR, font=get_font(22, bold=True))
    
    methods = [
        ("Active Recall", "Testing yourself instead of passively reading textbooks.", SHAPE_RED, 80, 120),
        ("Spaced Repetition", "Reviewing material at increasing intervals over time.", SHAPE_ORANGE, 450, 120),
        ("Feynman Technique", "Teaching a concept to a child to verify your own understanding.", SHAPE_BLUE, 80, 340),
        ("Pomodoro Technique", "Studying in focused 25-minute blocks with 5-minute breaks.", SHAPE_GREEN, 450, 340)
    ]
    
    for title, desc, color, x, y in methods:
        draw.rounded_rectangle([x, y, x + 320, y + 160], radius=12, fill=color, outline=BORDER_COLOR, width=2)
        draw.text((x + 20, y + 20), title, fill=TEXT_COLOR, font=get_font(16, bold=True))
        
        words = desc.split()
        lines = []
        line = []
        for w in words:
            if len(" ".join(line + [w])) > 32:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        lines.append(" ".join(line))
        
        dy = y + 60
        for l in lines:
            draw.text((x + 20, dy), l, fill=TEXT_COLOR, font=get_font(12))
            dy += 20
            
    img.save(os.path.join(OUTPUT_DIR, "infographic_study_methods.png"))

def create_extra_nitrogen():
    img = Image.new("RGB", (800, 600), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), "The Nitrogen Cycle", fill=TEXT_COLOR, font=get_font(20, bold=True))
    
    draw_label(draw, "Atmospheric Nitrogen (N2)", 400, 100, SHAPE_BLUE)
    
    draw_label(draw, "Nitrogen Fixation\n(Bacteria in Soil)", 180, 260, SHAPE_ORANGE)
    draw_arrow(draw, (330, 120), (200, 220), width=3)
    
    draw_label(draw, "Nitrification\n(Ammonium -> Nitrates)", 400, 420, SHAPE_GREEN)
    draw_arrow(draw, (200, 310), (330, 400), width=3)
    
    draw_label(draw, "Denitrification\n(Nitrates back to N2)", 620, 260, SHAPE_RED)
    draw_arrow(draw, (470, 400), (600, 310), width=3)
    draw_arrow(draw, (600, 220), (470, 120), width=3)
    
    img.save(os.path.join(OUTPUT_DIR, "infographic_nitrogen_cycle.png"))

def create_extra_history():
    img = Image.new("RGB", (800, 1000), (250, 246, 238))
    draw = ImageDraw.Draw(img)
    
    draw.text((60, 40), "THE RENAISSANCE PERIOD (1400 - 1600)", fill=TEXT_COLOR, font=get_font(16, bold=True))
    draw.line([(60, 70), (740, 70)], fill=TEXT_COLOR, width=2)
    
    text = (
        "The Renaissance was a fervent period of European cultural, artistic, political, and scientific rebirth "
        "following the Middle Ages. Promoted by a rediscovery of classical philosophy, literature, and art, "
        "it produced some of the greatest minds and artworks in human history. Key figures like Leonardo "
        "da Vinci, Michelangelo, and Galileo Galilei fundamentally reshaped human understanding."
    )
    words = text.split()
    y = 90
    lines = []
    line = []
    for w in words:
        if len(" ".join(line + [w])) > 80:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    lines.append(" ".join(line))
    for l in lines:
        draw.text((60, y), l, fill=TEXT_COLOR, font=get_font(12))
        y += 22
        
    timeline_y = 350
    draw.line([(100, timeline_y), (700, timeline_y)], fill=TEXT_COLOR, width=3)
    
    events = [
        ("1440", "Printing Press", 120, "down", SHAPE_RED),
        ("1492", "New World Voyage", 280, "up", SHAPE_ORANGE),
        ("1503", "Mona Lisa Painted", 440, "down", SHAPE_BLUE),
        ("1543", "Copernican Model", 600, "up", SHAPE_GREEN)
    ]
    
    for date, title, x, dir_pos, color in events:
        draw.line([(x, timeline_y - 10), (x, timeline_y + 10)], fill=TEXT_COLOR, width=2)
        draw.text((x - 15, timeline_y + 15 if dir_pos == "down" else timeline_y - 30), date, fill=TEXT_COLOR, font=get_font(11, bold=True))
        
        box_y = timeline_y + 45 if dir_pos == "down" else timeline_y - 145
        draw.rounded_rectangle([x - 65, box_y, x + 65, box_y + 70], radius=6, fill=color, outline=BORDER_COLOR, width=1)
        draw.text((x - 55, box_y + 15), title, fill=TEXT_COLOR, font=get_font(9, bold=True))
        
        if dir_pos == "down":
            draw_arrow(draw, (x, box_y), (x, timeline_y + 12))
        else:
            draw_arrow(draw, (x, box_y + 70), (x, timeline_y - 12))
            
    para2 = (
        "The impact of the movable-type printing press, pioneered by Johannes Gutenberg around 1440, "
        "cannot be overstated. It democratized information access, accelerated the spread of scientific "
        "knowledge, and laid the direct intellectual groundwork for the Protestant Reformation and the Scientific Revolution."
    )
    y = 650
    words = para2.split()
    lines = []
    line = []
    for w in words:
        if len(" ".join(line + [w])) > 80:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    lines.append(" ".join(line))
    for l in lines:
        draw.text((60, y), l, fill=TEXT_COLOR, font=get_font(12))
        y += 22
        
    img.save(os.path.join(OUTPUT_DIR, "difficult_mixed_content_3.png"))

def create_diff_scan_3():
    img = Image.new("RGB", (800, 1000), (252, 250, 245))
    draw = ImageDraw.Draw(img)
    
    draw.text((70, 50), "CHAPTER 8: ELECTRICITY & MAGNETISM", fill=TEXT_COLOR, font=get_font(14, bold=True))
    draw.text((70, 75), "8.3 Electromagnetic Induction and Faraday's Law", fill=TEXT_COLOR, font=get_font(18, bold=True))
    draw.line([(70, 105), (730, 105)], fill=TEXT_COLOR, width=2)
    
    text = (
        "Faraday's Law of Induction states that a changing magnetic flux through a loop of wire induces an "
        "electromotive force (EMF) in the wire. This fundamental principle is the basis for electric generators, "
        "transformers, and induction motors. The mathematical formula is represented as:"
    )
    words = text.split()
    y = 125
    lines = []
    line = []
    for w in words:
        if len(" ".join(line + [w])) > 80:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    lines.append(" ".join(line))
    for l in lines:
        draw.text((70, y), l, fill=TEXT_COLOR, font=get_font(12))
        y += 20
        
    formula_y = 220
    draw.rounded_rectangle([180, formula_y, 620, formula_y + 80], radius=8, fill=(255, 255, 255), outline=BORDER_COLOR, width=1)
    draw.text((250, formula_y + 25), "EMF = - d(Phi_B) / dt", fill=TEXT_COLOR, font=get_font(20, bold=True))
    draw_label(draw, "Faraday's Law Formula", 400, formula_y + 10, SHAPE_ORANGE)
    
    diag_y = 330
    draw.rectangle([120, diag_y, 680, diag_y + 350], fill=(255, 255, 255), outline=BORDER_COLOR, width=1)
    draw.text((140, diag_y + 20), "Figure 8.12: Moving magnet inducing current in coil", fill=TEXT_COLOR, font=get_font(12, bold=True))
    
    draw.rectangle([180, diag_y + 120, 260, diag_y + 180], fill=(240, 100, 100), outline=BORDER_COLOR, width=2)
    draw.text((215, diag_y + 140), "N", fill=(255, 255, 255), font=get_font(16, bold=True))
    draw.rectangle([260, diag_y + 120, 340, diag_y + 180], fill=(100, 100, 240), outline=BORDER_COLOR, width=2)
    draw.text((295, diag_y + 140), "S", fill=(255, 255, 255), font=get_font(16, bold=True))
    draw_label(draw, "Bar Magnet", 260, diag_y + 80, SHAPE_BLUE)
    
    draw_arrow(draw, (360, diag_y + 150), (450, diag_y + 150), width=4, arrow_len=15)
    draw_label(draw, "Direction of Motion", 400, diag_y + 200, SHAPE_ORANGE)
    
    for c in range(3):
        draw.ellipse([480 + c*20, diag_y + 100, 540 + c*20, diag_y + 200], fill=None, outline=BORDER_COLOR, width=3)
    draw_label(draw, "Coil of Wire", 560, diag_y + 70, SHAPE_GREEN)
    
    draw.circle((540, diag_y + 270), 30, fill=SHAPE_BLUE, outline=BORDER_COLOR, width=2)
    draw.line([(540, diag_y + 270), (520, diag_y + 250)], fill=TEXT_COLOR, width=3)
    draw.text((535, diag_y + 285), "G", fill=TEXT_COLOR, font=get_font(14, bold=True))
    draw_label(draw, "Deflected Galvanometer", 540, diag_y + 320, SHAPE_PURPLE)
    
    draw.line([(480, diag_y + 150), (480, diag_y + 270), (510, diag_y + 270)], fill=BORDER_COLOR, width=2)
    draw.line([(580, diag_y + 150), (600, diag_y + 150), (600, diag_y + 270), (570, diag_y + 270)], fill=BORDER_COLOR, width=2)
    
    para3 = (
        "As the bar magnet moves into the coil, the magnetic flux increases, prompting a counter-current "
        "whose magnetic field opposes the change (Lenz's Law). This is registered by the galvanometer needle's deflection."
    )
    y = 700
    words = para3.split()
    lines = []
    line = []
    for w in words:
        if len(" ".join(line + [w])) > 80:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    lines.append(" ".join(line))
    for l in lines:
        draw.text((70, y), l, fill=TEXT_COLOR, font=get_font(12))
        y += 20
        
    img.save(os.path.join(OUTPUT_DIR, "difficult_textbook_scan_3.png"))

def create_diff_scan_4():
    img = Image.new("RGB", (800, 1000), (245, 248, 245))
    draw = ImageDraw.Draw(img)
    
    draw.text((70, 50), "ORGANIC CHEMISTRY: CHEMICAL EQUILIBRIUM", fill=TEXT_COLOR, font=get_font(14, bold=True))
    draw.line([(70, 75), (730, 75)], fill=TEXT_COLOR, width=2)
    
    text = (
        "Chemical equilibrium is the state in which both reactants and products are present in concentrations "
        "which have no further tendency to change with time, so that there is no observable change in the properties "
        "of the system. The state results when the forward reaction proceeds at the same rate as the reverse reaction."
    )
    y = 90
    for l in [text[:100], text[100:200], text[200:]]:
        draw.text((70, y), l.strip(), fill=TEXT_COLOR, font=get_font(12))
        y += 20
        
    diag_y = 200
    draw.rectangle([150, diag_y, 650, diag_y + 280], fill=(255, 255, 255), outline=BORDER_COLOR, width=1)
    
    draw.line([(250, diag_y + 60), (250, diag_y + 220), (450, diag_y + 220), (450, diag_y + 60)], fill=TEXT_COLOR, width=4)
    draw.rectangle([254, diag_y + 130, 446, diag_y + 218], fill=(224, 242, 254), outline=None)
    draw_label(draw, "Reactants (A + B)", 350, diag_y + 160, SHAPE_BLUE)
    
    draw.rectangle([460, diag_y + 50, 560, diag_y + 110], fill=SHAPE_ORANGE, outline=BORDER_COLOR, width=2)
    draw_arrow(draw, (480, diag_y + 80), (410, diag_y + 110), width=3)
    draw_label(draw, "Adding Catalyst", 570, diag_y + 130, SHAPE_ORANGE)
    
    form_y = 520
    draw.rounded_rectangle([150, form_y, 650, form_y + 120], radius=8, fill=SHAPE_GREEN, outline=BORDER_COLOR, width=1)
    draw.text((240, form_y + 25), "Rate_forward = Rate_reverse", fill=TEXT_COLOR, font=get_font(18, bold=True))
    draw.text((240, form_y + 65), "K_eq = [Products] / [Reactants]", fill=TEXT_COLOR, font=get_font(18, bold=True))
    draw_label(draw, "Equilibrium Constant Expression", 400, form_y + 10, SHAPE_BLUE)
    
    draw.rounded_rectangle([70, 680, 730, 850], radius=6, fill=SHAPE_RED, outline=BORDER_COLOR, width=1)
    draw.text((90, 700), "Safety Warnings:", fill=TEXT_COLOR, font=get_font(14, bold=True))
    draw.text((90, 730), "1. Wear safety goggles and gloves at all times during this equilibrium experiment.", fill=TEXT_COLOR, font=get_font(12))
    draw.text((90, 760), "2. The reaction is exothermic. Add reagents slowly to prevent temperature spikes.", fill=TEXT_COLOR, font=get_font(12))
    draw.text((90, 790), "3. Dispose of waste in the designated organic halogenated waste container.", fill=TEXT_COLOR, font=get_font(12))
    
    img.save(os.path.join(OUTPUT_DIR, "difficult_textbook_scan_4.png"))

# ==========================================
# MASTER GENERATOR LIST CALL
# ==========================================

def run_all_generators():
    print(f"Generating 20 test datasets in: {OUTPUT_DIR}...")
    
    create_bio_digestive()
    create_bio_heart()
    create_bio_plant_cell()
    create_bio_animal_cell()
    print("   [OK] Biology diagrams generated.")
    
    create_phys_solar()
    create_phys_water_cycle()
    create_phys_circuit()
    create_phys_reflection()
    print("   [OK] Physics diagrams generated.")
    
    create_flow_admission()
    create_flow_sdlc()
    create_flow_order()
    create_flow_temperature()
    print("   [OK] Flowcharts generated.")
    
    create_info_parts_of_speech()
    create_info_learning_process()
    create_info_photosynthesis()
    create_info_water_conservation()
    print("   [OK] Infographics generated.")
    
    create_diff_scan_1()
    create_diff_scan_2()
    create_diff_scan_3()
    create_diff_scan_4()
    create_diff_mixed_1()
    create_diff_mixed_2()
    create_extra_pendulum()
    create_extra_chemistry()
    create_extra_study_methods()
    create_extra_nitrogen()
    create_extra_history()
    print("   [OK] Difficult cases & extra validation images generated.")
    
    print(f"\nVerification: Total files generated in test_data: {len(os.listdir(OUTPUT_DIR))}")

if __name__ == "__main__":
    run_all_generators()

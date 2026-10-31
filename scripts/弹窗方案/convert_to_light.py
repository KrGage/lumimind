"""
将四个学术写作 Skill 文件夹中的"计算机科学"相关内容
替换为"光学与人因工程"领域，输出到 _light 后缀的新文件夹。
只替换明确提到计算机/CS/AI特定概念的部分，通用学术术语保持不变。
"""
import os, shutil

BASE = r"d:\研究生\项目内容\杂事\光源智能体设计大赛\项目主体"

# ── 替换规则 ──────────────────────────────────────
# 只替换明确提到"计算机"或 CS 特有概念的部分
REPLACEMENTS = [
    # === 期刊名称 ===
    ("《计算机学报》、《软件学报》、《自动化学报》", "《光学学报》、《光子学报》、《照明工程学报》"),
    ("《计算机学报》、《软件学报》", "《光学学报》、《光子学报》"),
    ("《计算机学报》", "《光学学报》"),
    ("《软件学报》", "《光子学报》"),
    ("《自动化学报》", "《照明工程学报》"),

    # === 英文会议/期刊名称 ===
    ("NeurIPS/CVPR/ICML/ICLR", "CIE Session/ISOT/ICO/IEC"),
    ("NeurIPS/ICML/ICLR/ACL", "CIE Session/ISOT/ICO/IEC"),
    ("ICML/ICLR/NeurIPS/ACL", "CIE Session/ISOT/ICO/IEC"),
    ("NeurIPS/CVPR/ICML", "CIE Session/ISOT/ICO"),
    ("NeurIPS/ICML/ICLR", "CIE Session/ISOT/ICO"),
    ("ICML/ICLR", "CIE Session/ISOT"),
    ("NeurIPS, ICLR, ICML", "CIE Session, ISOT, ICO"),
    ("ACL, NeurIPS", "CIE Session, ISOT"),
    ("CVPR, NeurIPS", "CIE Session, ISOT"),
    ("NeurIPS", "CIE Session"),
    ("ICML", "ISOT"),
    ("ICLR", "ICO"),
    ("CVPR", "ISOT"),

    # === 领域名称（明确提到"计算机"的） ===
    ("计算机视觉与人工智能领域", "光学与人因工程领域"),
    ("计算机视觉与人工智能", "光学与人因工程"),
    ("计算机视觉", "光学与视觉科学"),
    ("计算机科学领域的", "光学与人因工程领域的"),
    ("计算机科学领域", "光学与人因工程领域"),
    ("计算机科学", "光学与人因工程"),
    ("计算机系", "光电系"),

    # === CS 特有技术术语（仅在明确是CS上下文时替换） ===
    ("Transformer, CNN, Few-shot", "CIE 1931色度图, 光谱功率分布, 明视觉"),
    ("深度学习、推荐系统、自然语言处理", "光环境设计、人因工程、节律照明"),
    ("深度学习、CNN、图像分割（U-Net、FCN等）", "光学设计、照明工程、光谱分析（积分球、分布光度计等）"),
    ("机器学习+推荐系统", "光学设计+人因工程"),
    ("机器学习和推荐系统", "光学与人因工程"),
    ("推荐系统", "照明优化"),
    ("图神经网络在推荐中的应用", "光谱优化在照明中的应用"),
    ("冷启动问题、跨域推荐、个性化推荐的可解释性", "视觉舒适度优化、节律健康照明、个性化光环境设计"),
    ("推荐系统、信息检索、用户建模", "照明设计、人因工程、视觉工效学"),
    ("基于深度学习的医学图像分割", "基于光谱调制的健康照明优化"),
    ("医学图像处理、计算机视觉", "光环境评价、人因照明"),
    ("肺部病变分割、肿瘤分割、器官分割、血管分割等", "视觉疲劳缓解、节律调节、情绪改善、视觉绩效提升等"),

    # === README 英文示例中的 CS 内容 ===
    ("Computer Science and Technology, specializing in machine learning",
     "Optical Engineering and Technology, specializing in lighting design"),
    ("Deep learning, recommender systems, and natural language processing",
     "Lighting design, human factors engineering, and circadian lighting"),
    ("Deep learning fundamentals, Python programming, and reading",
     "Optical fundamentals, lighting simulation software, and reading"),
    ("How can recommender systems become more accurate",
     "How can lighting environments improve visual comfort and well-being"),
    ("Deep learning for recommendation, graph neural networks for recommendation",
     "Spectral optimization for lighting, circadian lighting design"),
    ("the cold-start problem, cross-domain recommendation, and explainable",
     "visual comfort optimization, circadian health lighting, and personalized"),
    ("personalized recommendation",
     "personalized lighting environment design"),
    ("Recommender systems, information retrieval, and user modeling",
     "Lighting design, human factors engineering, and visual ergonomics"),
    ("machine learning and recommender systems",
     "optics and human factors engineering"),
    ("machine learning and",
     "optics and"),

    # === 技术类选题中的 CS 示例 ===
    ("如AI、区块链、大数据等", "如LED照明、智能控制、光谱调制等"),

    # === 特定上下文中的 AI 指代 ===
    # 仅在"人工智能"作为学科名称出现时替换
    ("人工智能领域", "光学与照明技术领域"),
    ("人工智能在教育评估中的应用研究", "智能照明在人因工程中的应用研究"),
    ("Artificial Intelligence in Educational Assessment",
     "Intelligent Lighting in Human Factors Engineering"),
    ("Machine Learning for Automated Student Assessment",
     "Spectral Optimization for Health-Oriented Lighting"),
]

# ── 需要跳过的文件/扩展名（二进制文件） ──────────────────
SKIP_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.xlsx', '.xls', '.db', '.db-shm', '.db-wal'}
SKIP_FILES = {'LICENSE'}

def should_skip(filepath):
    _, ext = os.path.splitext(filepath)
    if ext.lower() in SKIP_EXTS:
        return True
    basename = os.path.basename(filepath)
    if basename in SKIP_FILES:
        return True
    return False

def process_file(src_path, dst_path):
    """读取源文件，执行替换，写入目标文件"""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if should_skip(src_path):
        shutil.copy2(src_path, dst_path)
        return
    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        shutil.copy2(src_path, dst_path)
        return

    original = content
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)

    # 报告哪些替换生效了
    if content != original:
        rel = os.path.relpath(dst_path, BASE)
        for old, new in REPLACEMENTS:
            if old in original:
                print(f"  [{rel}] '{old}' → '{new}'")

    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    folders = [
        'research-writing-cn',
        'research-writing-aux',
        'research-writing-en',
        'academic-ai-prompt-skill',
    ]
    for folder in folders:
        src_dir = os.path.join(BASE, folder)
        dst_dir = os.path.join(BASE, folder + '_light')
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
            print(f"[清理] 删除已有: {dst_dir}")
        print(f"\n{'='*50}")
        print(f"[处理] {folder} → {folder}_light")
        print(f"{'='*50}")
        count = 0
        for root, dirs, files in os.walk(src_dir):
            for fname in files:
                src_file = os.path.join(root, fname)
                rel_path = os.path.relpath(src_file, src_dir)
                dst_file = os.path.join(dst_dir, rel_path)
                process_file(src_file, dst_file)
                count += 1
        print(f"  共处理 {count} 个文件")
    print(f"\n{'='*50}")
    print("✅ 全部完成！")
    print(f"新文件夹位于：")
    for f in folders:
        print(f"  {f}_light")

if __name__ == '__main__':
    main()

import os
import re

scripts_dir = r'd:\研究生\项目内容\杂事\光源智能体设计大赛\项目主体\.qoder\skills\ppt-master\scripts'

fixed = []
for root, dirs, files in os.walk(scripts_dir):
    for fname in files:
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(root, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()

        has_union_syntax = bool(re.search(r'\| None', content))
        has_future = 'from __future__ import annotations' in content

        if has_union_syntax and not has_future:
            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('#!'):
                    insert_idx = i + 1
                    continue
                if line.startswith('"""') or line.startswith("'''"):
                    break
                if line.startswith('#') or line.strip() == '':
                    insert_idx = i + 1
                    continue
                break

            lines.insert(insert_idx, 'from __future__ import annotations')
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            fixed.append(os.path.relpath(fpath, scripts_dir))

print(f'Fixed {len(fixed)} files:')
for f in fixed:
    print(f'  {f}')

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai_agent"))

from agent_framework.checkpointer import (
    clear_history,
    get_history,
    is_persistent,
    save_messages,
)

sid = "test_conv_001"
clear_history(sid)  # 清除历史运行残留

save_messages(sid, [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你？"},
])
save_messages(sid, [{"role": "user", "content": "推荐一个阅读灯光"}])

h = get_history(sid)
print("persistent:", is_persistent())
print("history:", h)
assert len(h) == 3 and h[-1]["role"] == "user", "history mismatch"

clear_history(sid)
after = get_history(sid)
print("after clear:", after)
assert after == [], "clear failed"
print("MEMORY TEST OK")
